"""PocketWAM model (HF Transformers format; preset: 3m).

A mini World Action Model: one end-to-end model JOINTLY generates a future
world state (an image of the task's final state) and the current action via
a denoising flow with a SHARED timestep:

    pi(o_goal, a | o_t, c, q_t) = pi(o_goal | o_t, c) * pi(a | o_t, o_goal, q_t)
    L = MSE(x1_hat, x1) + MSE(a1_hat, a1)          [shared t ~ U(0,1)]

The current observation is teacher-forced as clean context; the noisy
[goal | action] chunk is denoised by one shared trunk. Heads predict the
CLEAN targets (x0-parameterization); the flow velocity is recovered
analytically at inference. K Euler steps at inference (K=1 = Flash single-step
mode). Outputs ONLY (final-state image, action).

Save/load: model.save_hf(dir, tokenizer) / PocketWAM.load_hf(dir, device), or
transformers.AutoModel/AutoTokenizer.from_pretrained(dir, trust_remote_code=True).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel, PreTrainedTokenizer)


# Size presets: CNN channels / stride-1 blocks / trunk dims / action MLP /
# goal conv channels / denoising steps
MODEL_PRESETS = {
    "3m":   dict(ch=(24, 48, 72, 96, 96),       extra=4,
                 d_model=192, n_layer=5, d_ff=640,
                 act=(512, 128), img_ch=(32, 64), denoise_steps=4),
}

# arch overrides (mirror config.ARCH_KEYS)
ARCH_OVERRIDE_KEYS = ("ch", "extra_blocks", "d_model", "n_layer", "d_ff",
                      "act_hidden", "img_ch", "denoise_steps", "action_delta_cond")

# token type ids for the shared trunk
T_VISION, T_TEXT, T_PROPRIO, T_GOAL, T_ACTION = 0, 1, 2, 3, 4


def _gn(ch: int) -> int:
    return 4 if ch == 16 else 8


class PocketWAMConfig(PretrainedConfig):
    """WAM model config (HF format, JSON-serializable)."""

    model_type = "pocketwam"

    def __init__(self, size: str = "3m", obs_res: int = 64,
                 world_extent: float = 1.25, proprio_dim: int = 6,
                 action_dim: int = 3, lang_len: int = 10, vocab_size: int = 64,
                 action_max: float = 0.15, action_delta_cond: bool = False,
                 **kwargs):
        self.size = size
        # unknown size labels fall back to the 3m base; yaml overrides apply below
        p = MODEL_PRESETS.get(size, MODEL_PRESETS["3m"])
        self.obs_res = obs_res
        self.world_extent = world_extent
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.lang_len = lang_len
        self.vocab_size = vocab_size
        self.action_max = action_max
        self.ch = list(p["ch"])
        self.extra_blocks = p["extra"]
        self.d_model = p["d_model"]
        self.n_layer = p["n_layer"]
        self.d_ff = p["d_ff"]
        self.act_hidden = list(p["act"])
        self.img_ch = list(p["img_ch"])
        self.denoise_steps = p["denoise_steps"]
        self.action_delta_cond = bool(action_delta_cond)
        # architecture overrides from the yaml `model:` section (kwargs)
        for k in ARCH_OVERRIDE_KEYS:
            if k in kwargs:
                v = kwargs.pop(k)
                setattr(self, k, list(v) if isinstance(v, (list, tuple)) else v)
        super().__init__(**kwargs)


class MaskedMHSA(nn.Module):
    """Multi-head self-attention with key padding mask (plain matmul + softmax)."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.h = n_head
        self.dh = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # (B, h, L, dh)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)   # (B, h, L, L)
        att = att.masked_fill(~pad[:, None, None, :], float("-inf"))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedMHSA(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pad)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionEncoder(nn.Module):
    """CNN visual tokenizer: 64x64x3 -> 8x8xC spatial feature map (= 64 grid tokens)."""

    def __init__(self, obs_res: int = 64, channels=(16, 32, 48, 64, 64),
                 extra_blocks: int = 0):
        super().__init__()
        layers = []
        in_ch = 3
        for i, ch in enumerate(channels):
            stride = 2 if i < 3 else 1
            layers += [nn.Conv2d(in_ch, ch, 3, stride, 1),
                       nn.GroupNorm(_gn(ch), ch), nn.SiLU()]
            in_ch = ch
        for _ in range(extra_blocks):
            layers += [nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                       nn.GroupNorm(_gn(in_ch), in_ch), nn.SiLU()]
        self.features = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)                   # (B, C, 8, 8)


def timestep_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Sinusoidal embedding of the flow timestep t in [0, 1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                                      device=t.device) / half)
    ang = t[:, None].float() * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class GoalEncoder(nn.Module):
    """Noisy final-state image (B,3,H,W) -> 8x8 token grid (B, 64, d_model)."""

    def __init__(self, d_model: int, img_ch=(32, 64)):
        super().__init__()
        c1, c2 = img_ch
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 3, 2, 1), nn.GroupNorm(_gn(c1), c1), nn.SiLU(),   # 64->32
            nn.Conv2d(c1, c2, 3, 2, 1), nn.GroupNorm(_gn(c2), c2), nn.SiLU(),  # 32->16
            nn.Conv2d(c2, d_model, 3, 2, 1), nn.SiLU(),                          # 16->8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)                                # (B, d, 8, 8)
        return f.flatten(2).transpose(1, 2)            # (B, 64, d)


class GoalDecoder(nn.Module):
    """[goal hidden states | vision hidden states] -> the residual DELTA
    (B, 3, H, W): x1_hat = obs + delta. Cross-reads the vision tokens so the
    video loss supervises them; the residual frees capacity for dynamics."""

    def __init__(self, d_model: int, img_ch=(32, 64), obs_res: int = 64):
        super().__init__()
        c1, c2 = img_ch
        self.grid = obs_res // 8
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(2 * d_model, c2, 3, 1, 1), nn.SiLU(),   # 8->16
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c1, 3, 1, 1), nn.SiLU(),            # 16->32
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, 3, 3, 1, 1),                         # 32->64, raw delta
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 128, d_model) = [goal states | vision states]."""
        B, _, D = tokens.shape
        f = tokens.transpose(1, 2)                      # (B, D, 128)
        f = f.reshape(B, 2, D, self.grid, self.grid).flatten(1, 2)  # (B, 2D, 8, 8)
        return self.net(f)                              # (B, 3, 64, 64)



class WAMTrunk(nn.Module):
    """Shared bidirectional trunk over interleaved tokens:
    [vision(64) | text(L) | proprio(1) | goal(64) | action(1)]; clean context
    tokens are teacher-forced, goal/action carry the timestep embedding."""

    def __init__(self, vocab_size: int, d_model: int, n_head: int,
                 n_layer: int, d_ff: int, lang_len: int, n_vis: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos_text = nn.Embedding(lang_len, d_model)
        self.pos_vis = nn.Embedding(n_vis, d_model)
        self.pos_goal = nn.Embedding(n_vis, d_model)
        self.type_emb = nn.Embedding(5, d_model)
        self.t_proj = nn.Linear(64, d_model)            # timestep -> token space
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_ff) for _ in range(n_layer)]
        )
        self.n_vis = n_vis
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, pad: torch.Tensor, vis: torch.Tensor,
                prop_tok: torch.Tensor, goal_tok: torch.Tensor,
                act_tok: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns hidden states (B, n_vis+L+1+n_vis+1, d_model)."""
        B, L = tokens.shape
        n_goal = goal_tok.shape[1]
        device = tokens.device
        pos_t = torch.arange(L, device=device)
        pos_v = torch.arange(self.n_vis, device=device)
        tv = torch.full((B, self.n_vis), T_VISION, dtype=torch.long, device=device)
        tt = torch.full((B, L), T_TEXT, dtype=torch.long, device=device)
        tp = torch.full((B, 1), T_PROPRIO, dtype=torch.long, device=device)
        tg = torch.full((B, n_goal), T_GOAL, dtype=torch.long, device=device)
        ta = torch.full((B, 1), T_ACTION, dtype=torch.long, device=device)
        t_emb = self.t_proj(timestep_embedding(t, 64))          # (B, d)
        h = torch.cat([
            vis + self.pos_vis(pos_v) + self.type_emb(tv),            # clean vision ctx
            self.tok(tokens) + self.pos_text(pos_t) + self.type_emb(tt),  # clean text ctx
            prop_tok + self.type_emb(tp),                              # clean proprio ctx
            goal_tok + self.pos_goal(pos_v) + self.type_emb(tg) + t_emb[:, None, :],
            act_tok + self.type_emb(ta) + t_emb[:, None, :],
        ], dim=1)
        ones_v = pad.new_ones((B, self.n_vis))
        ones_k = pad.new_ones((B, 1 + n_goal + 1))
        pad_full = torch.cat([ones_v, pad, ones_k], dim=1)
        for blk in self.blocks:
            h = blk(h, pad_full)
        return h


class PocketWAM(PreTrainedModel):
    """World Action Model: jointly generates the imagined final state (image)
    and the action via flow matching. Outputs ONLY (goal image, action)."""

    config_class = PocketWAMConfig
    base_model_prefix = "pocketwam"

    def __init__(self, config: PocketWAMConfig):
        super().__init__(config)
        c = self.config
        n_head = max(4, min(12, c.d_model // 64))
        assert c.d_model % n_head == 0
        grid = c.obs_res // 8
        n_vis = grid * grid
        self.grid = grid
        self.n_vis = n_vis

        # clean context encoders
        self.vision = VisionEncoder(c.obs_res, c.ch, c.extra_blocks)
        self.vis_proj = nn.Linear(c.ch[-1], c.d_model)
        self.proprio_proj = nn.Linear(c.proprio_dim, c.d_model)
        # noisy generative chunk encoders
        self.goal_enc = GoalEncoder(c.d_model, c.img_ch)
        self.action_enc = nn.Linear(c.action_dim, c.d_model)
        # shared trunk
        self.lm = WAMTrunk(c.vocab_size, c.d_model, n_head,
                           c.n_layer, c.d_ff, c.lang_len, n_vis)
        # clean-target heads (x0-parameterization)
        self.goal_dec = GoalDecoder(c.d_model, c.img_ch, c.obs_res)
        # ablation (default off): the action head reads a coarse summary of
        # the model's own delta (internal conditioning on the imagined transition)
        self.action_delta_cond = bool(getattr(c, "action_delta_cond", False))
        if self.action_delta_cond:
            self.delta_proj = nn.Linear(6 * (c.obs_res // 8) ** 2, c.d_model)
        h1, h2 = c.act_hidden
        self.action_head = nn.Sequential(
            nn.Linear(c.d_model, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, c.action_dim), nn.Tanh(),
        )
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor,
                pad: torch.Tensor, proprio: torch.Tensor,
                goal_noisy: torch.Tensor, action_noisy: torch.Tensor,
                t: torch.Tensor, **unused: Any):
        """One joint denoising step (training / one Euler step).

        image (B,3,H,W); tokens/pad (B,L); proprio (B,6); goal_noisy x_t,
        action_noisy a_t (flow space); t (B,) SHARED timestep. Returns
        (x1_hat, a1_hat): estimates of the CLEAN final-state image and action
        (x0-parameterization; flow velocity recovered analytically at inference).
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        B = image.shape[0]
        fmap = self.vision(image)                              # (B, C, 8, 8)
        vis = self.vis_proj(fmap.flatten(2).transpose(1, 2))   # (B, n_vis, d)
        prop_tok = self.proprio_proj(proprio).unsqueeze(1)      # (B, 1, d)
        goal_tok = self.goal_enc(goal_noisy)                    # (B, n_vis, d)
        act_tok = self.action_enc(action_noisy).unsqueeze(1)   # (B, 1, d)
        h = self.lm(tokens, pad, vis, prop_tok, goal_tok, act_tok, t)
        n_vis = self.n_vis
        vis_states = h[:, :n_vis, :]                            # vision hidden states
        goal_states = h[:, n_vis + tokens.shape[1] + 1: n_vis + tokens.shape[1] + 1 + n_vis, :]
        act_state = h[:, -1, :]                                 # action token output
        delta = self.goal_dec(torch.cat([goal_states, vis_states], dim=1))
        x1_hat = 2.0 * image - 1.0 + delta                      # residual: obs + delta
        if self.action_delta_cond:
            # signed coarse summary of the imagined transition; DETACHED so the
            # action cannot corrupt the video pathway with its own gradients
            g8 = delta.shape[-1] // 8                            # 8x8 summary grid
            dpos = torch.nn.functional.adaptive_max_pool2d(delta.detach(), (g8, g8))
            dneg = torch.nn.functional.adaptive_max_pool2d(-delta.detach(), (g8, g8))
            dfeat = self.delta_proj(torch.cat([dpos, dneg], dim=1).flatten(1))
            a1_hat = self.action_head(act_state + dfeat)        # (B, 3)
        else:
            a1_hat = self.action_head(act_state)                # (B, 3)
        return x1_hat, a1_hat

    def forward_idm(self, image: torch.Tensor, tokens: torch.Tensor,
                    pad: torch.Tensor, proprio: torch.Tensor,
                    goal_clean: torch.Tensor, action_draft: torch.Tensor):
        """Pass B — decoupled inverse dynamics at t=1: the trunk is evaluated at
        the flow's clean end with the (imagined or true) final-state image as
        goal context, so the action is decoded FROM the imagined future:

            Pass A (shared t):  (x_t, a_t) -> (x1_hat, a1_hat_draft)
            Pass B (t=1):       (goal_hat, a1_hat_draft) -> a1_hat
        """
        B = image.shape[0]
        t_one = torch.ones(B, device=image.device, dtype=torch.float32)
        return self.forward(image, tokens, pad, proprio,
                            goal_clean, action_draft, t_one)

    @torch.no_grad()
    def generate(self, image: torch.Tensor, tokens: torch.Tensor,
                 pad: torch.Tensor, proprio: torch.Tensor,
                 steps: int | None = None, generator: torch.Generator | None = None):
        """Joint generation: K Euler steps of the flow from t=0 (noise), then ONE
        inverse-dynamics pass decoding the action from the imagined final
        state at t=1. Returns (goal, action); steps=1 = Flash mode."""
        device = image.device
        B = image.shape[0]
        H = W = self.config.obs_res
        steps = int(steps or self.config.denoise_steps)
        x = torch.randn(B, 3, H, W, device=device, generator=generator)
        a = torch.randn(B, self.config.action_dim, device=device, generator=generator)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B,), k * dt, device=device, dtype=torch.float32)
            x1_hat, a1_hat = self.forward(image, tokens, pad, proprio, x, a, t)
            # analytic flow velocity from the clean estimate (x0-parameterization)
            x = x + dt * (x1_hat - x) / (1.0 - k * dt)
            a = a + dt * (a1_hat - a) / (1.0 - k * dt)
        x = x.clamp(-1.0, 1.0)
        a = a.clamp(-1.0, 1.0)
        # finalize the action from the imagined final state
        _, a1_hat = self.forward_idm(image, tokens, pad, proprio, x, a)
        return x, a1_hat.clamp(-1.0, 1.0)

    # ---- pixel helpers ----
    @staticmethod
    def to_flow(image01: torch.Tensor) -> torch.Tensor:
        """[0,1] pixels -> flow space [-1,1]."""
        return image01.float() * 2.0 - 1.0

    @staticmethod
    def to_image01(x: torch.Tensor) -> torch.Tensor:
        """flow space [-1,1] -> [0,1] pixels."""
        return (x.clamp(-1.0, 1.0) + 1.0) / 2.0

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- HF save/load (weights + config + tokenizer in one dir) ----
    def save_hf(self, path: str, tokenizer: "PocketWAMTokenizer"):
        self.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)

    @classmethod
    def load_hf(cls, path: str, device: torch.device) -> tuple["PocketWAM", "PocketWAMTokenizer"]:
        model = PocketWAM.from_pretrained(path).to(device).eval()
        tok = PocketWAMTokenizer.from_pretrained(path)
        return model, tok


class PocketWAMTokenizer(PreTrainedTokenizer):
    # "input_ids" must come first so transformers' padding/truncation resolves;
    # the model takes an explicit pad mask, no attention_mask needed.
    model_input_names = ["input_ids", "tokens"]

    def __init__(self, vocab: dict[str, int] | None = None, **kwargs):
        self.vocab = dict(vocab) if vocab else self._default_vocab()
        self.id2w = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("unk_token", "<unk>")
        super().__init__(**kwargs)

    @staticmethod
    def _default_vocab() -> dict[str, int]:
        from ..lang.tokenizer import TOKENIZER
        return dict(TOKENIZER.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id2w.get(index, "<unk>")

    def build_inputs(self, text: str, lang_len: int) -> dict[str, Any]:
        """Text -> model inputs (padded token id array)."""
        ids = [self._convert_token_to_id(t) for t in self._tokenize(text)]
        arr = np.zeros(lang_len, dtype=np.int64)
        n = min(len(ids), lang_len)
        arr[:n] = ids[:n]
        return {"tokens": arr}

    def decode_tokens(self, tokens) -> str:
        return " ".join(self.id2w.get(int(t), "<unk>")
                        for t in tokens if int(t) != self.pad_id)

    # ---- store the small vocab in vocab.json ----
    def save_vocabulary(self, save_directory: str,
                       filename_prefix: str | None = None) -> tuple[str]:
        import os
        path = os.path.join(save_directory, "vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=1)
        return (path,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        import os
        vocab_path = os.path.join(str(pretrained_model_name_or_path), "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, encoding="utf-8") as f:
                vocab = json.load(f)
            kwargs["vocab"] = {k: int(v) for k, v in vocab.items()}
        else:
            kwargs.setdefault("vocab", None)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


# AutoClass registration
AutoConfig.register("pocketwam", PocketWAMConfig)
AutoModel.register(PocketWAMConfig, PocketWAM)
AutoTokenizer.register(PocketWAMConfig, slow_tokenizer_class=PocketWAMTokenizer)
