"""WAM SFT training: joint denoising flow of the imagined final state (video
loss) and the action (inverse-dynamics loss) with a SHARED timestep —
L = MSE(x1_hat, x1) + MSE(a1_hat, a1), t ~ U(0,1). Heads predict the CLEAN
targets (x0-parameterization; flow velocity recovered analytically at
inference). Validation measures one-step generation quality at t=0.
Reads train/val npz (WAM format with per-episode final-state images),
writes checkpoint + log + curve to train.output_dir.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, device_banner, ensure_dir, get_device
from .lang.tokenizer import COLORS, TOKENIZER
from .model.wam import PocketWAM, PocketWAMConfig, PocketWAMTokenizer
from . import visualize


class WAMDataset(torch.utils.data.Dataset):
    """Per-step samples; __getitem__ gathers the episode final-state image by
    ep_id. Grip weights are precomputed once for the whole split."""

    def __init__(self, data: dict, grip_w: torch.Tensor):
        self.images = torch.from_numpy(data["images"])        # uint8 (N,64,64,3)
        self.tokens = torch.from_numpy(data["tokens"])        # int64 (N,10)
        self.proprio = torch.from_numpy(data["proprio"])      # float32 (N,6)
        self.actions = torch.from_numpy(data["actions"])      # float32 (N,3)
        self.ep_id = torch.from_numpy(data["ep_id"]).long()  # (N,)
        self.ep_final = torch.from_numpy(data["ep_final"])   # uint8 (n_eps,64,64,3)
        self.grip_w = grip_w                                  # (N,) sample weights

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        return (self.images[i], self.tokens[i], self.proprio[i], self.actions[i],
                self.ep_final[self.ep_id[i]], self.grip_w[i])


def _grip_weight_dataset(data: dict) -> torch.Tensor:
    """Class-balanced inverse-frequency weights over (holding x label) groups,
    computed once per split (dataset-level statistics)."""
    lab = torch.from_numpy(data["actions"])[:, 2]
    holding = torch.from_numpy(data["proprio"])[:, 5] > 0.5
    n = lab.numel()
    w = torch.ones(n, dtype=torch.float32)
    for hm in (holding, ~holding):
        for lm in (lab > 0, lab <= 0):
            m = hm & lm
            cnt = int(m.sum())
            if cnt > 0:
                w[m] = float(np.sqrt(n / cnt))
    return w / w.mean().clamp(min=1e-6)


def _make_loader(data: dict, cfg: Config, shuffle: bool) -> DataLoader:
    grip_w = _grip_weight_dataset(data) if cfg.balanced_losses else \
        torch.ones(len(data["images"]), dtype=torch.float32)
    ds = WAMDataset(data, grip_w)
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=0, generator=g, drop_last=False)


def _grip_weight(lab: torch.Tensor, holding: torch.Tensor) -> torch.Tensor:
    """(kept for API parity; the training loop uses the precomputed weights)"""
    w = torch.ones_like(lab)
    for hm in (holding, ~holding):
        for lm in (lab > 0, lab <= 0):
            m = hm & lm
            n = int(m.sum())
            if n > 0:
                w[m] = float(np.sqrt(m.numel() / n))
    return w / w.mean().clamp(min=1e-6)


def _denorm_action(act: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Label actions (N,3) -> physical units; grip stays +/-1."""
    dq = act[:, :2] * cfg.action_max
    return torch.cat([dq, act[:, 2:3]], dim=1)


def _flow_setup(a1, x1, device, training, gen: torch.Generator | None = None):
    """Sample the shared timestep t, noise and noisy targets x_t / a_t (fp32,
    outside autocast). training: t~U(0,1); validation: t=0 with FIXED noise."""
    B = a1.shape[0]
    if training:
        t = torch.rand(B, device=device)
        eps = torch.randn(B, 3, x1.shape[-2], x1.shape[-1], device=device)
        eps_a = torch.randn(B, a1.shape[1], device=device)
    else:
        t = torch.zeros(B, device=device)
        eps = torch.randn(B, 3, x1.shape[-2], x1.shape[-1],
                          generator=gen).to(device)
        eps_a = torch.randn(B, a1.shape[1], generator=gen).to(device)
    x_t = t[:, None, None, None] * x1 + (1.0 - t[:, None, None, None]) * eps
    a_t = t[:, None] * a1 + (1.0 - t[:, None]) * eps_a
    return t, eps, eps_a, x_t, a_t


def _flow_loss(model, img, tok, pad, prop, a1, x1, t, eps, eps_a, x_t, a_t,
               grip_w, cfg):
    """Joint denoising forward + loss (training) or one-shot generation metrics
    at t=0 (validation)."""
    x1_hat, a1_hat = model(img, tok, pad, prop, x_t, a_t, t)

    parts = {}
    if grip_w is not None:
        # flow-timestep weight w(t) = (1-t)^p: down-weights the high-t
        # 'denoise-copy' shortcut, forcing semantic generation
        wt = (1.0 - t) ** cfg.t_weight_power if cfg.t_weight_power > 0 \
            else torch.ones_like(t)
        wt = wt[:, None, None, None]                      # (B,1,1,1) for video
        wt_a = wt[:, 0]                                   # (B,) for action

        # world model: imagine the final state; changed pixels (dynamics)
        # are weighted up so the static 'copy' pixels cannot drown the gradient
        if cfg.video_semantic_weight > 1.0:
            img_flow = 2.0 * img - 1.0
            changed = ((x1 - img_flow).abs().mean(dim=1, keepdim=True) > 0.12)
            wmap = 1.0 + (cfg.video_semantic_weight - 1.0) * changed.float()
        else:
            wmap = torch.ones_like(x1[:, :1])
        wv = wt * wmap                                   # (B,1,H,W)
        denom = (wv.expand_as(x1)).sum().clamp(min=1.0)
        video_loss = (wv * (x1_hat - x1) ** 2).sum() / denom

        # pass A: joint-denoising action draft
        dq_err_a = (a1_hat[:, :2] * cfg.action_max - a1[:, :2] * cfg.action_max) ** 2
        dq_loss_a = (wt_a * dq_err_a.mean(dim=1)).sum() / wt_a.sum().clamp(min=1e-6)
        grip_err_a = (a1_hat[:, 2] - a1[:, 2]) ** 2
        grip_loss_a = (wt_a * grip_w * grip_err_a).sum() / (wt_a * grip_w).sum().clamp(min=1e-6)

        # pass B: action decoded at t=1 from the (imagined or true) final
        # state; scheduled sampling mixes x1_hat with the true final state
        if True:
            use_true = (torch.rand(a1.shape[0], device=a1.device) < cfg.idm_true_prob)
            goal_b = torch.where(use_true[:, None, None, None], x1, x1_hat.detach())
            a_draft = a1_hat.detach()
            _, a1_hat_b = model.forward_idm(img, tok, pad, prop, goal_b, a_draft)
            dq_err_b = (a1_hat_b[:, :2] * cfg.action_max - a1[:, :2] * cfg.action_max) ** 2
            dq_loss_b = dq_err_b.mean()
            grip_err_b = (a1_hat_b[:, 2] - a1[:, 2]) ** 2
            grip_loss_b = (grip_w * grip_err_b).mean()
            dq_loss = dq_loss_a + cfg.idm_loss_weight * dq_loss_b
            grip_loss = grip_loss_a + cfg.idm_loss_weight * grip_loss_b
        else:
            dq_loss = dq_loss_a
            grip_loss = grip_loss_a
        loss = (cfg.video_loss_weight * video_loss
                + cfg.action_loss_weight * (dq_loss + cfg.action_grip_weight * grip_loss))
        parts = {"video": video_loss.detach(), "dq": dq_loss.detach(),
                 "grip": grip_loss.detach()}
        return loss, parts

    # ---- validation: one-shot generation metrics at t=0 ----
    with torch.no_grad():
        # video: imagined final state vs the true final state
        x1_hat = x1_hat.clamp(-1.0, 1.0)          # valid image range for metrics
        video_mse = F.mse_loss(x1_hat, x1)                        # flow space / dim
        mse01 = F.mse_loss((x1_hat + 1.0) / 2.0, (x1 + 1.0) / 2.0)
        psnr = 10.0 * torch.log10(1.0 / mse01.clamp(min=1e-10))
        # action: the deployed pathway — pass B on the model's own imagination
        _, a1_hat_b = model.forward_idm(img, tok, pad, prop, x1_hat.detach(),
                                        a1_hat.detach())
        dq_err = F.mse_loss(a1_hat_b[:, :2] * cfg.action_max, a1[:, :2] * cfg.action_max)
        grip_err = F.mse_loss(a1_hat_b[:, 2], a1[:, 2])
    parts = {"video": video_mse, "psnr": psnr, "dq": dq_err, "grip": grip_err}
    return video_mse + dq_err, parts


def _run_epoch(model, loader, device, cfg: Config, opt=None):
    """One epoch (opt=None -> validation). Metrics accumulate on the GPU;
    the only host sync per epoch happens at the very end."""
    training = opt is not None
    model.train() if training else model.eval()
    use_amp = device.type == "cuda"
    gen = None if training else torch.Generator().manual_seed(cfg.seed)
    totals: dict[str, torch.Tensor] = {}
    count = 0
    for batch in loader:
        img_nhwc, tok, prop, act, goal_nhwc, grip_w = batch
        img = img_nhwc.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        tok = tok.to(device, non_blocking=True)
        pad = tok != 0
        prop = prop.to(device, non_blocking=True)
        a1 = act.to(device, non_blocking=True)                     # (B,3) in [-1,1]
        grip_w = grip_w.to(device, non_blocking=True)
        x1 = goal_nhwc.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        x1 = 2.0 * x1 - 1.0                                        # flow space [-1,1]
        # flow-matching setup in fp32 (outside autocast)
        t, eps, eps_a, x_t, a_t = _flow_setup(a1, x1, device, training, gen=gen)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with torch.set_grad_enabled(training), amp:
            loss, parts = _flow_loss(model, img, tok, pad, prop, a1, x1,
                                     t, eps, eps_a, x_t, a_t,
                                     grip_w if training else None, cfg)
        if training:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        totals["loss"] = totals.get("loss", 0.0) + loss.detach()
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        count += 1
    k = max(count, 1)
    out = {k2: (v / k).item() for k2, v in totals.items()}
    return out["loss"], {k2: v for k2, v in out.items() if k2 != "loss"}


# ---------------- Stage 0: world-model pretraining ----------------

class PretrainDataset(torch.utils.data.Dataset):
    """Stage-0 rows: i < n_synth -> synthetic goal pair, else play-dynamics pair."""

    def __init__(self, synth: dict, play: dict, idx: np.ndarray):
        sg = {k: torch.from_numpy(synth[k]) for k in
              ("sg_scenes", "sg_proprio", "sg_scene_idx", "sg_tokens", "sg_goal")}
        pl = {k: torch.from_numpy(play[k]) for k in
              ("pl_images", "pl_tokens", "pl_proprio", "pl_action", "pl_goal")}
        self.sg, self.pl = sg, pl
        self.idx = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        self.n_synth = len(sg["sg_goal"])

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        if j < self.n_synth:
            s = int(self.sg["sg_scene_idx"][j])
            return (self.sg["sg_scenes"][s], self.sg["sg_tokens"][j],
                    self.sg["sg_proprio"][s], self.sg["sg_goal"][j],
                    torch.zeros(3), 0)
        k = j - self.n_synth
        return (self.pl["pl_images"][k], self.pl["pl_tokens"][k],
                self.pl["pl_proprio"][k], self.pl["pl_goal"][k],
                self.pl["pl_action"][k], 1)


def _pretrain_epoch(model, loader, device, cfg: Config, opt=None):
    """One Stage-0 epoch: VIDEO-ONLY loss (no action loss — play actions are
    random). Executed actions are injected as clean conditioning for play
    rows (10% CFG dropout); synthetic rows use pure noise."""
    training = opt is not None
    model.train() if training else model.eval()
    use_amp = device.type == "cuda"
    gen = None if training else torch.Generator().manual_seed(cfg.seed)
    totals: dict[str, torch.Tensor] = {}
    count = 0
    for batch in loader:
        img_nhwc, tok, prop, goal_nhwc, act_cond, has_act = batch
        img = img_nhwc.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        tok = tok.to(device, non_blocking=True)
        pad = tok != 0
        prop = prop.to(device, non_blocking=True)
        act_cond = act_cond.to(device, non_blocking=True)
        has_act = has_act.to(device, non_blocking=True).bool()
        x1 = goal_nhwc.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        x1 = 2.0 * x1 - 1.0
        B = img.shape[0]
        if training:
            t = torch.rand(B, device=device)
            eps = torch.randn(B, 3, x1.shape[-2], x1.shape[-1], device=device)
        else:
            t = torch.zeros(B, device=device)
            eps = torch.randn(B, 3, x1.shape[-2], x1.shape[-1],
                              generator=gen).to(device)
        x_t = t[:, None, None, None] * x1 + (1.0 - t[:, None, None, None]) * eps
        # action conditioning: clean for play rows (10% dropout -> noise),
        # pure noise for synthetic rows (no action)
        noise_a = torch.randn(B, act_cond.shape[1], device=device)
        use_cond = has_act & (torch.rand(B, device=device) > 0.10)
        a_in = torch.where(use_cond[:, None], act_cond, noise_a)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with torch.set_grad_enabled(training), amp:
            x1_hat, _ = model(img, tok, pad, prop, x_t, a_in, t)
            if training:
                wt = (1.0 - t) ** cfg.t_weight_power
                wt4 = wt[:, None, None, None]
                if cfg.video_semantic_weight > 1.0:
                    img_flow = 2.0 * img - 1.0
                    changed = ((x1 - img_flow).abs().mean(dim=1, keepdim=True) > 0.12)
                    wmap = 1.0 + (cfg.video_semantic_weight - 1.0) * changed.float()
                else:
                    wmap = torch.ones_like(x1[:, :1])
                wv = wt4 * wmap
                denom = (wv.expand_as(x1)).sum().clamp(min=1.0)
                loss = cfg.video_loss_weight * (wv * (x1_hat - x1) ** 2).sum() / denom
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                loss = None
        if training:
            totals["loss"] = totals.get("loss", 0.0) + loss.detach()
        else:
            x1_hat = x1_hat.clamp(-1.0, 1.0)
            totals["mse"] = totals.get("mse", 0.0) + torch.nn.functional.mse_loss(x1_hat, x1)
        count += 1
    k = max(count, 1)
    return {kk: (v / k).item() for kk, v in totals.items()}


def pretrain(cfg: Config) -> dict:
    """Stage 0 — world-model pretraining on video-only data (synthetic goals +
    play dynamics; no action labels)."""
    device = get_device()
    torch.manual_seed(cfg.seed)
    print(f"[pretrain] {device_banner(device)}")
    for f in (cfg.pretrain_synth_npz, cfg.pretrain_play_npz):
        if not Path(f).is_file():
            raise FileNotFoundError(
                f"[pretrain] {f} not found — run 'python3 -m pwam collect' first")
    if cfg.pretrain_dir is None:
        raise ValueError("[pretrain] yaml must set pretrain.output_dir")
    ensure_dir(cfg.pretrain_dir)
    synth = dict(np.load(cfg.pretrain_synth_npz, allow_pickle=False))
    play = dict(np.load(cfg.pretrain_play_npz, allow_pickle=False))
    n_synth = len(synth["sg_goal"])
    n_play = len(play["pl_goal"])
    # held-out tails: no scene/episode leakage into validation
    n_synth_val = (int(n_synth * 0.05) // 3) * 3
    train_idx = np.arange(n_synth - n_synth_val)
    val_idx = np.arange(n_synth - n_synth_val, n_synth)
    n_play_val = int(n_play * 0.05)
    train_idx = np.concatenate([train_idx, n_synth + np.arange(n_play - n_play_val)])
    val_idx = np.concatenate([val_idx, n_synth + np.arange(n_play - n_play_val, n_play)])
    print(f"[pretrain] video-only pairs: {len(train_idx)} train / {len(val_idx)} val  "
          f"({n_synth} synth-goal + {n_play} play-dynamics; disjoint from SFT demos)")

    tr_loader = DataLoader(PretrainDataset(synth, play, train_idx),
                            batch_size=cfg.pretrain_batch_size, shuffle=True,
                            num_workers=0,
                            generator=torch.Generator().manual_seed(cfg.seed))
    va_loader = DataLoader(PretrainDataset(synth, play, val_idx),
                           batch_size=cfg.pretrain_batch_size, shuffle=False, num_workers=0)

    hf_cfg = PocketWAMConfig(
        size=cfg.model_size, obs_res=cfg.obs_res, world_extent=cfg.world_extent,
        proprio_dim=cfg.proprio_dim, action_dim=cfg.action_dim,
        lang_len=cfg.lang_len, vocab_size=len(TOKENIZER),
        action_max=cfg.action_max, **cfg.model_arch)
    model = PocketWAM(hf_cfg).to(device)
    print(f"[pretrain] model params: {model.n_params() / 1e6:.2f}M  "
          f"(epochs={cfg.pretrain_epochs}, lr={cfg.pretrain_lr:.1e})")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.pretrain_epochs)

    log, best = [], float("inf")
    t0 = time.time()
    for epoch in range(1, cfg.pretrain_epochs + 1):
        tr = _pretrain_epoch(model, tr_loader, device, cfg, opt=opt)
        with torch.no_grad():
            va = _pretrain_epoch(model, va_loader, device, cfg, opt=None)
        sched.step()
        entry = {"epoch": epoch, "train_loss": tr["loss"], "val_mse": va["mse"],
                 "lr": sched.get_last_lr()[0]}
        log.append(entry)
        flag = ""
        if va["mse"] < best:
            best = va["mse"]
            model.save_hf(cfg.pretrain_hf_dir, PocketWAMTokenizer(TOKENIZER.vocab))
            flag = "  <- best, saved"
        print(f"  epoch {epoch:3d}/{cfg.pretrain_epochs}  train={tr['loss']:.5f}  "
              f"val_video_mse={va['mse']:.5f}  lr={entry['lr']:.2e}{flag}")
        Path(cfg.pretrain_dir, "pretrain_log.json").write_text(json.dumps(log, indent=1))
    visualize.simple_curve(log, str(Path(cfg.pretrain_dir) / "pretrain_curve.png"))
    stats = {"pairs": len(train_idx) + len(val_idx), "epochs": cfg.pretrain_epochs,
             "best_val_mse": best, "seconds": time.time() - t0,
             "params": model.n_params()}
    print(f"[pretrain] done: best val video mse={best:.5f}  "
          f"checkpoint={cfg.pretrain_hf_dir}  elapsed={stats['seconds']:.1f}s")
    # input / label / prediction example grids (test set + held-out pretrain pairs)
    stats.update(pretrain_examples(cfg))
    return stats


def pretrain_examples(cfg: Config, n_test: int = 6, n_val: int = 6) -> dict:
    """Visualize the pretrained world model's predictions (test scenarios and
    held-out pretraining pairs) into pretrain.output_dir PNG grids."""
    from .data.collect import scenario_at
    from .expert.analytic import AnalyticExpert
    from .sim.arm2d import Arm2DPickPlace

    device = get_device()
    out_dir = Path(cfg.pretrain_dir)
    model, _ = PocketWAM.load_hf(cfg.pretrain_hf_dir, device)
    model.eval()
    rng = np.random.default_rng(cfg.seed)

    def _predict(img64, tokens, proprio, action_cond=None, seed=0):
        img = torch.from_numpy(img64).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.
        tok = torch.from_numpy(np.asarray(tokens)).unsqueeze(0).to(device)
        pad = tok != 0
        pr = torch.from_numpy(np.asarray(proprio, dtype=np.float32)).unsqueeze(0).to(device)
        gen = torch.Generator(device=device.type); gen.manual_seed(seed)
        with torch.no_grad():
            if action_cond is None:      # goal prediction (SFT regime)
                goal_x, _ = model.generate(img, tok, pad, pr,
                                            steps=model.config.denoise_steps,
                                            generator=gen)
                x1_hat = goal_x[0]
            else:                        # play dynamics: condition on the action
                x = torch.randn(1, 3, cfg.obs_res, cfg.obs_res, device=device,
                                generator=gen)
                t = torch.zeros(1, device=device)
                a = torch.from_numpy(np.asarray(action_cond, dtype=np.float32)
                                     ).unsqueeze(0).to(device)
                x1_hat, _ = model(img, tok, pad, pr, x, a, t)
                x1_hat = x1_hat[0]
        im = model.to_image01(x1_hat).cpu().numpy().transpose(1, 2, 0)
        return (im * 255).round().astype(np.uint8)

    # (a) fixed test scenarios: input | prediction | expert-achieved label
    scen = dict(np.load(cfg.test_npz, allow_pickle=True))
    env = Arm2DPickPlace(cfg, seed=1)
    expert = AnalyticExpert(cfg, seed=0, noise=0.0)
    rows = []
    n_all = len(scen["q0"])
    for ep in rng.choice(n_all, size=n_test, replace=False):
        sc = scenario_at(scen, int(ep))
        obs = env.reset_scenario(sc)
        pred = _predict(obs["image"], obs["tokens"], obs["proprio"], seed=int(ep))
        obs2 = env.reset_scenario(sc)
        done = False
        while not done:                      # label = expert-achieved final state
            obs2, _, done, info = env.step(expert.act(env))
        rows.append((obs["image"], pred, obs2["image"],
                     f'"{obs["instruction"]}"'))
    visualize.compare_grid(
        rows, str(out_dir / "pretrain_examples_test.png"),
        col_titles=("input: observation\n(fixed test scenario)",
                    "pretrained model PREDICTION\n(one-shot imagined final state)",
                    "GROUND TRUTH label\n(expert-achieved final state)"),
        suptitle="Stage-0 pretrained world model — predictions on the fixed TEST set "
                 "(never seen in any training data)")

    # (b) held-out pretraining pairs
    synth = dict(np.load(cfg.pretrain_synth_npz, allow_pickle=False))
    play = dict(np.load(cfg.pretrain_play_npz, allow_pickle=False))
    n_synth = len(synth["sg_goal"])
    n_synth_val = (int(n_synth * 0.05) // 3) * 3     # same tail as pretrain()
    n_play = len(play["pl_goal"])
    n_play_val = int(n_play * 0.05)
    rows = []
    val_synth = np.arange(n_synth - n_synth_val, n_synth)
    for j in rng.choice(val_synth, size=n_val // 2, replace=False):
        j = int(j); si = int(synth["sg_scene_idx"][j])
        pred = _predict(synth["sg_scenes"][si], synth["sg_tokens"][j],
                        synth["sg_proprio"][si], seed=j)
        rows.append((synth["sg_scenes"][si], pred, synth["sg_goal"][j],
                     f"synthetic goal pair — command: {COLORS[int(synth['sg_color_id'][j])]}"
                     f"\n(held-out val scene)"))
    val_play = np.arange(n_play - n_play_val, n_play) + n_synth
    for k in rng.choice(val_play, size=n_val // 2, replace=False):
        k = int(k) - n_synth
        pred = _predict(play["pl_images"][k], play["pl_tokens"][k],
                        play["pl_proprio"][k], action_cond=play["pl_action"][k],
                        seed=k)
        rows.append((play["pl_images"][k], pred, play["pl_goal"][k],
                     "play-dynamics pair — predict the NEXT frame\n(given the executed action)"))
    visualize.compare_grid(
        rows, str(out_dir / "pretrain_examples_val.png"),
        col_titles=("input: observation", "pretrained model PREDICTION", "label (ground truth)"),
        suptitle="Stage-0 pretrained world model — held-out pretraining pairs "
                 "(synthetic goals + play dynamics)")
    return {"test_examples": n_test, "val_examples": len(rows)}


def train(cfg: Config) -> dict:
    device = get_device()
    torch.manual_seed(cfg.seed)
    print(f"[train] {device_banner(device)}")

    missing = [p for p in (cfg.train_npz, cfg.val_npz) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            "[train] dataset file(s) not found: " + ", ".join(missing) + "\n"
            "[train]   run 'python3 -m pwam collect --config <yaml>' first")
    if cfg.train_dir is None:
        raise ValueError("[train] yaml must set train.output_dir (SFT artifacts directory)")
    ensure_dir(cfg.train_dir)
    train_data = dict(np.load(cfg.train_npz, allow_pickle=False))
    val_data = dict(np.load(cfg.val_npz, allow_pickle=False))
    for name, d in (("train", train_data), ("val", val_data)):
        need = {"images", "tokens", "proprio", "actions", "ep_id", "ep_final"}
        miss = need - set(d)
        if miss:
            raise KeyError(f"[train] {name} set missing columns {sorted(miss)} "
                           f"(WAM dataset format required; re-run 'python3 -m pwam collect')")
    n = len(train_data["images"]) + len(val_data["images"])
    print(f"[train] dataset: train={cfg.train_npz}  val={cfg.val_npz}")
    print(f"[train] samples={n}  train={len(train_data['images'])}  "
          f"val={len(val_data['images'])}  (episode-level split, "
          f"no adjacent-frame leakage)")
    print(f"[train] world-model targets: "
          f"{len(train_data['ep_final'])} / {len(val_data['ep_final'])} "
          f"episode final-state images")

    train_loader = _make_loader(train_data, cfg, shuffle=True)
    val_loader = _make_loader(val_data, cfg, shuffle=False)

    hf_cfg = PocketWAMConfig(
        size=cfg.model_size, obs_res=cfg.obs_res, world_extent=cfg.world_extent,
        proprio_dim=cfg.proprio_dim, action_dim=cfg.action_dim,
        lang_len=cfg.lang_len, vocab_size=len(TOKENIZER),
        action_max=cfg.action_max, **cfg.model_arch,
    )
    if cfg.init_ckpt:
        # init SFT from the pretrained video backbone
        if not Path(cfg.init_ckpt).is_dir():
            raise FileNotFoundError(f"[train] init_ckpt not found: {cfg.init_ckpt}")
        model = PocketWAM.from_pretrained(cfg.init_ckpt).to(device)
        print(f"[train] initialized from pretrained world model: {cfg.init_ckpt}")
    else:
        model = PocketWAM(hf_cfg).to(device)
    print(f"[train] model params: {model.n_params() / 1e6:.2f}M  "
          f"(size={cfg.model_size}, lr={cfg.lr:.1e}, "
          f"denoise_steps={hf_cfg.denoise_steps})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    log, best_ctrl = [], float("inf")
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        tr, tr_parts = _run_epoch(model, train_loader, device, cfg, opt=opt)
        with torch.no_grad():
            va, va_parts = _run_epoch(model, val_loader, device, cfg, opt=None)
        sched.step()
        lr = sched.get_last_lr()[0]
        # selection: joint val loss + one-shot quality; dq weighted 5x so the
        # (numerically larger) video term cannot hide degraded actions
        va_ctrl = 5.0 * va_parts["dq"] + va_parts["video"]
        entry = {"epoch": epoch, "train": tr, "val": va, "lr": lr,
                 "train_video": tr_parts.get("video"),
                 "train_dq": tr_parts.get("dq"), "train_grip": tr_parts.get("grip"),
                 "val_video_mse": va_parts["video"], "val_video_psnr": va_parts["psnr"],
                 "val_dq_mse": va_parts["dq"], "val_grip_mse": va_parts["grip"],
                 "val_ctrl": va_ctrl, "denoise_steps": hf_cfg.denoise_steps}
        log.append(entry)
        flag = ""
        if va_ctrl < best_ctrl:
            best_ctrl = va_ctrl
            model.save_hf(cfg.hf_dir, PocketWAMTokenizer(TOKENIZER.vocab))
            flag = "  <- best, saved"
        print(f"  epoch {epoch:3d}/{cfg.epochs}  train={tr:.5f}  val={va:.5f}  "
              f"ctrl={va_ctrl:.5f}  video(mse={va_parts['video']:.4f} "
              f"psnr={va_parts['psnr']:.1f}dB)  dq={va_parts['dq']:.5f}  "
              f"grip={va_parts['grip']:.4f}  lr={lr:.2e}{flag}")
        Path(cfg.train_log_path).write_text(json.dumps(log, indent=1))

    visualize.loss_curve(log, cfg.loss_curve_path)
    stats = {"samples": n, "train_samples": len(train_data["images"]),
             "val_samples": len(val_data["images"]), "epochs": cfg.epochs,
             "best_val_ctrl": best_ctrl,
             "denoise_steps": hf_cfg.denoise_steps,
             "seconds": time.time() - t0, "params": model.n_params()}
    print(f"[train] done: best val ctrl={best_ctrl:.5f}  "
          f"checkpoint={cfg.hf_dir}  elapsed={stats['seconds']:.1f}s")
    return stats
