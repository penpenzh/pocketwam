"""Experiment config: ONE yaml fully defines ONE WAM pipeline (strict key
validation; unknown keys rejected). Sim / expert / task / eval constants are
identical to PocketVLA (test set seed+20,000+i, OOD set seed+80,000+i).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

import torch

# Ball color -> RGB (uint8)
COLOR_RGB = {
    "red": (222, 70, 70),
    "green": (70, 200, 100),
    "blue": (80, 130, 235),
}

# Bin colors (rim / interior)
BIN_RIM = (240, 160, 70)
BIN_FILL = (52, 45, 40)

# yaml `model:` keys (besides size) that override the preset arch
ARCH_KEYS = ("ch", "extra_blocks", "d_model", "n_layer", "d_ff",
             "act_hidden", "img_ch", "denoise_steps", "action_delta_cond")

# yaml section -> allowed config fields
_SECTIONS = {
    "sim": {k: k for k in (
        "link_lengths", "world_extent", "obs_res", "ball_radius", "n_balls",
        "bin_radius", "bin_r_range", "min_ball_dist", "min_ball_bin_dist",
        "action_max", "grasp_thresh", "place_overlap_frac", "max_steps",
        "target_r_range", "init_q1_range", "init_q2_range")},
    "expert": {k: k for k in (
        "expert_gain", "expert_fine_gain", "expert_fine_radius",
        "expert_noise", "expert_grasp_dist", "expert_release_dist")},
    "task": {k: k for k in ("proprio_dim", "action_dim", "lang_len")},
    "pretrain": {"output_dir": "pretrain_dir", "epochs": "pretrain_epochs",
                 "batch_size": "pretrain_batch_size", "lr": "pretrain_lr"},
    "train": {"output_dir": "train_dir", "epochs": "epochs",
              "batch_size": "batch_size", "lr": "lr",
              "init_ckpt": "init_ckpt",
              "weight_decay": "weight_decay", "balanced_losses": "balanced_losses",
              "video_loss_weight": "video_loss_weight",
              "action_loss_weight": "action_loss_weight",
              "action_grip_weight": "action_grip_weight",
              "video_semantic_weight": "video_semantic_weight",
              "t_weight_power": "t_weight_power",
              "idm_loss_weight": "idm_loss_weight",
              "idm_true_prob": "idm_true_prob"},
    "eval": {"output_dir": "eval_dir", "gif_fps": "gif_fps",
             "frame_res": "frame_res", "ckpt": "eval_ckpt", "gifs": "eval_gifs",
             "denoise_steps": "eval_denoise_steps"},
}
# yaml `data:` keys -> config fields
_DATA_KEYS = {"dataset_dir": "dataset_dir", "episodes_collect": "episodes_collect",
              "val_frac": "val_frac", "eval_episodes": "eval_episodes",
              "warm_start_prob": "warm_start_prob", "play_max_steps": "play_max_steps",
              "pretrain_scenes": "pretrain_scenes",
              "pretrain_play_episodes": "pretrain_play_episodes"}
_ROOT_KEYS = ("seed",)
_ALL_SECTIONS = ("model", "data") + tuple(_SECTIONS)


@dataclass
class Config:
    # ---- Simulation (identical to PocketVLA) ----
    seed: int = 42
    link_lengths: tuple = (0.55, 0.55)   # (l1, l2)
    world_extent: float = 1.25           # world coords span [-E, E]^2
    obs_res: int = 64                    # observation image resolution
    ball_radius: float = 0.075
    n_balls: int = 3                     # red / green / blue
    bin_radius: float = 0.20             # ball must lie fully inside the rim
    bin_r_range: tuple = (0.50, 0.95)    # bin center sampling radius
    min_ball_dist: float = 0.40          # min distance between balls
    min_ball_bin_dist: float = 0.35      # min distance ball-to-bin center
    action_max: float = 0.15             # per-step joint delta clamp (rad)
    grasp_thresh: float = 0.09           # snap-on distance when closing gripper
    place_overlap_frac: float = 1.0      # 1.0 = ball fully inside the bin rim
    max_steps: int = 90
    target_r_range: tuple = (0.45, 1.0)  # ball sampling radius range
    init_q1_range: tuple = (-2.5, 2.5)
    init_q2_range: tuple = (-2.6, -0.5)  # elbow-down initial configurations

    # ---- Expert (identical to PocketVLA) ----
    expert_gain: float = 0.55            # proportional IK gain (cruise)
    expert_fine_gain: float = 0.30       # slower gain near the goal
    expert_fine_radius: float = 0.18    # fine-gain switch distance
    expert_noise: float = 0.002          # demo joint-action noise (rad)
    expert_grasp_dist: float = 0.11      # close gripper within this distance
    expert_release_dist: float = 0.05   # release near the bin center

    # ---- Language (identical to PocketVLA) ----
    lang_len: int = 10                   # instruction token sequence length

    # ---- Model (yaml `model:` section) ----
    model_size: str = "3m"               # preset key (3m) or free label
    model_arch: dict = field(default_factory=dict)   # arch overrides (ARCH_KEYS)
    balanced_losses: bool = True         # class-balanced grip loss on/off

    # ---- Task I/O dims (identical to PocketVLA) ----
    proprio_dim: int = 6                 # q1, q2, ee_x, ee_y, grip, holding
    action_dim: int = 3                  # dq1, dq2, grip
    video_loss_weight: float = 1.0       # world-model (video) loss weight
    action_loss_weight: float = 4.0      # action (inverse dynamics) loss weight
    action_grip_weight: float = 0.3      # grip loss weight (after class balancing)
    video_semantic_weight: float = 10.0  # loss weight of CHANGED pixels (obs->final)
    t_weight_power: float = 1.0         # flow-timestep loss weight w(t)=(1-t)^p; 0 = uniform
    idm_loss_weight: float = 1.0        # pass-B inverse-dynamics loss weight
    idm_true_prob: float = 0.5          # pass B scheduled sampling: P(true final state)

    # ---- Data generation (WAM expansion) ----
    episodes_collect: int = 10000        # demo episodes
    pretrain_scenes: int = 30000          # pretrain scenes (x3 goal variants)
    pretrain_play_episodes: int = 12000  # pretrain random-play episodes
    val_frac: float = 0.30               # validation split by episode
    warm_start_prob: float = 0.35        # fraction of warm-started episodes
    play_max_steps: int = 8              # max random play pre-roll steps per episode

    # ---- Stage 0: world-model pretraining ----
    pretrain_epochs: int = 12
    pretrain_batch_size: int = 256
    pretrain_lr: float = 4.0e-4
    pretrain_dir: str | None = None       # yaml pretrain.output_dir

    # ---- Training (video+action SFT) ----
    init_ckpt: str | None = None         # pretrained backbone to init SFT from
    batch_size: int = 256
    epochs: int = 15
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-5

    # ---- Inference / evaluation ----
    eval_episodes: int = 500            # fixed test scenarios
    ood_test_episodes: int = 300        # OOD test scenarios
    eval_ckpt: str | None = None         # checkpoint to evaluate (--ckpt overrides)
    eval_gifs: int = 30                  # random replay GIFs saved
    eval_denoise_steps: int | None = None  # eval denoise-steps override (Flash mode)
    gif_fps: int = 12
    frame_res: int = 256                # visualization frame resolution

    # ---- Output locations (from the yaml) ----
    dataset_dir: str = "dataset"         # yaml data.dataset_dir
    train_dir: str | None = None         # yaml train.output_dir
    eval_dir: str | None = None          # yaml eval.output_dir

    # ---- YAML loading ----
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load one experiment from a yaml file (strict key validation)."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"config yaml must be a mapping: {path}")
        cfg = cls()
        raw = dict(raw)

        for k in _ROOT_KEYS:                       # root scalars
            if k in raw:
                setattr(cfg, k, raw.pop(k))

        model = raw.pop("model", None) or {}       # model section
        if not isinstance(model, dict):
            raise ValueError("`model:` section must be a mapping")
        bad = set(model) - {"size"} - set(ARCH_KEYS)
        if bad:
            raise ValueError(f"unknown `model:` keys {sorted(bad)}; "
                             f"valid: size, {', '.join(ARCH_KEYS)}")
        cfg.model_size = str(model.pop("size", cfg.model_size))
        cfg.model_arch = {k: model[k] for k in ARCH_KEYS if k in model}

        data = raw.pop("data", None) or {}         # data section
        if not isinstance(data, dict):
            raise ValueError("`data:` section must be a mapping")
        bad = set(data) - set(_DATA_KEYS)
        if bad:
            raise ValueError(f"unknown `data:` keys {sorted(bad)}; "
                             f"valid: {', '.join(_DATA_KEYS)}")
        for yk, fk in _DATA_KEYS.items():
            if yk in data:
                setattr(cfg, fk, data[yk])

        for sec, keys in _SECTIONS.items():        # remaining sections
            vals = raw.pop(sec, None) or {}
            if not isinstance(vals, dict):
                raise ValueError(f"`{sec}:` section must be a mapping")
            bad = set(vals) - set(keys)
            if bad:
                raise ValueError(f"unknown `{sec}:` keys {sorted(bad)}; "
                                 f"valid: {', '.join(keys)}")
            for yk, fk in keys.items():
                if yk in vals:
                    setattr(cfg, fk, vals[yk])

        if raw:                                    # anything left is a typo
            raise ValueError(f"unknown config keys/sections {sorted(raw)}; "
                             f"valid sections: {', '.join(_ALL_SECTIONS)} "
                             f"(root: {', '.join(_ROOT_KEYS)})")
        return cfg

    # ---- Derived paths ----
    @property
    def train_npz(self) -> str:
        """Training set (written by the dataset generation stage)."""
        return str(Path(self.dataset_dir) / "train.npz")

    @property
    def val_npz(self) -> str:
        """Validation set (held-out episodes, split by episode)."""
        return str(Path(self.dataset_dir) / "val.npz")

    @property
    def test_npz(self) -> str:
        """Fixed closed-loop test set: one initial scenario per episode."""
        return str(Path(self.dataset_dir) / "test.npz")

    @property
    def ood_npz(self) -> str:
        """OOD test set (seed + 80,000 + i)."""
        return str(Path(self.dataset_dir) / "ood_test.npz")

    @property
    def pretrain_synth_npz(self) -> str:
        """Stage-0 pretraining set, component 1: synthetic goal pairs."""
        return str(Path(self.dataset_dir) / "pretrain_synth.npz")

    @property
    def pretrain_play_npz(self) -> str:
        """Stage-0 pretraining set, component 2: play dynamics pairs."""
        return str(Path(self.dataset_dir) / "pretrain_play.npz")

    @property
    def pretrain_hf_dir(self) -> str:
        """HF-format pretrained WAM directory."""
        return str(Path(self.pretrain_dir) / "pocketwam-3m-pretrain")

    @property
    def demos_dir(self) -> str:
        """Expert demo replay GIFs from the dataset generation stage."""
        return str(Path(self.dataset_dir) / "demos")

    @property
    def hf_dir(self) -> str:
        """HF-format model directory: <train.output_dir>/pocketwam-<size>."""
        return str(Path(self.train_dir) / f"pocketwam-{self.model_size}")

    @property
    def train_log_path(self) -> str:
        return str(Path(self.train_dir) / "train_log.json")

    @property
    def loss_curve_path(self) -> str:
        return str(Path(self.train_dir) / "loss_curve.png")

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        arch = ", ".join(f"{k}={self.model_arch[k]}" for k in ARCH_KEYS
                         if k in self.model_arch) or "preset defaults"
        return (f"model={self.model_size} ({arch})  dataset_dir={self.dataset_dir}  "
                f"train_dir={self.train_dir}  eval_dir={self.eval_dir}  "
                f"seed={self.seed}  lr={self.lr:.1e}  epochs={self.epochs}")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> torch.device:
    """Compute device: CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_banner(device: torch.device) -> str:
    if device.type == "cuda":
        return f"Accelerator: {torch.cuda.get_device_name(0)} (via torch.cuda)"
    return "Accelerator: CPU"
