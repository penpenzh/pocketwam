"""Unified CLI: python3 -m pwam <command> [--config <yaml>]

Commands: collect (dataset gen) / pretrain (Stage-0 video-only) / train
(Stage-1 SFT, init from Stage-0) / evaluate (closed-loop; --ckpt alone works)
/ all (pretrain -> train -> evaluate).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import Config, device_banner, get_device


def _add_common(p: argparse.ArgumentParser, config_required: bool = True):
    p.add_argument("--config", "-c", type=str, default=None, required=config_required,
                   help="experiment yaml (one file = one model: arch + dirs + hyperparameters; "
                        "optional for 'collect' and for 'evaluate' with --ckpt)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="smoke-test override: redirect dataset/train/eval outputs to "
                        "<dir>/dataset, <dir>/train, <dir>/eval (yaml stays untouched)")
    p.add_argument("--seed", type=int, default=None,
                   help="override the seed (yaml value or the built-in default)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pwam",
        description="PocketWAM: lightweight World Action Model pipeline "
                    "(joint future-world-state + action generation: dataset gen / train / evaluate)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect",
                       help="WAM dataset generation: expert demos + play/warm-start "
                            "expansion -> dataset/{train,val,test,ood_test,pretrain_*}.npz "
                            "(no yaml needed; built-in defaults + CLI overrides)")
    _add_common(p, config_required=False)
    p.add_argument("--dataset-dir", type=str, default=None,
                   help="override the dataset output directory "
                        "(default: dataset; or data.dataset_dir when --config is given)")
    p.add_argument("--episodes", type=int, default=None,
                   help="override the number of demo episodes (default 10000)")
    p.add_argument("--val-frac", type=float, default=None,
                   help="override the validation fraction, split by episode (default 0.30)")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="override the number of fixed test scenarios (default 500)")

    p = sub.add_parser("pretrain",
                       help="Stage 0: world-model pretraining (video-only loss on "
                            "synthetic goals + play dynamics; saves the pretrained "
                            "backbone referenced by train.init_ckpt)")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=None, help="override pretrain.epochs")

    p = sub.add_parser("train", help="SFT training (joint denoising + inverse dynamics)")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    p.add_argument("--lr", type=float, default=None, help="override train.lr")

    p = sub.add_parser("evaluate",
                       help="closed-loop eval on the fixed test set (one sweep: stats + "
                            "per-scenario replay GIFs with the imagined final state + "
                            "summary figures); with only --ckpt it runs standalone")
    _add_common(p, config_required=False)
    p.add_argument("--episodes", type=int, default=None,
                   help="override eval_episodes (number of test scenarios)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint directory to evaluate (overrides eval.ckpt in "
                        "the yaml; with no --config this is all you need to pass "
                        "for a standalone evaluation)")
    p.add_argument("--gifs", type=int, default=None,
                   help="override eval.gifs (number of random replay GIFs)")
    p.add_argument("--steps", type=int, default=None,
                   help="override the denoising steps at inference (1 = Flash mode)")
    p.add_argument("--test-npz", type=str, default=None,
                   help="test scenario file to evaluate (default: "
                        "<dataset_dir>/test.npz; pass dataset/ood_test.npz for "
                        "the OOD set)")
    p.add_argument("--no-gif", action="store_true",
                   help="skip per-scenario GIF export (summary figures still drawn)")

    p = sub.add_parser("all", help="train -> evaluate for one yaml (no dataset collection; "
                                   "run `collect` first)")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="override eval episodes")
    return parser


def _load_cfg(args) -> Config:
    if args.config is not None:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()   # built-in defaults
    if args.out_dir is not None:
        root = Path(args.out_dir)
        cfg.dataset_dir = str(root / "dataset")
        cfg.train_dir = str(root / "train")
        cfg.eval_dir = str(root / "eval")
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def _banner(title: str):
    print("\n" + "=" * 62 + f"\n {title}\n" + "=" * 62)


def main():
    args = build_parser().parse_args()
    cfg = _load_cfg(args)

    # per-command CLI overrides
    if args.cmd == "collect":
        if args.episodes is not None:
            cfg.episodes_collect = args.episodes
        if args.eval_episodes is not None:
            cfg.eval_episodes = args.eval_episodes
        if args.val_frac is not None:
            cfg.val_frac = args.val_frac
        if args.dataset_dir is not None:
            cfg.dataset_dir = args.dataset_dir
    elif args.cmd == "pretrain":
        if args.epochs is not None:
            cfg.pretrain_epochs = args.epochs
    elif args.cmd == "train":
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.lr is not None:
            cfg.lr = args.lr
    elif args.cmd == "evaluate":
        if args.episodes is not None:
            cfg.eval_episodes = args.episodes
        if args.gifs is not None:
            cfg.eval_gifs = args.gifs
        if args.steps is not None:
            cfg.eval_denoise_steps = args.steps
        # standalone: derive eval dir from the checkpoint
        if args.config is None and args.ckpt is not None:
            ckpt_root = Path(args.ckpt).resolve()
            if cfg.eval_dir is None:
                cfg.eval_dir = str(ckpt_root / ("eval_ood" if args.test_npz else "eval"))
    elif args.cmd == "all":
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.eval_episodes is not None:
            cfg.eval_episodes = args.eval_episodes

    print(device_banner(get_device()))
    if args.config is not None:
        print(f"[cfg] {args.config}:  {cfg.summary()}")
    elif args.cmd == "evaluate":
        print(f"[cfg] no yaml: standalone evaluation on built-in defaults  "
              f"(test set={args.test_npz or cfg.test_npz}  episodes={cfg.eval_episodes}  "
              f"gifs={cfg.eval_gifs}  seed={cfg.seed})")
    else:
        print(f"[cfg] no yaml: dataset generation on built-in defaults  "
              f"(dataset_dir={cfg.dataset_dir}  episodes={cfg.episodes_collect}  "
              f"val_frac={cfg.val_frac}  eval_episodes={cfg.eval_episodes}  "
              f"seed={cfg.seed})")

    if args.cmd == "collect":
        from .data.collect import collect
        _banner("WAM dataset generation (train / val / test / ood_test / pretrain)")
        collect(cfg)

    elif args.cmd == "pretrain":
        from .train import pretrain
        _banner("Stage 0: world-model pretraining (video-only: goals + dynamics)")
        pretrain(cfg)

    elif args.cmd == "train":
        from .train import train
        _banner("Stage 1: WAM SFT (joint denoising + inverse dynamics)")
        train(cfg)

    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        _banner("WAM closed-loop evaluation (pick & place)")
        evaluate(cfg, ckpt_path=args.ckpt, save_gifs=not args.no_gif,
                 n_gifs=cfg.eval_gifs, test_npz=args.test_npz)

    elif args.cmd == "all":
        from .evaluate import evaluate
        from .train import pretrain, train


        t0 = time.time()
        _banner("Stage 0/3: world-model pretraining (video-only)")
        pretrain(cfg)
        _banner("Stage 1/3: WAM SFT (joint denoising + inverse dynamics)")
        train(cfg)
        _banner("Stage 2/3: WAM closed-loop evaluation (fixed test set)")
        evaluate(cfg, n_gifs=cfg.eval_gifs)
        _banner(f"Full pipeline done in {time.time() - t0:.1f}s")
        _list_artifacts(cfg)


def _list_artifacts(cfg: Config):
    print("Artifacts:")
    for root in (cfg.dataset_dir, cfg.train_dir, cfg.eval_dir):
        d = Path(root)
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file():
                print(f"  {f}  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
