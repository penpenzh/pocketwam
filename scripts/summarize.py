#!/usr/bin/env python3
"""Summarize a finished PocketWAM pipeline run: training log + test eval + OOD eval."""
import json
import sys
from pathlib import Path

root = Path("outputs_sft_wam_3m")

print("=" * 70)
print("TRAINING (outputs_sft_wam_3m/train/train_log.json)")
print("=" * 70)
log = json.load(open(root / "train" / "train_log.json"))
best = min(log, key=lambda e: e["val_ctrl"])
for e in log[::max(1, len(log) // 10)]:
    print(f"  ep {e['epoch']:2d}  train={e['train']:.4f}  video_mse={e['val_video_mse']:.4f} "
          f"psnr={e['val_video_psnr']:.1f}dB  dq={e['val_dq_mse']:.5f}  grip={e['val_grip_mse']:.4f}")
print(f"  ... best epoch {best['epoch']}: ctrl={best['val_ctrl']:.4f} "
      f"psnr={best['val_video_psnr']:.1f}dB dq={best['val_dq_mse']:.5f} grip={best['val_grip_mse']:.4f}")

for name, path in (("TEST (500 scenarios, identical to PocketVLA)", root / "eval" / "eval_report.json"),
                   ("OOD  (300 scenarios, identical to PocketVLA)", root / "eval_ood" / "eval_report.json")):
    print("=" * 70)
    print(name)
    print("=" * 70)
    if not path.is_file():
        print("  (missing)")
        continue
    r = json.load(open(path))
    wm = r["world_model"]
    print(f"  success rate     : {r['success_rate']:.1%}")
    print(f"  grasp right ball : {r['grasp_rate']:.1%}")
    print(f"  mean final dist  : {r['mean_final_ball_bin_dist']:.3f}")
    print(f"  steps if success : {r['mean_steps_if_success']}")
    print(f"  world model PSNR : {wm['mean_psnr']:.2f} dB  "
          f"(success {wm['psnr_if_success'] and round(wm['psnr_if_success'],2)} / "
          f"fail {wm['psnr_if_fail'] and round(wm['psnr_if_fail'],2)})")
    print(f"  failures         : {r['failure_taxonomy']}")
    for c, v in r["per_color"].items():
        print(f"    {c:5s}: success {v['success_rate']:6.1%}  grasp {v['grasp_rate']:6.1%}  N={v['n']}")
