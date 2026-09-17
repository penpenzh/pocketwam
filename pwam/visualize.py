"""Summary figures (matplotlib Agg): dataset grid / training curves / eval
summary / trajectory comparison / dream strip. Called in-flight by the
collect / train / evaluate stages."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import BIN_RIM, COLOR_RGB  # noqa: E402


def compare_grid(rows: list, out_path: str, col_titles: tuple,
                 suptitle: str = "", title_fs: int = 10) -> None:
    """rows: (img1, img2, img3, row_caption) — 3 titled columns per sample."""
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(8.6, 2.55 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (a, b, c, cap) in enumerate(rows):
        for k, im in enumerate((a, b, c)):
            ax = axes[r][k]
            ax.imshow(im)
            if r == 0:
                ax.set_title(col_titles[k], fontsize=title_fs)
            ax.axis("off")
        axes[r][0].text(-0.06, 0.5, cap, transform=axes[r][0].transAxes,
                        fontsize=8, va="center", ha="right", wrap=True,
                        color="#20242c")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[viz] comparison grid ({len(rows)} rows) -> {out_path}")


def dataset_grid(data, out_path: str, n: int = 12, seed: int = 0) -> None:
    """Dataset grid: observation | episode final-state target."""
    if isinstance(data, (str, Path)):
        data = dict(np.load(str(data), allow_pickle=False))
    N = len(data["images"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n, N), replace=False)
    cols = 2
    rows = len(idx)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6, 3.0 * rows))
    axes = np.atleast_2d(axes)
    for k, i in enumerate(idx):
        ax = axes[k][0]
        ax.imshow(data["images"][i])
        a = data["actions"][i]
        grip = "CLOSE" if a[2] > 0 else "open"
        ax.set_title(f'obs t: "{str(data["instruction"][i])}"\n'
                     f'a=({a[0]:+.2f},{a[1]:+.2f}) grip={grip}', fontsize=8)
        ax.axis("off")
        ax = axes[k][1]
        ax.imshow(data["ep_final"][data["ep_id"][i]])
        ax.set_title("final-state target (imagined by the WAM)", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"WAM dataset samples (N={N}): observation -> imagined final state", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] dataset grid -> {out_path}")


def simple_curve(log: list[dict], out_path: str) -> None:
    """Stage-0 pretraining curve: train loss + val video MSE (log axis)."""
    if isinstance(log, (str, Path)):
        log = json.loads(Path(log).read_text())
    epochs = [d["epoch"] for d in log]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(epochs, [d["train_loss"] for d in log], lw=1.6, label="train")
    ax.plot(epochs, [d["val_mse"] for d in log], lw=1.6, label="val video MSE")
    best_i = int(np.argmin([d["val_mse"] for d in log]))
    ax.axvline(epochs[best_i], color="gray", ls=":", lw=1)
    ax.annotate(f"best val {log[best_i]['val_mse']:.5f}\n(epoch {epochs[best_i]})",
                xy=(epochs[best_i], log[best_i]["val_mse"]), xytext=(10, 30),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Stage-0 world-model pretraining (video-only)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] pretraining curve -> {out_path}")


def loss_curve(log: list[dict], out_path: str) -> None:
    """Training curves: total loss / world-model video / action panels."""
    if isinstance(log, (str, Path)):
        log = json.loads(Path(log).read_text())
    epochs = [d["epoch"] for d in log]
    train = [d["train"] for d in log]
    val = [d["val"] for d in log]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs = list(axs)

    ax = axs[0]
    ax.plot(epochs, train, lw=1.6, label="train")
    ax.plot(epochs, val, lw=1.6, label="val")
    best_i = int(np.argmin(val))
    ax.axvline(epochs[best_i], color="gray", ls=":", lw=1)
    ax.annotate(f"best val {val[best_i]:.4f}\n(epoch {epochs[best_i]})",
                xy=(epochs[best_i], val[best_i]), xytext=(10, 30),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("Total loss (joint flow matching)")
    ax.set_ylabel("MSE loss")

    ax = axs[1]
    ax.plot(epochs, [d["train_video"] for d in log], lw=1.4, label="train video MSE")
    ax.plot(epochs, [d["val_video_mse"] for d in log], lw=1.6, label="val video MSE")
    psnr = [d["val_video_psnr"] for d in log]
    ax2 = ax.twinx()
    ax2.plot(epochs, psnr, lw=1.6, color="tab:green", label="val PSNR (1-step)")
    ax2.set_ylabel("PSNR (dB)", color="tab:green")
    ax.set_title("World model: imagined final state")
    ax.set_ylabel("MSE (flow space)")

    ax = axs[2]
    ax.plot(epochs, [d["train_dq"] for d in log], lw=1.4, label="train dq")
    ax.plot(epochs, [d["val_dq_mse"] for d in log], lw=1.6, label="val dq")
    ax.plot(epochs, [d["val_grip_mse"] for d in log], lw=1.6, label="val grip")
    ax.set_title("Action (inverse dynamics)")
    ax.set_ylabel("MSE")

    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] training curve -> {out_path}")


def eval_summary(report: dict, out_path: str) -> None:
    """Per-color success/grasp bars + final ball-bin distance histogram."""
    per_color = report["per_color"]
    colors = list(per_color.keys())
    rates = [per_color[c]["success_rate"] for c in colors]
    grasps = [per_color[c]["grasp_rate"] for c in colors]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = np.arange(len(colors) + 1)
    all_rates = rates + [report["success_rate"]]
    all_grasps = grasps + [report["grasp_rate"]]
    b1 = ax.bar(x - 0.19, all_grasps, width=0.38,
                color=[tuple(v / 255 for v in COLOR_RGB[c]) for c in colors] + [(0.6, 0.6, 0.6)],
                label="grasp right ball")
    b2 = ax.bar(x + 0.19, all_rates, width=0.38,
                color=[tuple(v / 255 for v in COLOR_RGB[c]) for c in colors] + [(0.6, 0.6, 0.6)],
                alpha=0.45, label="placed in bin (success)")
    for b, v in list(zip(b1, all_grasps)) + list(zip(b2, all_rates)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(colors + ["overall"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate")
    ax.set_title(f"Grasp / place by commanded color (N={report['episodes']})")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    dists = report["final_dists"]
    ax.hist(dists, bins=30, color=(0.4, 0.6, 0.9))
    v = report.get("place_dist_thresh")
    if v is not None:
        ax.axvline(v, color="crimson", ls="--",
                   label=f"place dist thresh={v:.3f}")
    wm = report.get("world_model", {})
    if wm:
        ax.set_xlabel(f"final ball-to-bin distance   "
                      f"(imagined-final-state PSNR {wm['mean_psnr']:.1f} dB)")
    else:
        ax.set_xlabel("final ball-to-bin distance")
    ax.set_title("Final ball-to-bin distance distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] eval summary -> {out_path}")


def traj_compare(report: dict, out_path: str, world_extent: float = 1.25,
                link_lengths=(0.55, 0.55)) -> None:
    """Expert vs WAM EE trajectories + commanded-ball path + bin."""
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    # bin
    bin_pos = np.array(report["bin_pos"])
    bin_rgb = tuple(v / 255 for v in BIN_RIM)
    circ = plt.Circle(bin_pos, report.get("bin_radius", 0.14),
                      facecolor=(0.16, 0.14, 0.12), edgecolor=bin_rgb, lw=2.2)
    ax.add_patch(circ)
    ax.text(bin_pos[0], bin_pos[1] + report.get("bin_radius", 0.14) + 0.06,
            "bin", ha="center", fontsize=9, color=bin_rgb)
    # balls (initial positions)
    for color, pos in report["balls"]:
        rgb = tuple(v / 255 for v in COLOR_RGB[color])
        circ = plt.Circle((pos[0], pos[1]), 0.075, color=rgb, alpha=0.45)
        ax.add_patch(circ)
        ax.text(pos[0], pos[1] + 0.13, color, ha="center", fontsize=9)
        if color == report["commanded"]:
            ring = plt.Circle((pos[0], pos[1]), 0.11, fill=False,
                              color="k", ls="--", lw=1.2)
            ax.add_patch(ring)
    # initial arm configuration
    q0 = np.array(report["q0"])
    l1, l2 = link_lengths
    p1 = np.array([l1 * np.cos(q0[0]), l1 * np.sin(q0[0])])
    p2 = p1 + np.array([l2 * np.cos(q0[0] + q0[1]), l2 * np.sin(q0[0] + q0[1])])
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], color="0.75", lw=2,
            marker="o", ms=4, label="arm @ start")
    # trajectories
    ee_vla = np.array(report["vla_ee_path"])
    ee_exp = np.array(report["expert_ee_path"])
    ball_rgb = tuple(v / 255 for v in COLOR_RGB[report["commanded"]])
    if "vla_ball_path" in report:
        ball_vla = np.array(report["vla_ball_path"])
        ax.plot(ball_vla[:, 0], ball_vla[:, 1], "-", color=ball_rgb, lw=2.4,
                alpha=0.85, label=f"{report['commanded']} ball path")
        ax.plot(*ball_vla[-1], marker="o", ms=8, color=ball_rgb)
    ax.plot(ee_exp[:, 0], ee_exp[:, 1], "--", color="0.5", lw=1.5,
            label=f"expert EE ({report['expert_steps']} steps)")
    ax.plot(ee_vla[:, 0], ee_vla[:, 1], "-", color="crimson", lw=1.6,
            label=f"WAM EE ({report['vla_steps']} steps)")
    ax.plot(*ee_vla[-1], marker="*", ms=12, color="crimson")
    ax.set_xlim(-world_extent, world_extent)
    ax.set_ylim(-world_extent, world_extent)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f'Pick&place: "{report["instruction"]}"  '
                 f'WAM {"SUCCESS" if report["vla_success"] else "FAIL"}')
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] trajectory comparison -> {out_path}")


def dream_strip(strip: dict, out_path: str) -> None:
    """Imagination across one episode: initial obs -> dreams over time ->
    the true final state."""
    first_obs = strip.get("first_obs")
    dreams = strip["dreams"]
    true_final = strip["true_final"]
    n = len(dreams) + 2 + (1 if first_obs is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 2.3))
    axes = np.atleast_1d(axes)
    k = 0
    if first_obs is not None:
        axes[k].imshow(first_obs)
        axes[k].set_title("initial obs", fontsize=8)
        k += 1
    for s, d in dreams:
        axes[k].imshow(d)
        axes[k].set_title(f"dream @ step {s}", fontsize=8)
        k += 1
    axes[k].imshow(true_final)
    axes[k].set_title("TRUE final", fontsize=8)
    k += 1
    axes[k].axis("off")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f'WAM imagination vs reality: "{strip["instruction"]}"  '
                 f'({"SUCCESS" if strip["success"] else "FAIL"})', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] dream strip -> {out_path}")
