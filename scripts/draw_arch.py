"""Draw the PocketWAM architecture figure (docs/imgs/arch_wam.png):
(a) one forward pass — encoders, the shared trunk, the two heads;
(b) the two-pass inference pattern (Pass A joint denoising -> Pass B IDM)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

C_CTX, C_GEN, C_TRUNK, C_OUT, C_HEAD = "#dce8f7", "#fdeede", "#e6e2f5", "#dff0e2", "#f5e8d9"


def box(ax, xy, w, h, text, fc, fs=8.6, ec="#2c3242", bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#20242c", fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color="#5a6478", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(15.4, 10.2))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 10.2)
    ax.axis("off")
    ax.text(7.7, 9.95, "PocketWAM-3M (3.23M) — final architecture: one shared trunk, two passes per step",
            ha="center", fontsize=13, fontweight="bold")

    # panel (a): one forward pass
    ax.text(0.25, 9.35, "(a) ONE FORWARD PASS (shared weights, used by both passes)",
            fontsize=10.5, color="#33507a", fontweight="bold")

    inputs = [
        ("observation 64x64x3", C_CTX),
        ("instruction (10 tokens)", C_CTX),
        ("proprio (6 floats)", C_CTX),
        ("goal canvas  x_t", C_GEN),
        ("action slot   a_t", C_GEN),
    ]
    y0 = 8.75
    for i, (t, c) in enumerate(inputs):
        box(ax, (0.3, y0 - i * 0.72), 2.5, 0.6, t, c)
    encoders = [
        ("CNN vision encoder -> 64 vision tokens", C_CTX),
        ("token embedding -> 10 text tokens", C_CTX),
        ("Linear -> 1 proprio token", C_CTX),
        ("Conv goal encoder -> 64 goal tokens", C_GEN),
        ("Linear -> 1 action token", C_GEN),
    ]
    for i, (t, c) in enumerate(encoders):
        box(ax, (3.2, y0 - i * 0.72), 3.15, 0.6, t, c, fs=7.9)
        arrow(ax, (2.85, y0 - i * 0.72 + 0.3), (3.2, y0 - i * 0.72 + 0.3))

    # trunk
    box(ax, (6.9, 5.15), 3.6, 4.2,
        "SHARED TRUNK\nPre-LN bidirectional\nTransformer x 6\n(d=192, 4 heads, d_ff=448)\n\n"
        "140 tokens:\n[ vision 64 | text 10 |\n  proprio 1 | goal 64 | action 1 ]\n\n"
        "full self-attention\n(text padding only mask)\n\n+ timestep t embedding\n  on goal/action tokens",
        C_TRUNK, fs=8.6)
    for i in range(5):
        arrow(ax, (6.4, y0 - i * 0.72 + 0.3), (6.9, 7.2), color="#5a6478")

    # heads
    box(ax, (11.1, 7.35), 3.9, 1.55,
        "GoalDecoder (conv upsample)\nreads [ goal states + vision states ]\n"
        "-> raw output: delta (the change)\nx1_hat = obs + delta  (imagined final state)",
        C_HEAD, fs=7.6)
    box(ax, (11.1, 5.35), 3.9, 1.55,
        "ActionHead (MLP 640->160->3, tanh)\nreads the ACTION-TOKEN hidden state\n"
        "-> action (dq1, dq2, grip)\n(the action token ATTENDED to the goal\n tokens inside the trunk)",
        C_HEAD, fs=7.6)
    arrow(ax, (10.5, 8.2), (11.1, 8.1), color="#2d6a3f")
    arrow(ax, (10.5, 6.1), (11.1, 6.1), color="#2d6a3f")
    ax.text(12.9, 4.95, "no direct delta->action link (action_delta_cond: false)",
            ha="center", fontsize=7.2, style="italic", color="#8a5a1e")

    # panel (b): two-pass inference
    ax.text(0.25, 4.35, "(b) INFERENCE per control step  (denoise_steps=1: exactly 2 forward calls)",
            fontsize=10.5, color="#8a5a1e", fontweight="bold")

    box(ax, (0.3, 2.6), 2.0, 0.6, "noise x = eps", C_GEN)
    box(ax, (0.3, 1.7), 2.0, 0.6, "noise a = eps_a", C_GEN)

    box(ax, (3.1, 1.45), 3.5, 2.35,
        "PASS A — joint denoising\n(t = 0)\n\nforward(obs, instr, proprio,\n        canvas=eps, slot=eps, t=0)\n\n"
        "-> x1_hat = obs + delta\n   (imagined final state)\n-> draft action",
        C_TRUNK, fs=8.0)

    box(ax, (7.7, 1.45), 3.5, 2.35,
        "PASS B — inverse dynamics\n(t = 1, same weights)\n\nforward_idm(obs, instr, proprio,\n        canvas=x1_hat, slot=draft, t=1)\n\n"
        "x1_hat re-encoded by the\ngoal encoder -> 64 goal tokens\n-> FINAL ACTION\n(Pass B's own delta discarded)",
        C_TRUNK, fs=8.0)

    box(ax, (12.2, 2.35), 2.9, 0.9, "FINAL ACTION\n(dq1, dq2, grip) -> environment", C_OUT, fs=8.0)
    box(ax, (12.2, 1.25), 2.9, 0.9, "imagined final state\n(display / debugging)", C_OUT, fs=8.0)

    arrow(ax, (2.3, 2.9), (3.1, 2.9), color="#8a5a1e")
    arrow(ax, (2.3, 2.0), (3.1, 2.1), color="#8a5a1e")
    # Pass A -> Pass B
    arrow(ax, (6.6, 2.9), (7.7, 2.75), color="#2d6a3f", lw=1.7)
    arrow(ax, (6.6, 2.1), (7.7, 2.15), color="#2d6a3f", lw=1.7)
    ax.text(7.15, 3.72, "x1_hat", ha="center", fontsize=8, color="#2d6a3f", fontweight="bold")
    ax.text(7.15, 1.32, "draft", ha="center", fontsize=8, color="#2d6a3f")
    arrow(ax, (11.2, 2.85), (12.2, 2.85), color="#2d6a3f")
    arrow(ax, (11.2, 2.0), (12.2, 1.75), color="#2d6a3f")

    ax.text(7.7, 0.35,
            "training: Pass A loss = MSE(obs+delta, true final) + action-draft loss (weighted w(t)=(1-t), changed-pixels x10, class-balanced grip); "
            "Pass B loss = final-action loss (canvas = true label 50% / own dream 50%)",
            ha="center", fontsize=8.6, style="italic", color="#20242c",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6f8", ec="#c9ccd4"))

    out = Path(__file__).resolve().parents[1] / "docs" / "imgs" / "arch_wam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"[arch] -> {out}")


if __name__ == "__main__":
    main()
