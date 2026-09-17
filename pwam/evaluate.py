"""Closed-loop evaluation on the fixed test set (protocol identical to
PocketVLA): one sweep produces all artifacts under eval.output_dir —
case/ep{iii}.gif replays (observation | imagined final state),
eval_report.json, eval_summary.png and traj_compare.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import Config, device_banner, ensure_dir, get_device
from .data.collect import scenario_at
from .expert.analytic import AnalyticExpert
from .lang.tokenizer import COLORS
from .model.wam import PocketWAM
from .sim.arm2d import Arm2DPickPlace, overlap_dist_thresh
from .sim.render import SceneRenderer, add_title, annotate, save_gif
from . import visualize


# ---------------- closed-loop machinery ----------------

def obs_to_tensors(obs: dict, device: torch.device):
    img = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).to(device)
    tok = torch.from_numpy(obs["tokens"]).unsqueeze(0).to(device)
    pad = tok != 0
    prop = torch.from_numpy(obs["proprio"]).unsqueeze(0).to(device)
    return img, tok, pad, prop


@torch.no_grad()
def policy_step(model: PocketWAM, obs: dict, device: torch.device,
                cfg: Config, generator: torch.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """WAM joint generation: obs dict -> (action (3,), imagined final state
    (64,64,3) uint8). Runs in fp32 (the Euler integration is precision-sensitive)."""
    img, tok, pad, prop = obs_to_tensors(obs, device)
    goal_x, act = model.generate(img, tok, pad, prop,
                                  steps=cfg.eval_denoise_steps, generator=generator)
    a = act[0].float().cpu().numpy()
    action = np.array([a[0] * cfg.action_max, a[1] * cfg.action_max, a[2]])
    dream = model.to_image01(goal_x[0].float()).cpu().numpy()        # (3,64,64) [0,1]
    dream = (np.transpose(dream, (1, 2, 0)) * 255.0).round().astype(np.uint8)
    return action, dream


def _upscale(img64: np.ndarray, res: int) -> np.ndarray:
    """Nearest-neighbor upscale of a 64x64 uint8 image (for GIF panels)."""
    f = res // img64.shape[0]
    return np.kron(img64, np.ones((f, f, 1), dtype=np.uint8))


def _delta_colormap(dream_u8: np.ndarray, obs_u8: np.ndarray) -> np.ndarray:
    """Visualize the delta = imagined final state - current observation:
    RED = erase, GREEN = draw."""
    d = dream_u8.astype(np.int16) - obs_u8.astype(np.int16)   # (H,W,3) signed
    pos = np.clip(d, 0, None).max(-1)                 # draw strength per pixel
    neg = np.clip(-d, 0, None).max(-1)                 # erase strength per pixel
    out = np.zeros(d.shape, dtype=np.uint8)
    out[..., 0] = np.clip(24 + neg * 0.9, 0, 255)      # red: erase
    out[..., 1] = np.clip(27 + pos * 0.9, 0, 255)      # green: draw
    out[..., 2] = 34
    return out


def _prediction_panel(first_obs: np.ndarray, first_dream: np.ndarray,
                      true_final: np.ndarray, cfg: Config) -> np.ndarray:
    """4-panel comparison: input | delta | imagined final | ground truth."""
    p1 = add_title(_upscale(first_obs, cfg.frame_res), "INPUT: initial observation", 10)
    p2 = add_title(_upscale(_delta_colormap(first_dream, first_obs), cfg.frame_res),
                   "MODEL OUTPUT 1: delta (red=erase, green=draw)", 10)
    p3 = add_title(_upscale(first_dream, cfg.frame_res),
                   "MODEL OUTPUT 2: imagined final (obs + delta)", 10)
    p4 = add_title(_upscale(true_final, cfg.frame_res),
                   "GROUND TRUTH: reached final state", 10)
    return np.concatenate([p1, p2, p3, p4], axis=1)


def run_episode(env: Arm2DPickPlace, model: PocketWAM, device: torch.device,
                cfg: Config, pretty: SceneRenderer, want_frames: bool,
                generator: torch.Generator | None = None,
                scenario: dict | None = None):
    """Closed-loop rollout of one episode (scenario != None replays a fixed
    scenario). Returns (frames|None, EE path, ball path, q0, info, dreams,
    true_final)."""
    obs = env.reset_scenario(scenario) if scenario is not None else env.reset()
    env.bg_style = int(scenario.get("bg_style", 0)) if scenario is not None else 0
    first_obs = obs["image"].copy()
    q0 = env.q.copy()
    ee_path = [env.ee.copy()]
    ball_path = [env.target_pos.copy()]
    frames: list[np.ndarray] | None = [] if want_frames else None
    dreams: list[tuple[int, np.ndarray]] = []      # (step, imagined final state)
    done = False
    while not done:
        a, dream = policy_step(model, obs, device, cfg, generator)
        dreams.append((env.t, dream))
        obs_in = obs["image"].copy()          # this step's delta + dream input
        obs, _, done, info = env.step(a)
        ee_path.append(env.ee.copy())
        ball_path.append(env.target_pos.copy())
        if frames is not None:
            p1 = add_title(_upscale(obs_in, cfg.frame_res), "INPUT: observation", 10)
            p2 = add_title(_upscale(_delta_colormap(dream, obs_in), cfg.frame_res),
                           "MODEL OUTPUT 1: delta (red=erase, green=draw)", 10)
            p3 = add_title(_upscale(dream, cfg.frame_res),
                           "MODEL OUTPUT 2: imagined final (obs + delta)", 10)
            panel = np.concatenate([p1, p2, p3], axis=1)
            held = f"holding {env.held}" if env.held else "empty hand"
            panel = annotate(panel, [
                obs["instruction"],
                f"step {env.t}/{env.cfg.max_steps}  "
                f"{env.phase}  {held}  ball={info['dist']:.2f}  bin={info['bin_dist']:.2f}"
                + ("  PLACED!" if info["success"] else ""),
                "middle = raw output delta (what must change) | right = imagined final state (obs + delta)",
            ])
            frames.append(panel)
    if frames is not None:
        # final frame: final prediction vs the truth
        if info["success"]:
            outcome = "RESULT: SUCCESS (ball placed in the bin)"
        elif info["grasped_wrong"] and not info["grasped_right"]:
            outcome = "RESULT: FAIL (grasped wrong ball)"
        elif info["grasped_right"]:
            outcome = "RESULT: FAIL (not placed in the bin)"
        else:
            outcome = "RESULT: FAIL (never grasped right ball)"
        d0 = _delta_colormap(dreams[0][1], first_obs)
        p1 = add_title(_upscale(d0, cfg.frame_res),
                       "MODEL OUTPUT 1: delta (initial step)", 10)
        p2 = add_title(_upscale(dreams[0][1], cfg.frame_res),
                       "MODEL OUTPUT 2: imagined final state", 10)
        p3 = add_title(_upscale(obs["image"], cfg.frame_res),
                       "GROUND TRUTH: reached final state", 10)
        frames.append(annotate(np.concatenate([p1, p2, p3], axis=1), [
            obs["instruction"], outcome,
            f"steps={env.t}  final ball-bin dist={info['ball_bin_dist']:.3f}",
            "left = raw delta | middle = final predicted output (obs + delta) | right = what actually happened",
        ]))
    return (frames, np.array(ee_path), np.array(ball_path), q0, info, dreams,
            obs["image"].copy(), first_obs)


def _fail_reason(info: dict) -> str:
    if info["success"]:
        return ""
    if not info["grasped_right"] and not info["grasped_wrong"]:
        return "never_grasped"
    if not info["grasped_right"]:
        return "grasped_wrong"
    return "grasped_but_not_placed"


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) / 255.0 - b.astype(np.float64) / 255.0) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10))), mse


def _resolve_ckpt(cfg: Config, ckpt_path: str | None) -> tuple[str, str]:
    """Resolve the checkpoint: CLI --ckpt wins over eval.ckpt (yaml)."""
    resolved = ckpt_path or cfg.eval_ckpt
    if not resolved:
        raise ValueError(
            "[eval] checkpoint not specified: set eval.ckpt in the yaml or pass "
            "--ckpt (e.g. --ckpt outputs_sft_wam_3m/train/pocketwam-3m)")
    if not Path(resolved).is_dir():
        raise FileNotFoundError(f"[eval] checkpoint directory not found: {resolved}")
    source = "via --ckpt" if ckpt_path else "via eval.ckpt (yaml)"
    return resolved, source


# ---------------- single-sweep evaluation ----------------

def evaluate(cfg: Config, ckpt_path: str | None = None,
             n_episodes: int | None = None, save_gifs: bool = True,
             n_gifs: int | None = None, gif_seed: int = 0,
             test_npz: str | None = None) -> dict:
    """Evaluate the fixed test set once: stats, GIFs and figures in one sweep.
    test_npz overrides the test-set file."""
    device = get_device()
    print(f"[eval] {device_banner(device)}")
    test_file = test_npz or cfg.test_npz
    if not Path(test_file).is_file():
        raise FileNotFoundError(
            f"[eval] test set not found: {test_file}\n"
            f"[eval]   run 'python3 -m pwam collect --config <yaml>' first")
    path, how = _resolve_ckpt(cfg, ckpt_path)
    print(f"[eval] checkpoint: {path}  ({how})")
    model, _ = PocketWAM.load_hf(path, device)
    size = model.config.size
    steps = cfg.eval_denoise_steps or model.config.denoise_steps
    print(f"[eval] model loaded: size={size}  params={model.n_params() / 1e6:.2f}M  "
          f"denoise_steps={steps}")
    if cfg.eval_dir is None:
        eval_root = None
        case_dir = None
    else:
        eval_root = Path(cfg.eval_dir)
        case_dir = eval_root / "case"          # one replay GIF per scenario
        if save_gifs:
            ensure_dir(case_dir)
    print(f"[eval] artifacts -> {cfg.eval_dir}")

    n = cfg.eval_episodes if n_episodes is None else n_episodes
    scen = dict(np.load(test_file, allow_pickle=True))   # object arrays (OOD)
    if len(scen["q0"]) < n:
        print(f"[eval] test set holds {len(scen['q0'])} scenarios (< {n} requested); "
              f"using all of them")
    n = min(n, len(scen["q0"]))
    print(f"[eval] fixed test set: {test_file}  ({n} scenarios)")
    env = Arm2DPickPlace(cfg, seed=cfg.seed)   # rng unused: scenarios are replayed
    pretty = SceneRenderer(cfg.frame_res, cfg.world_extent)
    expert = AnalyticExpert(cfg, seed=0, noise=0.0)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(cfg.seed)

    per_color = {c: {"success": 0, "n": 0, "grasp": 0, "steps": [],
                     "ball_bin": []} for c in COLORS}
    final_dists, succ_steps, results = [], [], []
    fails = {"never_grasped": 0, "grasped_wrong": 0, "grasped_but_not_placed": 0}
    comparison = None
    wm_psnr, wm_mse = [], []          # world-model: step-0 imagination vs true final
    wm_psnr_succ, wm_psnr_fail = [], []
    dream_strip = None

    # random GIF subset (None = every scenario)
    if save_gifs and n_gifs is not None and n_gifs < n:
        gif_eps = set(np.random.default_rng(gif_seed).choice(n, size=n_gifs,
                                                             replace=False).tolist())
        print(f"[eval] saving {n_gifs} random replay GIFs (of {n} scenarios)")
    else:
        gif_eps = None if save_gifs else set()

    for ep in range(n):
        sc = scenario_at(scen, ep)
        want_gif = save_gifs and (gif_eps is None or ep in gif_eps)
        frames, ee_path, ball_path, q0, info, dreams, true_final, first_obs = run_episode(
            env, model, device, cfg, pretty, want_frames=want_gif,
            generator=generator, scenario=sc)

        # world-model quality: step-0 imagination vs the true final state
        p, m = _psnr(dreams[0][1], true_final)
        wm_psnr.append(p)
        wm_mse.append(m)
        (wm_psnr_succ if info["success"] else wm_psnr_fail).append(p)

        c = info["commanded"]
        per_color[c]["n"] += 1
        per_color[c]["success"] += int(info["success"])
        per_color[c]["grasp"] += int(info["grasped_right"])
        per_color[c]["ball_bin"].append(info["ball_bin_dist"])
        final_dists.append(info["ball_bin_dist"])
        if info["success"]:
            per_color[c]["steps"].append(info["steps"])
            succ_steps.append(info["steps"])
        else:
            reason = _fail_reason(info)
            fails[reason] += 1
        results.append({
            "episode": ep, "instruction": info["instruction"], "commanded": c,
            "steps": info["steps"], "success": info["success"],
            "final_ball_bin_dist": info["ball_bin_dist"],
            "wm_step0_psnr": p,
        })

        if want_gif and case_dir is not None:
            save_gif(frames, str(case_dir / f"ep{ep:03d}.gif"), fps=cfg.gif_fps)
            # static comparison: input | final prediction | ground truth
            Image.fromarray(_prediction_panel(first_obs, dreams[0][1],
                                             true_final, cfg)).save(
                str(case_dir / f"ep{ep:03d}_prediction.png"))
        if ep == 0:
            # zero-noise expert reference on the same fixed scenario
            exp = expert.rollout(env, scenario=sc)
            comparison = {
                "instruction": info["instruction"], "commanded": c,
                "balls": [[col, sc["balls"][i].tolist()] for i, col in enumerate(COLORS)],
                "bin_pos": sc["bin_pos"].tolist(), "bin_radius": cfg.bin_radius,
                "q0": sc["q0"].tolist(),
                "vla_ee_path": ee_path.tolist(), "vla_ball_path": ball_path.tolist(),
                "expert_ee_path": exp["ee_path"].tolist(),
                "vla_steps": info["steps"], "expert_steps": exp["steps"],
                "vla_success": info["success"], "expert_success": exp["success"],
            }
            if eval_root is not None:
                visualize.traj_compare(comparison, str(eval_root / "traj_compare.png"))
                # dream strip: imagination evolution over the episode
                dream_strip = {
                    "instruction": info["instruction"],
                    "dreams": [(s, d) for s, d in dreams
                               if s % max(1, len(dreams) // 8) == 0 or s == dreams[-1][0]],
                    "true_final": true_final,
                    "first_obs": first_obs,
                    "success": info["success"],
                }
                visualize.dream_strip(dream_strip, str(eval_root / "dream_strip.png"))
        if (ep + 1) % 25 == 0:
            print(f"  episode {ep + 1}/{n}")

    n_success = sum(v["success"] for v in per_color.values())
    n_grasp = sum(v["grasp"] for v in per_color.values())
    report = {
        "episodes": n,
        "test_set": test_file,
        "model_size": size,
        "denoise_steps": steps,
        "success_rate": n_success / n,
        "grasp_rate": n_grasp / n,
        "place_overlap_frac": cfg.place_overlap_frac,
        "place_dist_thresh": overlap_dist_thresh(cfg.place_overlap_frac,
                                                 cfg.ball_radius, cfg.bin_radius),
        "mean_final_ball_bin_dist": float(np.mean(final_dists)),
        "mean_steps_if_success": float(np.mean(succ_steps)) if succ_steps else None,
        "final_dists": [float(d) for d in final_dists],
        "failure_taxonomy": fails,
        "world_model": {
            "note": "PSNR of the step-0 imagined final state vs the episode's true "
                    "final observation (video prediction quality)",
            "mean_psnr": float(np.mean(wm_psnr)),
            "mean_mse": float(np.mean(wm_mse)),
            "psnr_if_success": float(np.mean(wm_psnr_succ)) if wm_psnr_succ else None,
            "psnr_if_fail": float(np.mean(wm_psnr_fail)) if wm_psnr_fail else None,
        },
        "per_color": {
            c: {
                "success_rate": v["success"] / v["n"] if v["n"] else 0.0,
                "grasp_rate": v["grasp"] / v["n"] if v["n"] else 0.0,
                "n": v["n"],
                "mean_final_ball_bin_dist": float(np.mean(v["ball_bin"])) if v["ball_bin"] else None,
                "mean_steps": float(np.mean(v["steps"])) if v["steps"] else None,
            }
            for c, v in per_color.items()
        },
        "results": results,
        "comparison": comparison,
        "ckpt": str(path),
    }
    if eval_root is not None:
        out = eval_root / "eval_report.json"
        out.write_text(json.dumps(report, indent=1))
        visualize.eval_summary(report, str(eval_root / "eval_summary.png"))
        print(f"[eval] report -> {out}")

    print(f"[eval] place success rate: {report['success_rate']:.1%}  "
          f"grasp right ball: {report['grasp_rate']:.1%}  "
          f"mean ball-bin dist: {report['mean_final_ball_bin_dist']:.4f}")
    wm = report["world_model"]
    psnr_succ = f"{wm['psnr_if_success']:.2f}dB" if wm["psnr_if_success"] is not None else "n/a"
    psnr_fail = f"{wm['psnr_if_fail']:.2f}dB" if wm["psnr_if_fail"] is not None else "n/a"
    print(f"[eval] world model (imagined final state): "
          f"PSNR={wm['mean_psnr']:.2f}dB  "
          f"(success {psnr_succ} / fail {psnr_fail})")
    for c in COLORS:
        v = report["per_color"][c]
        dist = f"{v['mean_final_ball_bin_dist']:.4f}" if v["n"] else "n/a"
        print(f"  {c:5s}: success {v['success_rate']:6.1%}  "
              f"grasp right {v['grasp_rate']:6.1%}  "
              f"ball-bin dist {dist}  (N={v['n']})")
    nf = n - n_success
    if nf > 0:
        print(f"[eval] failure attribution ({nf} episodes): "
              f"never grasped={fails['never_grasped']}  "
              f"wrong ball={fails['grasped_wrong']}  "
              f"not placed={fails['grasped_but_not_placed']}")
    if save_gifs and case_dir is not None:
        n_gifs_saved = n if gif_eps is None else len(gif_eps)
        print(f"[eval] artifacts: {n_gifs_saved} scenario GIFs -> {case_dir}  report -> {out}")
    elif eval_root is None:
        print(f"[eval] lightweight stats-only evaluation (no artifacts)")
    else:
        print(f"[eval] report -> {out}  (GIFs disabled)")
    return report
