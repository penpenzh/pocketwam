"""WAM dataset generation -> dataset/{train, val, test, ood_test,
pretrain_synth, pretrain_play}.npz

train/val: per-step demo samples + the episode's FINAL-STATE image (the
world-model target). A fraction of episodes is warm-started into diverse
mid-task states: play (random pre-roll with expert relabeling), carry_right
(commanded ball held), carry_wrong (wrong ball held).

pretrain (Stage-0, video-only, no action labels): synthetic goals (random
scenes x ALL 3 commanded colors, goal states rendered by hindsight teleport)
+ play dynamics ((o_t, a_t, o_{t+1}) forward-dynamics pairs), disjoint seed
spaces (+500,000 / +700,000).

test.npz (seed+20,000+i) and ood_test.npz (seed+80,000+i) use exactly
PocketVLA's seed spaces. `python3 -m pwam collect`.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import Config, ensure_dir
from ..expert.analytic import AnalyticExpert, ik_2r
from ..lang.tokenizer import COLORS, TEMPLATES, TOKENIZER, make_instruction
from ..sim.arm2d import Arm2DPickPlace
from ..sim.render import SceneRenderer, annotate, save_gif
from .. import visualize

# seed offsets (disjoint spaces; test identical to PocketVLA)
TEST_SEED_OFFSET = 20_000
# pretraining seed spaces (disjoint from demos and scenarios)
SYNTH_GOAL_SEED_OFFSET = 500_000
PLAY_SEED_OFFSET = 700_000

# fields of one scenario record
SCEN_KEYS = ("balls", "bin_pos", "q0", "color_id", "tokens", "instruction")


def scenario_at(scen: dict, i: int) -> dict:
    """Row i of a scenario set, ready for env.reset_scenario(); keeps ALL
    fields (OOD sets carry extra distractor fields — filtering to SCEN_KEYS
    would silently drop them and turn OOD eval into ID eval)."""
    return {k: scen[k][i] for k in scen.keys()}


def _gen_scenarios(cfg: Config, n: int, seed_offset: int = TEST_SEED_OFFSET) -> dict:
    """Generate n fixed scenarios (scenario i uses env seed cfg.seed + offset + i;
    identical to PocketVLA)."""
    out: dict[str, list] = {k: [] for k in SCEN_KEYS}
    for ep in range(n):
        env = Arm2DPickPlace(cfg, seed=cfg.seed + seed_offset + ep)
        env.reset()
        out["balls"].append(np.array([p for _, p in env.balls], dtype=np.float64))
        out["bin_pos"].append(env.bin_pos.astype(np.float64))
        out["q0"].append(env.q.astype(np.float64))
        out["color_id"].append(np.int64(COLORS.index(env.commanded)))
        out["tokens"].append(env.tokens.copy())
        out["instruction"].append(env.instruction)
    return {
        "balls": np.stack(out["balls"]),                    # (n,3,2) red/green/blue order
        "bin_pos": np.stack(out["bin_pos"]),                # (n,2)
        "q0": np.stack(out["q0"]),                          # (n,2)
        "color_id": np.array(out["color_id"], dtype=np.int64),
        "tokens": np.stack(out["tokens"]),                  # (n,lang_len)
        "instruction": np.array(out["instruction"]),         # (n,) unicode
    }


# ---------------- WAM warm-start expansion ----------------

def _warm_start(env: Arm2DPickPlace, expert: AnalyticExpert, cfg: Config,
                rng: np.random.Generator, recorder) -> str:
    """Perturb one fresh episode into a diverse mid-task state; returns the
    warm-start type. All traversed states are recorded with expert labels."""
    kind = rng.choice(("play", "carry_right", "carry_wrong"),
                      p=(0.40, 0.30, 0.30))
    if kind == "play":
        # random pre-roll: every visited state recorded with the EXPERT action
        k = int(rng.integers(1, cfg.play_max_steps + 1))
        for _ in range(k):
            recorder.record(env, expert.act(env))
            a = np.array([
                rng.uniform(-cfg.action_max, cfg.action_max),
                rng.uniform(-cfg.action_max, cfg.action_max),
                1.0 if rng.random() < 0.35 else -1.0,
            ])
            env.step(a)
    else:
        # carry-start: attach a ball to the EE (commanded / wrong color)
        color = env.commanded if kind == "carry_right" else \
            str(rng.choice([c for c in COLORS if c != env.commanded]))
        env.force_hold(color)
    return str(kind)


class _Recorder:
    """Buffers one episode's samples; commits them once the episode succeeds."""

    def __init__(self):
        self.images: list[np.ndarray] = []
        self.tokens: list[np.ndarray] = []
        self.proprios: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.color_id = -1
        self.instruction = ""

    def record(self, env: Arm2DPickPlace, expert_action: np.ndarray):
        obs = env._obs()
        self.images.append(obs["image"])
        self.tokens.append(obs["tokens"])
        self.proprios.append(obs["proprio"])
        self.actions.append(np.array([
            expert_action[0] / env.cfg.action_max,
            expert_action[1] / env.cfg.action_max,
            expert_action[2]]))
        self.color_id = COLORS.index(env.commanded)
        self.instruction = env.instruction


# ---------------- Stage-0 world-model pretraining data ----------------

def _release_state(env: Arm2DPickPlace, cfg: Config, rng: np.random.Generator,
                    commanded: str):
    """Sample a FINAL state for the given commanded color: the ball teleported
    inside the bin (matching the expert's release distribution) and the arm at
    the IK release pose — rendered directly (hindsight goal synthesis)."""
    r = rng.uniform(0.02, 0.05)
    th = rng.uniform(0.0, 2.0 * np.pi)
    release = env.bin_pos + r * np.array([np.cos(th), np.sin(th)])
    q_goal = ik_2r(release, *env.cfg.link_lengths)
    balls_goal = [(c, release.copy() if c == commanded else p.copy())
                  for c, p in env.balls] + list(getattr(env, "distractor_balls", []))
    return env._obs_renderer.render(
        q_goal, env.cfg.link_lengths, balls_goal,
        bin_pos=env.bin_pos, bin_r=env.cfg.bin_radius,
        grip_closed=False, held=None, line_w=2, bg_style=0,
        extra_objects=getattr(env, "ood_objects", None))


def gen_pretrain(cfg: Config) -> dict:
    """Stage-0 pretraining set (video-only, no action labels), written as two
    streamed files: pretrain_synth.npz (synthetic goal pairs, all 3 colors per
    scene, 40% warm-started) + pretrain_play.npz (play dynamics pairs)."""
    rng = np.random.default_rng(cfg.seed + 1)
    t0 = time.time()

    # ---- synthetic goal pairs ----
    S = cfg.pretrain_scenes
    sg_scenes, sg_proprio = [], []
    sg_scene_idx, sg_color_id, sg_tokens, sg_goal = [], [], [], []
    for i in range(S):
        env = Arm2DPickPlace(cfg, seed=cfg.seed + SYNTH_GOAL_SEED_OFFSET + i)
        env.reset()
        # 40% warm-start into mid-task states
        if rng.random() < 0.40:
            kind = rng.choice(("play", "carry"))
            if kind == "play":
                for _ in range(int(rng.integers(1, cfg.play_max_steps + 1))):
                    env.step(np.array([
                        rng.uniform(-cfg.action_max, cfg.action_max),
                        rng.uniform(-cfg.action_max, cfg.action_max),
                        1.0 if rng.random() < 0.35 else -1.0]))
            else:
                color = str(rng.choice([c for c in COLORS if c != env.commanded])) \
                    if rng.random() < 0.5 else env.commanded
                env.force_hold(color)
        obs = env._obs()                       # after the optional warm start
        sg_scenes.append(obs["image"])
        sg_proprio.append(obs["proprio"])
        for cid, c in enumerate(COLORS):
            tid = int(rng.integers(len(TEMPLATES)))
            sg_scene_idx.append(i)
            sg_color_id.append(cid)
            sg_tokens.append(TOKENIZER.encode_padded(make_instruction(c, tid), cfg.lang_len))
            sg_goal.append(_release_state(env, cfg, rng, c))
        if (i + 1) % 5000 == 0:
            print(f"  synth-goal scene {i + 1}/{S}  ({time.time() - t0:.1f}s)")

    # ---- play dynamics pairs ----
    E = cfg.pretrain_play_episodes
    pl_images, pl_tokens, pl_proprio, pl_action, pl_goal = [], [], [], [], []
    for i in range(E):
        env = Arm2DPickPlace(cfg, seed=cfg.seed + PLAY_SEED_OFFSET + i)
        obs = env.reset()
        k = int(rng.integers(6, 21))
        for _ in range(k):
            a = np.array([
                rng.uniform(-cfg.action_max, cfg.action_max),
                rng.uniform(-cfg.action_max, cfg.action_max),
                1.0 if rng.random() < 0.30 else -1.0])
            pl_images.append(obs["image"])
            pl_tokens.append(obs["tokens"])
            pl_proprio.append(obs["proprio"])
            pl_action.append(np.array([a[0] / cfg.action_max,
                                        a[1] / cfg.action_max, a[2]]))
            obs, _, _, _ = env.step(a)
            pl_goal.append(obs["image"].copy())
        if (i + 1) % 3000 == 0:
            print(f"  play episode {i + 1}/{E}  ({time.time() - t0:.1f}s)")

    # ---- synthetic goals (streamed to disk, then freed) ----
    synth = {
        "sg_scenes": np.stack(sg_scenes).astype(np.uint8),          # (S,64,64,3)
        "sg_proprio": np.stack(sg_proprio).astype(np.float32),       # (S,6)
        "sg_scene_idx": np.array(sg_scene_idx, dtype=np.int64),   # (P,)
        "sg_color_id": np.array(sg_color_id, dtype=np.int64),
        "sg_tokens": np.stack(sg_tokens).astype(np.int64),           # (P,10)
        "sg_goal": np.stack(sg_goal).astype(np.uint8),               # (P,64,64,3)
    }
    del sg_scenes, sg_proprio, sg_scene_idx, sg_color_id, sg_tokens, sg_goal
    synth_path = Path(cfg.dataset_dir) / "pretrain_synth.npz"
    np.savez_compressed(synth_path, **synth)
    n_synth = len(synth["sg_goal"])
    del synth
    print(f"[collect] pretrain synth-goals -> {synth_path}  ({n_synth} pairs / "
          f"{S} scenes x3 colors, seed space +{SYNTH_GOAL_SEED_OFFSET}+i)")

    # ---- play dynamics ----
    play = {
        "pl_images": np.stack(pl_images).astype(np.uint8),         # (N,64,64,3)
        "pl_tokens": np.stack(pl_tokens).astype(np.int64),           # (N,10)
        "pl_proprio": np.stack(pl_proprio).astype(np.float32),      # (N,6)
        "pl_action": np.stack(pl_action).astype(np.float32),        # (N,3) executed
        "pl_goal": np.stack(pl_goal).astype(np.uint8),                # (N,64,64,3) next
    }
    del pl_images, pl_tokens, pl_proprio, pl_action, pl_goal
    play_path = Path(cfg.dataset_dir) / "pretrain_play.npz"
    np.savez_compressed(play_path, **play)
    n_play = len(play["pl_goal"])
    del play
    print(f"[collect] pretrain play-dynamics -> {play_path}  ({n_play} pairs, "
          f"seed space +{PLAY_SEED_OFFSET}+i; disjoint from SFT demos)")
    return {"synth_pairs": n_synth, "play_pairs": n_play, "total": n_synth + n_play}


def collect(cfg: Config) -> dict:
    """Dataset generation: WAM demo+play episodes -> train / val / test sets."""
    ds_dir = ensure_dir(cfg.dataset_dir)
    demos_dir = ensure_dir(Path(cfg.dataset_dir) / "demos")
    print(f"[collect] dataset dir: {cfg.dataset_dir}")
    print(f"[collect] collecting {cfg.episodes_collect} episodes "
          f"(warm_start_prob={cfg.warm_start_prob:.0%}: play / carry_right / carry_wrong) ...")
    env = Arm2DPickPlace(cfg, seed=cfg.seed)
    expert = AnalyticExpert(cfg, seed=cfg.seed + 1)
    warm_rng = np.random.default_rng(cfg.seed + 2)   # warm-start rng
    pretty = SceneRenderer(cfg.frame_res, cfg.world_extent)
    gif_eps = {0, cfg.episodes_collect // 2} if cfg.episodes_collect >= 2 else {0}

    images, tokens, proprios, actions = [], [], [], []
    color_ids, instructions, ep_ids, ep_finals = [], [], [], []
    ep_success, ep_steps = 0, []
    warm_counts = {"fresh": 0, "play": 0, "carry_right": 0, "carry_wrong": 0}

    t0 = time.time()
    committed_eps = 0
    for ep in range(cfg.episodes_collect):
        obs = env.reset()
        frames: list[np.ndarray] | None = [] if ep in gif_eps else None
        rec = _Recorder()
        kind = "fresh"
        if cfg.warm_start_prob > 0 and warm_rng.random() < cfg.warm_start_prob:
            kind = _warm_start(env, expert, cfg, warm_rng, rec)
        warm_counts[kind] += 1

        done = False
        while not done:
            a = expert.act(env)
            rec.record(env, a)
            obs, _, done, info = env.step(a)
            if frames is not None:
                frame = pretty.render_env(env, line_w=4)
                held = f"holding {env.held}" if env.held else "empty hand"
                frame = annotate(frame, [
                    env.instruction,
                    f"step {env.t}/{cfg.max_steps}  {env.phase}  {held}  "
                    f"ball={info['dist']:.2f}  bin={info['bin_dist']:.2f}"
                    + ("  PLACED!" if info["success"] else ""),
                ])
                frames.append(frame)

        ep_success += int(info["success"])
        ep_steps.append(info["steps"])
        if info["success"]:
            # commit episode: all its samples + its FINAL-STATE image
            n = len(rec.images)
            images.extend(rec.images)
            tokens.extend(rec.tokens)
            proprios.extend(rec.proprios)
            actions.extend(rec.actions)
            color_ids.extend([rec.color_id] * n)
            instructions.extend([rec.instruction] * n)
            ep_ids.extend([committed_eps] * n)
            ep_finals.append(obs["image"].copy())
            committed_eps += 1
            if frames is not None:
                # append the final state (the generation target) to the GIF
                frame = pretty.render_env(env, line_w=4)
                frame = annotate(frame, [
                    env.instruction, "FINAL STATE (world-model target)",
                    f"steps={info['steps']}  warm_start={kind}",
                ])
                frames.append(frame)
        elif frames is not None:
            frame = pretty.render_env(env, line_w=4)
            frame = annotate(frame, [env.instruction,
                                     f"FAILED episode (dropped)  steps={info['steps']}"])
            frames.append(frame)
        if frames is not None:
            save_gif(frames, str(demos_dir / f"expert_ep{ep}.gif"), fps=cfg.gif_fps)
        if (ep + 1) % 100 == 0:
            print(f"  episode {ep + 1}/{cfg.episodes_collect}  "
                  f"({time.time() - t0:.1f}s)")

    data = {
        "images": np.stack(images).astype(np.uint8),
        "tokens": np.stack(tokens).astype(np.int64),
        "proprio": np.stack(proprios).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),     # (N,3) normalized
        "color_id": np.array(color_ids, dtype=np.int64),
        "instruction": np.array(instructions),
        "ep_id": np.array(ep_ids, dtype=np.int32),            # local episode index
        "ep_final": np.stack(ep_finals).astype(np.uint8),     # (n_eps,64,64,3)
    }
    n_samples = len(data["images"])
    del images, tokens, proprios, actions, color_ids, instructions  # free memory

    # ---- episode-level train/val split (frozen by seed; no frame leakage) ----
    n_eps = len(data["ep_final"])
    eps = np.arange(n_eps)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(eps)
    n_val_ep = max(1, int(len(perm) * cfg.val_frac))
    val_eps = np.sort(perm[:n_val_ep])
    val_mask = np.isin(data["ep_id"], val_eps)
    train_idx, val_idx = np.where(~val_mask)[0], np.where(val_mask)[0]

    def _save_split(name: str, idx: np.ndarray, keep_eps: np.ndarray):
        remap = {int(e): i for i, e in enumerate(keep_eps)}
        out = {k: v[idx] for k, v in data.items() if k != "ep_final"}
        out["ep_id"] = np.array([remap[int(e)] for e in data["ep_id"][idx]],
                                dtype=np.int32)
        out["ep_final"] = data["ep_final"][keep_eps]
        out_path = ds_dir / f"{name}.npz"
        np.savez_compressed(out_path, **out)
        print(f"[collect] {name} set -> {out_path}  ({len(idx)} samples / "
              f"{len(keep_eps)} episodes)")

    train_eps = np.setdiff1d(eps, val_eps)
    _save_split("train", train_idx, train_eps)
    _save_split("val", val_idx, val_eps)

    # ---- fixed test set (identical seed space to PocketVLA) ----
    scen = _gen_scenarios(cfg, cfg.eval_episodes)
    np.savez_compressed(ds_dir / "test.npz", **scen)
    counts = {c: int(np.sum(scen["color_id"] == i)) for i, c in enumerate(COLORS)}

    # ---- OOD test set (seed+80,000+i) ----
    from ..scenarios import OOD_TEST_SEED_OFFSET, gen_ood_scenarios
    ood_test = gen_ood_scenarios(cfg, cfg.ood_test_episodes,
                                 seed_offset=OOD_TEST_SEED_OFFSET, seed=cfg.seed)
    np.savez_compressed(ds_dir / "ood_test.npz", **ood_test)
    print(f"[collect] OOD test scenarios -> {ds_dir / 'ood_test.npz'}  "
          f"({len(ood_test['q0'])} scenarios, seed space seed+{OOD_TEST_SEED_OFFSET}+i)")

    visualize.dataset_grid(data, str(ds_dir / "dataset_grid.png"), n=12)
    n_train_samples, n_val_samples = int(len(train_idx)), int(len(val_idx))
    del data, train_idx, val_idx      # free memory

    # ---- Stage-0 pretraining set (disjoint seeds) ----
    gen_pretrain(cfg)

    stats = {
        "episodes": cfg.episodes_collect,
        "episodes_committed": committed_eps,
        "samples": n_samples,
        "warm_start": warm_counts,
        "expert_success_rate": ep_success / cfg.episodes_collect,
        "mean_steps": float(np.mean(ep_steps)),
        "train_samples": n_train_samples,
        "val_samples": n_val_samples,
        "test_scenarios": int(len(scen["q0"])),
        "ood_scenarios": int(len(ood_test["q0"])),
        "seconds": time.time() - t0,
    }
    print(f"[collect] expert success rate={stats['expert_success_rate']:.1%}  "
          f"mean steps={stats['mean_steps']:.1f}")
    print(f"[collect] warm-start expansion: " +
          "  ".join(f"{k}={v}" for k, v in warm_counts.items()))
    print(f"[collect] train/val: {stats['train_samples']} / {stats['val_samples']} samples "
          f"({len(train_eps)} / {n_val_ep} episodes, split by episode)")
    print(f"[collect] test: {stats['test_scenarios']} fixed closed-loop scenarios  "
          f"colors {counts}")
    print(f"[collect] dataset ready in '{cfg.dataset_dir}'  elapsed={stats['seconds']:.1f}s")
    return stats
