"""
eval_forgetting.py

Closed-loop evaluation of a continual-learning checkpoint in the LIBERO
simulator, producing the stage x suite success-rate matrix and the forgetting
numbers derived from it.

WHAT YOU SEE WHILE IT RUNS
    Nothing graphical. The simulator (MuJoCo via robosuite) renders OFFSCREEN
    into numpy arrays that are fed to the model; there is no window, and on a
    cluster node there is no display anyway. You get:
      * one progress line per episode in the log,
      * a per-suite summary and, at the end, the stage x suite table,
      * results/forgetting_matrix.json (updated after every suite),
      * OPTIONAL rollout GIFs (--save-videos N = first N episodes per task),
      * with --smoke-test: results/sanity/orientation_check.png.

MODES
    --smoke-test     No model, no GPU needed. Imports LIBERO, builds one task
                      environment, steps it, saves an image so you can check
                      that the camera orientation matches the training data.
    --check-imports  Only tries `import libero...`; prints MISSING_MODULE=<x>
                      on failure (used by the sbatch wrapper).
    --show-results   Pretty-print an existing results JSON and exit.
    (default)        Evaluate --checkpoint on every suite in --suites.

BASELINE (out-of-the-box, zero-shot) ROW
    Run once, before any fine-tuned stage, pointing --checkpoint at the plain
    HF repo rather than a stage directory, and give it a --stage-name with
    "baseline" in it so it sorts first and is excluded from forgetting math:

        python eval_forgetting.py --checkpoint openvla/openvla-7b \\
            --stage-name baseline_zeroshot \\
            --suites libero_spatial libero_object libero_goal \\
            --results-json results/forgetting_matrix.json

    Writes into the SAME results JSON as the fine-tuned stages, so
    --show-results shows it as the leftmost column alongside stage0, stage1,
    etc. for a direct before/after comparison.

Conventions (verified against the LIBERO source; the image-flip and gripper
conventions are inferred from the dataset lineage -- see orientation_check.png):
    * task init states come from suite.get_task_init_states(), never a random
      reset, so every checkpoint sees the same 50 starting configurations;
    * 10 idle steps at the start let objects settle (as in OpenVLA's LIBERO eval);
    * per-suite step budgets follow OpenVLA's LIBERO eval;
    * the prompt is the same template used in training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Packages installed at runtime with `pip --target` (see run_eval_forgetting.sh)
# are appended, never prepended, so they can only FILL GAPS in the container and
# can never shadow torch/numpy.
_pydeps = os.environ.get("KT_PYDEPS")
if _pydeps and os.path.isdir(_pydeps) and _pydeps not in sys.path:
    sys.path.append(_pydeps)

UNNORM_KEY = "kappatune_libero"
SUITE_TO_BENCHMARK = {"libero_long": "libero_10"}     # project nickname -> LIBERO name

# Exact public LeRobot repositories used by the training pipeline. Keep this
# mapping local to evaluation so eval does not depend on a helper that may not
# exist in a particular train_continual.py revision.
LIBERO_REPO_IDS = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_long": "lerobot/libero_10_image",
}


def dataset_repo_id(namespace: str, suite: str) -> str:
    """Return the exact LeRobot repository ID for a LIBERO suite."""
    if suite in LIBERO_REPO_IDS:
        return LIBERO_REPO_IDS[suite]
    # Explicit fallback for custom datasets while preserving the CLI option.
    return f"{namespace}/{suite}"
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
NUM_WAIT_STEPS = 10
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                    help="Stage checkpoint directory to evaluate, OR a plain HF repo id "
                         "(e.g. openvla/openvla-7b) to evaluate the out-of-the-box, "
                         "zero-shot model as a baseline row. A baseline checkpoint has no "
                         "action_stats.json, so stats are computed fresh per-suite from "
                         "that suite's own data -- the same correct-normalization fallback "
                         "used for any checkpoint missing a suite's entry. Give it a "
                         "--stage-name containing 'baseline' (e.g. baseline_zeroshot) so "
                         "it sorts first and is excluded from forgetting computations.")
    p.add_argument("--suites", nargs="+", default=["libero_spatial"],
                    help="Suites to evaluate on (all suites trained so far).")
    p.add_argument("--stage-name", default=None,
                    help="Row label in the results matrix (default: checkpoint dir name).")
    p.add_argument("--episodes-per-task", type=int, default=5,
                    help="LIBERO provides 50 fixed init states per task.")
    p.add_argument("--task-ids", default=None,
                    help="Comma-separated subset of task ids, e.g. 0,1 (quick tests).")
    p.add_argument("--max-steps", type=int, default=None,
                    help="Override the per-suite step budget.")
    p.add_argument("--results-json", default="results/forgetting_matrix.json")
    p.add_argument("--save-videos", type=int, default=0,
                    help="Save GIFs of the first N episodes of every task.")
    p.add_argument("--video-dir", default="results/videos")
    p.add_argument("--sanity-dir", default="results/sanity")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-flip", action="store_true",
                    help="Do not rotate env images by 180 degrees.")
    p.add_argument("--hf-namespace", default="lerobot")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--check-imports", action="store_true")
    p.add_argument("--show-results", default=None, metavar="JSON")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable, no simulator / GPU needed)
# --------------------------------------------------------------------------- #

def stage_sort_key(name: str):
    # Baseline rows (zero-shot, pre-fine-tuning) sort first, before stage0,
    # regardless of what they're named -- "baseline" anywhere in the name
    # is enough, so "baseline_zeroshot", "baseline", etc. all work.
    if "baseline" in name.lower():
        return (-1, name)
    m = re.match(r"stage(\d+)", name)
    return (int(m.group(1)) if m else 10**9, name)


def is_continual_stage(name: str) -> bool:
    """True for proper stageN checkpoints, False for baseline/reference rows.
    Only continual stages participate in forgetting as a 'previous stage':
    forgetting measures whether a LATER fine-tuning stage degraded what an
    EARLIER one learned, which is a different question from 'did fine-tuning
    beat the zero-shot baseline at all'. Conflating the two would make a
    baseline row's low zero-shot score look like stage0 'forgot' something
    it never had in the first place."""
    return re.match(r"stage\d+", name) is not None


def compute_forgetting(success: dict) -> dict:
    """forgetting[stage][suite] = best success rate on `suite` at any EARLIER
    stage minus the success rate at this stage (0 = retained, >0 = forgotten,
    <0 = backward transfer). Undefined (omitted) for a suite's first stage."""
    stages = sorted(success, key=stage_sort_key)
    out: dict = {}
    for i, st in enumerate(stages):
        out[st] = {}
        for suite, sr in success[st].items():
            prev = [success[s][suite] for s in stages[:i]
                    if suite in success[s] and is_continual_stage(s)]
            if prev:
                out[st][suite] = max(prev) - sr
    return out


def postprocess_action(action, gripper_is_signed: bool) -> np.ndarray:
    """Model output -> LIBERO env action (gripper: -1 = open, +1 = close).
    gripper_is_signed=True : dataset gripper already in {-1,+1} -> binarize only.
    gripper_is_signed=False: OpenVLA-RLDS convention (0 = close, 1 = open in
                              [0,1]) -> rescale to [-1,1], binarize, invert."""
    a = np.array(action, dtype=np.float64).copy()
    if gripper_is_signed:
        a[-1] = 1.0 if a[-1] >= 0 else -1.0
    else:
        g = 2.0 * a[-1] - 1.0
        a[-1] = -(1.0 if g >= 0 else -1.0)
    return a


def format_table(success: dict, forgetting: dict, suites_order: list[str]) -> str:
    stages = sorted(success, key=stage_sort_key)
    suites = [s for s in suites_order if any(s in success[st] for st in stages)]
    w = max([len(s) for s in stages] + [12]) + 2
    head = "stage".ljust(w) + "".join(s.rjust(16) for s in suites)
    lines = ["Success rate (forgetting vs best earlier stage in brackets)", head,
             "-" * len(head)]
    for st in stages:
        row = st.ljust(w)
        for su in suites:
            if su in success[st]:
                cell = f"{success[st][su]:.2f}"
                if su in forgetting.get(st, {}):
                    cell += f" ({forgetting[st][su]:+.2f})"
            else:
                cell = "-"
            row += cell.rjust(16)
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LIBERO plumbing
# --------------------------------------------------------------------------- #

def ensure_libero_config() -> Path:
    """`import libero.libero` calls input() if ~/.libero/config.yaml is missing,
    which crashes any batch job (EOFError). Create the file ourselves, without
    importing libero.libero (find_spec only imports the parent namespace)."""
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero")))
    cfg = cfg_dir / "config.yaml"
    if cfg.exists():
        return cfg
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the libero package (libero.libero).")
    root = Path(list(spec.submodule_search_locations)[0])
    import yaml
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.dump({
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root.parent / "datasets"),
        "assets": str(root / "assets"),
    }))
    print(f"Created LIBERO config: {cfg}", flush=True)
    return cfg


def import_libero():
    ensure_libero_config()
    import torch
    # LIBERO calls torch.load(path) on pickled numpy init states; torch>=2.6
    # defaults to weights_only=True and rejects that. Restore the old default.
    if not getattr(torch.load, "_kt_patched", False):
        _orig = torch.load

        def _load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig(*a, **k)
        _load._kt_patched = True
        torch.load = _load
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as e:
        print(f"MISSING_MODULE={(e.name or '').split('.')[0]}", flush=True)
        raise
    return benchmark, get_libero_path, OffScreenRenderEnv


def make_env(OffScreenRenderEnv, get_libero_path, task, seed):
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env


def env_image(obs, flip: bool) -> np.ndarray:
    img = obs["agentview_image"]
    if flip:
        img = img[::-1, ::-1]          # same 180-degree rotation OpenVLA applies for LIBERO
    return np.ascontiguousarray(img)


def save_gif(frames: list[np.ndarray], path: Path, every: int = 2) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE) for f in frames[::every]]
    if imgs:
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100 * every, loop=0)


# --------------------------------------------------------------------------- #
# Smoke test (no model)
# --------------------------------------------------------------------------- #

def smoke_test(args) -> None:
    benchmark, get_libero_path, OffScreenRenderEnv = import_libero()
    from PIL import Image
    suite_name = SUITE_TO_BENCHMARK.get(args.suites[0], args.suites[0])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(0)
    print(f"Suite {suite_name}: {suite.n_tasks} tasks. Task 0: {task.language!r}", flush=True)

    env = make_env(OffScreenRenderEnv, get_libero_path, task, args.seed)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(NUM_WAIT_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION)
    print("obs keys:", sorted(k for k in obs if k.endswith("_image")), flush=True)
    raw = np.ascontiguousarray(obs["agentview_image"])
    panels = [raw, env_image(obs, True)]
    labels = ["env raw", "env rotated 180 (what eval feeds the model)"]
    env.close()

    try:   # add a TRAINING frame for comparison
        from data_utils import SuiteSpec, load_suite_dataset
        ds = load_suite_dataset(SuiteSpec(
            name=args.suites[0],
            repo_id=dataset_repo_id(args.hf_namespace, args.suites[0]),
        ))
        import torch
        img = ds[0]["observation.images.image"]
        if torch.is_tensor(img):
            img = (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        panels.append(np.ascontiguousarray(img)); labels.append("training-data frame")
    except Exception as e:                                   # noqa: BLE001
        print(f"(training frame skipped: {type(e).__name__}: {e})", flush=True)

    out = Path(args.sanity_dir); out.mkdir(parents=True, exist_ok=True)
    h = min(p.shape[0] for p in panels)
    strip = np.concatenate([np.array(Image.fromarray(p).resize((h, h))) for p in panels], axis=1)
    Image.fromarray(strip).save(out / "orientation_check.png")
    print("Panels (left to right):", " | ".join(labels), flush=True)
    print(f"Saved {out / 'orientation_check.png'}", flush=True)
    print("SMOKE TEST OK", flush=True)


# --------------------------------------------------------------------------- #
# Model side
# --------------------------------------------------------------------------- #

def load_action_stats_registry(checkpoint: str, args) -> dict:
    """Returns {suite: (q01, q99)} for every suite with a saved entry in this
    checkpoint's action_stats.json. A suite missing from the registry (e.g.
    evaluating a checkpoint produced before action_stats.json existed, or
    checking an untrained suite) falls back to recomputing from that SAME
    suite's own data -- never from a different suite's stats, which is the
    bug this replaced: reusing one suite's normalization for every suite
    being evaluated silently corrupts de-normalization for every suite
    except the one the stats happened to come from."""
    p = Path(checkpoint) / "action_stats.json"
    registry = json.loads(p.read_text()) if p.is_file() else {}
    from data_utils import SuiteSpec, load_suite_dataset
    from train_continual import compute_action_bounds

    out = {}
    for suite in args.suites:
        if suite in registry:
            d = registry[suite]
            out[suite] = (np.asarray(d["q01"], dtype=np.float32), np.asarray(d["q99"], dtype=np.float32))
        else:
            print(f"WARNING: no action_stats.json entry for '{suite}' in {checkpoint} -> "
                  f"recomputing from {suite}'s own data. This is only correct if training "
                  f"used these exact statistics (true for a freshly patched run; NOT "
                  f"guaranteed for a checkpoint trained before this suite's stats were "
                  f"ever persisted).", flush=True)
            ds = load_suite_dataset(SuiteSpec(name=suite, repo_id=dataset_repo_id(args.hf_namespace, suite)))
            out[suite] = compute_action_bounds(ds)
    return out


def load_policy(checkpoint: str):
    # Reuse the training loader so the SDPA workaround and dtype handling match.
    from train_continual import load_model_and_processor
    model, processor = load_model_and_processor(checkpoint)
    model.eval()
    return model, processor


def set_suite_norm_stats(model, q01, q99) -> bool:
    """Must be called before evaluating EACH suite (not once for the whole
    checkpoint): different suites can have different action ranges and
    gripper encodings, so the normalization -- and the inferred gripper
    convention -- has to be switched per suite. Returns gripper_is_signed."""
    model.norm_stats[UNNORM_KEY] = {"action": {
        "q01": [float(x) for x in q01], "q99": [float(x) for x in q99],
        "mask": [True] * len(q01)}}
    return bool(q01[-1] < -0.5)


def predict_action(model, processor, image: np.ndarray, instruction: str, gripper_is_signed: bool):
    import torch
    from PIL import Image
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    return postprocess_action(action, gripper_is_signed)


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #

def evaluate_suite(model, processor, suite_name: str, args, gripper_is_signed: bool, stage: str) -> dict:
    benchmark, get_libero_path, OffScreenRenderEnv = import_libero()
    bname = SUITE_TO_BENCHMARK.get(suite_name, suite_name)
    suite = benchmark.get_benchmark_dict()[bname]()
    max_steps = args.max_steps or MAX_STEPS.get(bname, 300)
    task_ids = ([int(x) for x in args.task_ids.split(",")] if args.task_ids
                else list(range(suite.n_tasks)))
    flip = not args.no_flip

    per_task, n_ok, n_all, t0 = {}, 0, 0, time.time()
    for k, tid in enumerate(task_ids):
        task = suite.get_task(tid)
        init_states = suite.get_task_init_states(tid)
        env = make_env(OffScreenRenderEnv, get_libero_path, task, args.seed)
        ok_task = 0
        for ep in range(args.episodes_per_task):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(NUM_WAIT_STEPS):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            record = ep < args.save_videos
            frames, success, steps = [], False, 0
            for t in range(max_steps):
                img = env_image(obs, flip)
                if record:
                    frames.append(img)
                action = predict_action(model, processor, img, task.language, gripper_is_signed)
                obs, _, done, _ = env.step(action.tolist())
                steps = t + 1
                if done:
                    success = True
                    break
            n_all += 1; n_ok += int(success); ok_task += int(success)
            print(f"[{stage} | {suite_name}] task {k+1}/{len(task_ids)} ep {ep+1}/{args.episodes_per_task} "
                  f"-> {'SUCCESS' if success else 'fail   '} ({steps:3d} steps) | "
                  f"suite SR so far {n_ok/n_all:.2f} | {time.time()-t0:.0f}s", flush=True)
            if record:
                save_gif(frames, Path(args.video_dir) / stage / suite_name /
                         f"task{tid:02d}_ep{ep:02d}_{'success' if success else 'fail'}.gif")
        env.close()
        per_task[str(tid)] = {"language": task.language, "success": ok_task, "n": args.episodes_per_task}
        print(f"  task {tid}: {ok_task}/{args.episodes_per_task}  {task.language}", flush=True)

    return {"success_rate": n_ok / max(n_all, 1), "n_episodes": n_all, "per_task": per_task}


def main() -> None:
    args = parse_args()

    if args.show_results:
        d = json.loads(Path(args.show_results).read_text())
        order = d.get("suites_order") or sorted({s for st in d["success"].values() for s in st})
        print(format_table(d["success"], d.get("forgetting", {}), order))
        return
    if args.check_imports:
        import_libero(); print("LIBERO imports OK"); return
    if args.smoke_test:
        smoke_test(args); return
    if not args.checkpoint:
        sys.exit("--checkpoint is required")

    stage = args.stage_name or Path(args.checkpoint).name
    stats_registry = load_action_stats_registry(args.checkpoint, args)
    model, processor = load_policy(args.checkpoint)

    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(results_path.read_text()) if results_path.exists() else {}
    for key in ("success", "n_episodes", "per_task", "forgetting"):
        data.setdefault(key, {})
    order = data.setdefault("suites_order", [])
    data.setdefault("success", {}).setdefault(stage, {})
    data["n_episodes"].setdefault(stage, {}); data["per_task"].setdefault(stage, {})

    gripper_by_suite = {}
    for suite in args.suites:
        q01, q99 = stats_registry[suite]
        gripper_is_signed = set_suite_norm_stats(model, q01, q99)
        gripper_by_suite[suite] = gripper_is_signed
        print(f"[{suite}] gripper convention: "
              f"{'signed [-1,+1] (binarized)' if gripper_is_signed else '[0,1] OpenVLA-RLDS (rescaled + inverted)'}",
              flush=True)

        res = evaluate_suite(model, processor, suite, args, gripper_is_signed, stage)
        data["success"][stage][suite] = res["success_rate"]
        data["n_episodes"][stage][suite] = res["n_episodes"]
        data["per_task"][stage][suite] = res["per_task"]
        if suite not in order:
            order.append(suite)
        data["forgetting"] = compute_forgetting(data["success"])
        data["config"] = {"episodes_per_task": args.episodes_per_task, "seed": args.seed,
                          "flip_images": not args.no_flip, "gripper_signed": gripper_by_suite}
        results_path.write_text(json.dumps(data, indent=2))       # persist after every suite
        print(f"==> [{stage}] {suite}: success rate {res['success_rate']:.3f} "
              f"({res['n_episodes']} episodes)\n", flush=True)

    print(format_table(data["success"], data["forgetting"], order))


if __name__ == "__main__":
    main()
