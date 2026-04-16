#!/usr/bin/env python3
"""
LIBERO benchmark for the pi0 / pi0-FAST / pi0.5 models on NVIDIA CUDA.

Two modes
---------
inference-only  (default, no LIBERO install required)
    Feeds synthetic observations through the policy and measures latency /
    throughput on NVIDIA GPUs.  Mirrors exactly the observation format
    used by examples/libero/main.py so numbers are representative.

simulation      (--sim, requires LIBERO + MuJoCo)
    Runs full episodes in the LIBERO MuJoCo sim and reports task success rate
    alongside per-inference latency.  Requires the openpi server to be running
    in a second terminal:
        python scripts/serve_policy.py --env LIBERO

Usage
-----
# Inference benchmark (no sim needed)
python benchmark_pi0_libero_cuda.py --config pi0_libero --quick
python benchmark_pi0_libero_cuda.py --config pi05_libero --num-warmup 5 --num-runs 50

# Simulation benchmark (LIBERO + MuJoCo installed)
python benchmark_pi0_libero_cuda.py --sim --task-suite libero_spatial --num-trials 10

# Save results to JSON
python benchmark_pi0_libero_cuda.py --config pi0_libero --num-runs 30 --output-json results/libero_cuda.json
"""

import argparse
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LIBERO constants (from examples/libero/main.py)
# ---------------------------------------------------------------------------

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256   # resolution used to render training data
LIBERO_RESIZE = 224            # resize target sent to policy

# Maximum episode steps per task suite (covers the longest training demo + margin)
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}

# LIBERO state breakdown: robot0_eef_pos(3) + axisangle(3) + gripper_qpos(2) = 8
LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7   # policy outputs are truncated to first 7 dims


# ---------------------------------------------------------------------------
# CUDA / device helpers
# ---------------------------------------------------------------------------

def detect_cuda_devices() -> list[str]:
    """Detect NVIDIA GPUs via nvidia-smi (framework-independent)."""
    devices: list[str] = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split('\n'):
                if line:
                    idx, name = line.split(',', 1)
                    devices.append(f"nvidia-smi:gpu{idx.strip()} ({name.strip()})")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"torch:cuda:{i} ({torch.cuda.get_device_name(i)})")
    except Exception:
        pass
    return devices


def get_gpu_memory_mb() -> dict[str, float]:
    stats: dict[str, float] = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    idx = parts[0].strip()
                    used_mb = float(parts[1].strip())
                    total_mb = float(parts[2].strip())
                    stats[f"gpu{idx}_used_mb"] = used_mb
                    stats[f"gpu{idx}_total_mb"] = total_mb
    except Exception:
        pass
    return stats


# ---------------------------------------------------------------------------
# Observation / state helpers  (match examples/libero/main.py exactly)
# ---------------------------------------------------------------------------

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle (from robosuite via libero/main.py)."""
    q = quat.copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - q[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def make_libero_obs(resize: int = LIBERO_RESIZE) -> dict[str, Any]:
    """
    Synthetic observation matching the element dict in examples/libero/main.py.

    Keys and shapes:
        observation/image       : (resize, resize, 3)  uint8   agentview (rotated 180°)
        observation/wrist_image : (resize, resize, 3)  uint8   wrist cam  (rotated 180°)
        observation/state       : (8,)                 float32
            [eef_pos(3), axisangle(3), gripper_qpos(2)]
        prompt                  : str
    """
    img       = np.random.randint(0, 256, (resize, resize, 3), dtype=np.uint8)
    wrist_img = np.random.randint(0, 256, (resize, resize, 3), dtype=np.uint8)

    eef_pos      = np.random.uniform(-0.5, 0.5, 3).astype(np.float32)
    quat         = np.random.uniform(-1.0, 1.0, 4).astype(np.float64)
    quat        /= np.linalg.norm(quat) + 1e-8
    axisangle    = _quat2axisangle(quat).astype(np.float32)
    gripper_qpos = np.random.uniform(0.0, 0.05, 2).astype(np.float32)
    state        = np.concatenate([eef_pos, axisangle, gripper_qpos])   # (8,)

    return {
        "observation/image":       img,
        "observation/wrist_image": wrist_img,
        "observation/state":       state,
        "prompt":                  "pick up the bowl and place it on the plate",
    }


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Results from the inference-only benchmark."""
    config_name:   str
    backend:       str
    num_warmup:    int
    num_runs:      int
    latencies_ms:  list[float] = field(default_factory=list)
    action_shapes: list[tuple] = field(default_factory=list)
    errors:        list[str]   = field(default_factory=list)
    gpu_mem_before: dict[str, float] = field(default_factory=dict)
    gpu_mem_after:  dict[str, float] = field(default_factory=dict)

    # derived
    mean_ms:   float = 0.0
    median_ms: float = 0.0
    std_ms:    float = 0.0
    min_ms:    float = 0.0
    max_ms:    float = 0.0
    p90_ms:    float = 0.0
    p99_ms:    float = 0.0
    fps:       float = 0.0

    def summarize(self) -> None:
        if not self.latencies_ms:
            return
        lats = sorted(self.latencies_ms)
        n = len(lats)
        self.mean_ms   = statistics.mean(lats)
        self.median_ms = statistics.median(lats)
        self.std_ms    = statistics.stdev(lats) if n > 1 else 0.0
        self.min_ms    = lats[0]
        self.max_ms    = lats[-1]
        self.p90_ms    = lats[int(0.90 * n)]
        self.p99_ms    = lats[int(0.99 * n)]
        self.fps       = 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def print_report(self) -> None:
        self.summarize()
        bar = "=" * 62
        print(f"\n{bar}")
        print(f"  LIBERO Inference Benchmark  [{self.config_name}]  [{self.backend.upper()}]")
        print(bar)
        print(f"  Warmup : {self.num_warmup}   Timed runs : {self.num_runs}")
        if self.latencies_ms:
            print(f"\n  Latency (ms)")
            print(f"    mean    : {self.mean_ms:8.2f}")
            print(f"    median  : {self.median_ms:8.2f}")
            print(f"    std     : {self.std_ms:8.2f}")
            print(f"    min     : {self.min_ms:8.2f}")
            print(f"    max     : {self.max_ms:8.2f}")
            print(f"    p90     : {self.p90_ms:8.2f}")
            print(f"    p99     : {self.p99_ms:8.2f}")
            print(f"\n  Throughput : {self.fps:.3f} inferences/sec")
        if self.action_shapes:
            print(f"  Action shape : {self.action_shapes[0]}")
        if self.gpu_mem_before or self.gpu_mem_after:
            print(f"\n  GPU VRAM (MB)")
            for k in self.gpu_mem_before:
                before = self.gpu_mem_before.get(k, 0)
                after  = self.gpu_mem_after.get(k, 0)
                print(f"    {k}: before={before:.0f}  after={after:.0f}  delta={after-before:.0f}")
        if self.errors:
            print(f"\n  Errors: {len(self.errors)}")
            for e in self.errors[:3]:
                print(f"    {e}")
        print(bar)

    def to_dict(self) -> dict:
        self.summarize()
        return {
            "mode": "inference",
            "config_name":   self.config_name,
            "backend":       self.backend,
            "num_warmup":    self.num_warmup,
            "num_runs":      self.num_runs,
            "latency_ms": {
                "mean": self.mean_ms, "median": self.median_ms,
                "std":  self.std_ms,  "min":    self.min_ms,
                "max":  self.max_ms,  "p90":    self.p90_ms,
                "p99":  self.p99_ms,  "all":    self.latencies_ms,
            },
            "throughput_fps":   self.fps,
            "action_shape":     list(self.action_shapes[0]) if self.action_shapes else [],
            "gpu_mem_before_mb": self.gpu_mem_before,
            "gpu_mem_after_mb":  self.gpu_mem_after,
            "errors": self.errors,
        }


@dataclass
class TaskResult:
    task_id:      int
    description:  str
    episodes:     int = 0
    successes:    int = 0
    infer_ms_all: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def mean_infer_ms(self) -> float:
        return statistics.mean(self.infer_ms_all) if self.infer_ms_all else 0.0


@dataclass
class SimResult:
    """Results from the full simulation benchmark."""
    task_suite:     str
    num_trials:     int
    replan_steps:   int
    task_results:   list[TaskResult] = field(default_factory=list)
    errors:         list[str]        = field(default_factory=list)

    @property
    def total_episodes(self) -> int:
        return sum(t.episodes for t in self.task_results)

    @property
    def total_successes(self) -> int:
        return sum(t.successes for t in self.task_results)

    @property
    def overall_success_rate(self) -> float:
        return self.total_successes / self.total_episodes if self.total_episodes else 0.0

    @property
    def all_infer_ms(self) -> list[float]:
        ms = []
        for t in self.task_results:
            ms.extend(t.infer_ms_all)
        return ms

    def print_report(self) -> None:
        bar = "=" * 62
        print(f"\n{bar}")
        print(f"  LIBERO Simulation Benchmark  [{self.task_suite}]")
        print(bar)
        print(f"  Trials/task : {self.num_trials}   Replan every : {self.replan_steps} steps")
        print(f"\n  Per-task results")
        for t in self.task_results:
            print(f"    Task {t.task_id:2d}  {t.success_rate*100:5.1f}%  ({t.successes}/{t.episodes})"
                  f"  avg_infer={t.mean_infer_ms:.1f}ms  '{t.description[:50]}'")
        lats = self.all_infer_ms
        if lats:
            lats_s = sorted(lats)
            n = len(lats_s)
            print(f"\n  Inference latency (ms)  [n={n} calls]")
            print(f"    mean : {statistics.mean(lats_s):.2f}")
            print(f"    p90  : {lats_s[int(0.90*n)]:.2f}")
            print(f"    max  : {lats_s[-1]:.2f}")
        print(f"\n  Overall success rate : {self.overall_success_rate*100:.1f}%"
              f"  ({self.total_successes}/{self.total_episodes})")
        if self.errors:
            print(f"  Errors: {len(self.errors)}")
        print(bar)

    def to_dict(self) -> dict:
        lats = sorted(self.all_infer_ms)
        n = len(lats)
        return {
            "mode":         "simulation",
            "task_suite":   self.task_suite,
            "num_trials":   self.num_trials,
            "replan_steps": self.replan_steps,
            "total_episodes":    self.total_episodes,
            "total_successes":   self.total_successes,
            "overall_success_rate": self.overall_success_rate,
            "latency_ms": {
                "mean": statistics.mean(lats) if lats else 0,
                "p90":  lats[int(0.90*n)] if lats else 0,
                "max":  lats[-1] if lats else 0,
            },
            "tasks": [
                {
                    "id": t.task_id,
                    "description": t.description,
                    "episodes": t.episodes,
                    "successes": t.successes,
                    "success_rate": t.success_rate,
                    "mean_infer_ms": t.mean_infer_ms,
                }
                for t in self.task_results
            ],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Inference-only benchmark
# ---------------------------------------------------------------------------

def _configure_jax_cuda() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    try:
        import jax
        jax.config.update("jax_default_matmul_precision", "float32")
        log.info(f"JAX devices: {jax.devices()}")
    except Exception as e:
        log.warning(f"JAX device check failed: {e}")


def run_inference_benchmark(
    config_name: str,
    backend: str,
    checkpoint_dir: str | None,
    num_warmup: int,
    num_runs: int,
) -> InferenceResult:
    result = InferenceResult(
        config_name=config_name,
        backend=backend,
        num_warmup=num_warmup,
        num_runs=num_runs,
    )

    if backend == "jax":
        _configure_jax_cuda()

    try:
        from openpi.training import config as _config
        from openpi.policies import policy_config
        from openpi.shared import download
    except ImportError as e:
        print(f"[ERROR] openpi not found: {e}")
        sys.exit(1)

    # --- load config ---
    log.info(f"Loading config '{config_name}' ...")
    cfg = _config.get_config(config_name)

    # --- resolve checkpoint ---
    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir)
        log.info(f"Checkpoint: {ckpt_path}")
    else:
        gs_path = f"gs://openpi-assets/checkpoints/{config_name}"
        log.info(f"Downloading checkpoint from {gs_path} ...")
        try:
            ckpt_path = download.maybe_download(gs_path)
        except Exception as e:
            print(f"[ERROR] Checkpoint download failed: {e}")
            print("  Use --checkpoint-dir to point at a local path.")
            sys.exit(1)

    # Warn if safetensors absent for pytorch backend
    if backend == "pytorch" and not (ckpt_path / "model.safetensors").exists():
        log.warning(
            "model.safetensors not found. Convert the JAX checkpoint first:\n"
            "  python examples/convert_jax_model_to_pytorch.py "
            f"--checkpoint_dir {ckpt_path} --config_name {config_name} "
            f"--output_path {ckpt_path}_pytorch"
        )

    # --- create policy ---
    log.info("Creating policy (first run triggers JIT compilation) ...")
    t0 = time.perf_counter()
    try:
        policy = policy_config.create_trained_policy(cfg, str(ckpt_path))
    except Exception as e:
        print(f"[ERROR] Policy creation failed: {e}")
        raise
    log.info(f"Policy loaded in {time.perf_counter()-t0:.1f}s")

    result.gpu_mem_before = get_gpu_memory_mb()

    # --- warmup ---
    log.info(f"Warmup ({num_warmup} passes) ...")
    for i in range(num_warmup):
        obs = make_libero_obs()
        try:
            out = policy.infer(obs)
            ms = out.get("policy_timing", {}).get("infer_ms", -1)
            print(f"  warmup {i+1}/{num_warmup}  internal={ms:.1f}ms")
        except Exception as e:
            print(f"  warmup {i+1} error: {e}")

    # --- timed runs ---
    log.info(f"Timed benchmark ({num_runs} passes) ...")
    for i in range(num_runs):
        obs = make_libero_obs()
        t_start = time.perf_counter()
        try:
            out = policy.infer(obs)
        except Exception as e:
            result.errors.append(f"run {i}: {e}")
            continue
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        internal_ms = out.get("policy_timing", {}).get("infer_ms", wall_ms)
        result.latencies_ms.append(wall_ms)
        actions = out.get("actions", np.array([]))
        if hasattr(actions, "shape"):
            result.action_shapes.append(actions.shape)
        if (i + 1) % max(1, num_runs // 5) == 0 or i == num_runs - 1:
            print(f"  run {i+1:3d}/{num_runs}  wall={wall_ms:7.1f}ms  internal={internal_ms:7.1f}ms")

    result.gpu_mem_after = get_gpu_memory_mb()
    return result


# ---------------------------------------------------------------------------
# Simulation benchmark
# ---------------------------------------------------------------------------

def run_sim_benchmark(
    task_suite_name: str,
    host: str,
    port: int,
    num_trials: int,
    replan_steps: int,
    seed: int,
    video_dir: str | None,
) -> SimResult:
    result = SimResult(
        task_suite=task_suite_name,
        num_trials=num_trials,
        replan_steps=replan_steps,
    )

    # --- import LIBERO (optional dependency) ---
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        print(
            "[ERROR] LIBERO not installed.\n"
            "  Install it via:\n"
            "    git submodule update --init --recursive\n"
            "    pip install -e third_party/libero"
        )
        sys.exit(1)

    try:
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy as _wcp
    except ImportError:
        print(
            "[ERROR] openpi_client not installed.\n"
            "  pip install -e packages/openpi-client"
        )
        sys.exit(1)

    np.random.seed(seed)
    max_steps = SUITE_MAX_STEPS.get(task_suite_name, 400)
    num_steps_wait = 10   # wait for objects to settle in sim

    log.info(f"Connecting to policy server at {host}:{port} ...")
    client = _wcp.WebsocketClientPolicy(host, port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    log.info(f"Task suite '{task_suite_name}': {task_suite.n_tasks} tasks, "
             f"{num_trials} trials each, max {max_steps} steps/episode")

    if video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    import collections
    try:
        import imageio
    except ImportError:
        imageio = None

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        task_desc = task.language
        initial_states = task_suite.get_task_init_states(task_id)
        task_bddl = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env_args = {
            "bddl_file_name": str(task_bddl),
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths":  LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        task_result = TaskResult(task_id=task_id, description=task_desc)

        for episode_idx in range(min(num_trials, len(initial_states))):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            action_plan: collections.deque = collections.deque()
            replay_frames = []
            done = False

            for t in range(max_steps + num_steps_wait):
                try:
                    if t < num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        continue

                    # Preprocess images — rotate 180° to match training data
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    from openpi_client.image_tools import convert_to_uint8, resize_with_pad
                    img   = convert_to_uint8(resize_with_pad(img,   LIBERO_RESIZE, LIBERO_RESIZE))
                    wrist = convert_to_uint8(resize_with_pad(wrist, LIBERO_RESIZE, LIBERO_RESIZE))
                    replay_frames.append(img)

                    if not action_plan:
                        state = np.concatenate([
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ])
                        element = {
                            "observation/image":       img,
                            "observation/wrist_image": wrist,
                            "observation/state":       state,
                            "prompt":                  str(task_desc),
                        }
                        t_infer = time.perf_counter()
                        chunk = client.infer(element)["actions"]
                        infer_ms = (time.perf_counter() - t_infer) * 1000.0
                        task_result.infer_ms_all.append(infer_ms)
                        action_plan.extend(chunk[:replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_result.successes += 1
                        break
                except Exception as e:
                    result.errors.append(f"task {task_id} ep {episode_idx}: {e}")
                    log.error(f"Episode error: {e}")
                    break

            task_result.episodes += 1

            if video_dir and imageio is not None and replay_frames:
                suffix = "success" if done else "failure"
                vpath = Path(video_dir) / f"task{task_id:02d}_ep{episode_idx:02d}_{suffix}.mp4"
                imageio.mimwrite(str(vpath), replay_frames, fps=10)

            log.info(
                f"  Task {task_id} ep {episode_idx+1}/{num_trials} "
                f"{'SUCCESS' if done else 'FAIL'}  "
                f"sr={task_result.success_rate*100:.0f}%"
            )

        result.task_results.append(task_result)
        log.info(
            f"Task {task_id} done: {task_result.success_rate*100:.1f}%  "
            f"({task_result.successes}/{task_result.episodes})"
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

LIBERO_CONFIGS = [
    "pi0_libero",
    "pi0_fast_libero",
    "pi05_libero",
    "pi0_libero_low_mem_finetune",
    "pi0_fast_libero_low_mem_finetune",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LIBERO benchmark for pi0 on NVIDIA CUDA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # mode
    p.add_argument("--sim", action="store_true",
                   help="Run full MuJoCo simulation benchmark (requires LIBERO install)")

    # inference-only options
    inf = p.add_argument_group("Inference benchmark options")
    inf.add_argument("--config",        default="pi0_libero", choices=LIBERO_CONFIGS)
    inf.add_argument("--backend",       default="pytorch", choices=["pytorch", "jax"])
    inf.add_argument("--checkpoint-dir", default=None,
                     help="Local checkpoint path (skips GCS download)")
    inf.add_argument("--num-warmup",    type=int, default=3)
    inf.add_argument("--num-runs",      type=int, default=20)
    inf.add_argument("--quick",         action="store_true",
                     help="1 warmup + 3 runs smoke test")

    # simulation options
    sim = p.add_argument_group("Simulation benchmark options")
    sim.add_argument("--task-suite",    default="libero_spatial",
                     choices=list(SUITE_MAX_STEPS.keys()))
    sim.add_argument("--num-trials",    type=int, default=50,
                     help="Episodes per task")
    sim.add_argument("--replan-steps",  type=int, default=5,
                     help="Execute N actions before querying policy again")
    sim.add_argument("--host",          default="0.0.0.0")
    sim.add_argument("--port",          type=int, default=8000)
    sim.add_argument("--seed",          type=int, default=7)
    sim.add_argument("--video-dir",     default=None,
                     help="Save episode rollout videos here")

    # output
    p.add_argument("--output-json",     default=None,
                   help="Write results to this JSON file")
    p.add_argument("--list-devices",    action="store_true",
                   help="Print detected NVIDIA GPUs and exit")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        devs = detect_cuda_devices()
        if devs:
            print("Detected NVIDIA/CUDA devices:")
            for d in devs:
                print(f"  {d}")
        else:
            print("No NVIDIA/CUDA devices detected.")
        sys.exit(0)

    # Print device inventory
    devs = detect_cuda_devices()
    if devs:
        print("[+] NVIDIA/CUDA devices:")
        for d in devs:
            print(f"    {d}")
    else:
        print("[WARN] No NVIDIA/CUDA devices detected — will run on CPU.")

    if args.quick:
        args.num_warmup = 1
        args.num_runs   = 3

    # --- run selected mode ---
    if args.sim:
        result = run_sim_benchmark(
            task_suite_name=args.task_suite,
            host=args.host,
            port=args.port,
            num_trials=args.num_trials,
            replan_steps=args.replan_steps,
            seed=args.seed,
            video_dir=args.video_dir,
        )
        result.print_report()
        data = result.to_dict()
    else:
        result = run_inference_benchmark(
            config_name=args.config,
            backend=args.backend,
            checkpoint_dir=args.checkpoint_dir,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )
        result.print_report()
        data = result.to_dict()

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[+] Results saved to {out}")


if __name__ == "__main__":
    main()
