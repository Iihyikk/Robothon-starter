#!/usr/bin/env python3
"""Run the Long-Horizon Sorting Cell demo.

Examples
--------
    python run_demo.py --check-scene          # compile the model and print stats
    python run_demo.py                        # headless run, writes outputs/<run>/demo.mp4
    python run_demo.py --viewer --no-video    # interactive MuJoCo viewer
    python run_demo.py --episodes 3 --seed 7  # short benchmark over three layouts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "MuJoCo is not installed. Install the demo dependencies with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sortbot.config import (  # noqa: E402  (path bootstrap must run first)
    DEFAULT_OUTPUT_DIR,
    SCENE_PATH,
    GripperConfig,
    IKConfig,
    RecordConfig,
    SimConfig,
    TaskConfig,
)
from sortbot.recorder import EpisodeRecorder  # noqa: E402
from sortbot.task import SortingTask  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-Horizon Sorting Cell demo")
    parser.add_argument("--scene", type=Path, default=SCENE_PATH, help="path to the MJCF scene")
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the cube layout")
    parser.add_argument("--max-seconds", type=float, default=300.0, help="episode time budget")
    parser.add_argument("--camera", type=str, default="overview", help="camera used for the video")
    parser.add_argument("--fps", type=int, default=30, help="video frame rate")
    parser.add_argument("--width", type=int, default=1280, help="video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="video height in pixels")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="output directory")
    parser.add_argument("--no-video", action="store_true", help="skip rendering, dataset only")
    parser.add_argument("--no-randomize", action="store_true", help="use the fixed cube layout")
    parser.add_argument("--viewer", action="store_true", help="open the interactive viewer")
    parser.add_argument(
        "--check-scene",
        action="store_true",
        help="compile the model, print its statistics and exit",
    )
    return parser


def describe_model(model) -> str:
    lines = [
        "Model compiled successfully.",
        f"  bodies      : {model.nbody}",
        f"  geoms       : {model.ngeom}",
        f"  joints      : {model.njnt}",
        f"  DoF         : {model.nv}",
        f"  actuators   : {model.nu}",
        f"  sensors     : {model.nsensor}",
        f"  cameras     : {model.ncam}",
        f"  timestep    : {model.opt.timestep}",
    ]
    return "\n".join(lines)


def run_episode(
    model,
    data,
    task: SortingTask,
    rng: np.random.Generator,
    recorder: EpisodeRecorder | None,
    run_dir: Path,
    max_seconds: float,
    verbose: bool = True,
) -> dict:
    control_dt = 1.0 / task.sim_cfg.control_hz
    steps_per_control = max(1, int(round(control_dt / model.opt.timestep)))

    task.reset(rng)
    if recorder is not None:
        recorder.start(run_dir)

    last_state = None
    wall_start = time.time()

    while not task.finished and data.time < max_seconds:
        task.update(control_dt)
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)
        if recorder is not None:
            recorder.capture(data.time, task)

        if verbose and task.state is not last_state:
            last_state = task.state
            print(
                f"  [{data.time:7.2f}s] {task.state.value:<9}"
                f" cube={task.current or '-':<14} sorted={task.sorted_count}/6"
            )

    summary = task.evaluate()
    summary["wall_clock_seconds"] = round(time.time() - wall_start, 2)
    summary["events"] = [event.__dict__ for event in task.events]
    return summary


def run_viewer(model, data, task: SortingTask, rng: np.random.Generator, max_seconds: float) -> dict:
    import mujoco.viewer

    control_dt = 1.0 / task.sim_cfg.control_hz
    steps_per_control = max(1, int(round(control_dt / model.opt.timestep)))
    task.reset(rng)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not task.finished and data.time < max_seconds:
            cycle_start = time.time()
            task.update(control_dt)
            for _ in range(steps_per_control):
                mujoco.mj_step(model, data)
            viewer.sync()
            sleep_for = control_dt - (time.time() - cycle_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    return task.evaluate()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    scene_path = args.scene.resolve()
    if not scene_path.exists():
        raise SystemExit(f"scene not found: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    if args.check_scene:
        print(describe_model(model))
        return 0

    sim_cfg = SimConfig(
        scene_path=scene_path,
        seed=args.seed,
        max_episode_seconds=args.max_seconds,
        randomize_layout=not args.no_randomize,
    )
    task = SortingTask(
        model,
        data,
        sim_config=sim_cfg,
        task_config=TaskConfig(),
        ik_config=IKConfig(),
        gripper_config=GripperConfig(),
    )
    rng = np.random.default_rng(args.seed)

    if args.viewer:
        summary = run_viewer(model, data, task, rng, args.max_seconds)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = args.output.resolve()
    summaries = []

    for episode in range(args.episodes):
        run_dir = root / f"{stamp}-episode{episode + 1:02d}"
        print(f"Episode {episode + 1}/{args.episodes} -> {run_dir}")

        record_cfg = replace(
            RecordConfig(),
            enabled=not args.no_video,
            camera=args.camera,
            fps=args.fps,
            width=args.width,
            height=args.height,
            output_dir=root,
        )
        recorder = EpisodeRecorder(model, data, record_cfg)

        summary = run_episode(
            model, data, task, rng, recorder, run_dir, args.max_seconds
        )
        artifacts = recorder.finish(
            run_dir,
            {
                "episode": episode + 1,
                "seed": args.seed,
                "scene": str(scene_path),
                "summary": summary,
            },
        )
        summary["artifacts"] = artifacts
        summaries.append(summary)

        print(
            f"  done: {summary['cubes_in_correct_bin']}/{summary['cubes_total']} cubes sorted"
            f" in {summary['episode_seconds']}s of simulated time"
        )
        for key, value in artifacts.items():
            print(f"    {key}: {value}")

    total = sum(s["cubes_in_correct_bin"] for s in summaries)
    possible = sum(s["cubes_total"] for s in summaries)
    print(f"\nOverall: {total}/{possible} cubes placed in the correct bin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
