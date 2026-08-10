"""Recording utilities: demo video plus a machine-readable trajectory dataset.

The recorder streams video frames straight to disk so that a long episode does
not have to be buffered in memory, and it collects a synchronised observation /
action table that can be used to train or evaluate a policy offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np

from .config import ARM_JOINTS, CUBE_BODIES, RecordConfig
from .task import SortingTask, TaskState

# Wrist images are only used for qualitative inspection, so a small, capped set
# keeps the dataset file comfortably under a few tens of megabytes.
MAX_WRIST_FRAMES = 240
WRIST_EVERY_N_ROWS = 5

STATE_IDS = {state: index for index, state in enumerate(TaskState)}


class EpisodeRecorder:
    """Capture an episode as an MP4 video and an NPZ trajectory dataset."""

    def __init__(self, model, data, config: Optional[RecordConfig] = None) -> None:
        self.model = model
        self.data = data
        self.cfg = config or RecordConfig()

        self._scene_renderer = None
        self._wrist_renderer = None
        self._writer = None
        self._video_path: Optional[Path] = None

        self._next_frame_time = 0.0
        self._rows = 0
        self.wrist_frames: List[np.ndarray] = []
        self.columns: Dict[str, List] = {
            "time": [],
            "state": [],
            "arm_qpos": [],
            "arm_qvel": [],
            "arm_action": [],
            "gripper_travel": [],
            "gripper_action": [],
            "ee_pos": [],
            "ee_quat": [],
            "touch": [],
            "cube_pos": [],
        }

        self._arm_qpos_ids = np.asarray(
            [
                model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in ARM_JOINTS
            ],
            dtype=int,
        )
        self._arm_dof_ids = np.asarray(
            [
                model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in ARM_JOINTS
            ],
            dtype=int,
        )
        self._cube_body_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in CUBE_BODIES],
            dtype=int,
        )
        self._ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # ------------------------------------------------------------------ #

    def start(self, run_dir: Path) -> None:
        """Open the renderers and the video writer."""
        run_dir.mkdir(parents=True, exist_ok=True)
        if not self.cfg.enabled:
            return

        import imageio.v2 as imageio

        self._scene_renderer = mujoco.Renderer(
            self.model, height=self.cfg.height, width=self.cfg.width
        )
        if self.cfg.record_wrist_frames:
            self._wrist_renderer = mujoco.Renderer(
                self.model, height=self.cfg.wrist_height, width=self.cfg.wrist_width
            )

        self._video_path = run_dir / "demo.mp4"
        self._writer = imageio.get_writer(
            self._video_path,
            fps=self.cfg.fps,
            macro_block_size=None,
            quality=8,
        )
        self._next_frame_time = 0.0

    def capture(self, time_s: float, task: SortingTask) -> None:
        """Record one control step. Frames are throttled to the target fps."""
        self._record_row(time_s, task)

        if self._writer is None or self._scene_renderer is None:
            return
        if time_s + 1e-9 < self._next_frame_time:
            return

        self._scene_renderer.update_scene(self.data, camera=self.cfg.camera)
        self._writer.append_data(self._scene_renderer.render())
        self._next_frame_time += 1.0 / float(self.cfg.fps)

    def _record_row(self, time_s: float, task: SortingTask) -> None:
        if self._rows % max(self.cfg.dataset_stride, 1) != 0:
            self._rows += 1
            return

        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, self.data.site_xmat[self._ee_site_id])

        self.columns["time"].append(float(time_s))
        self.columns["state"].append(STATE_IDS[task.state])
        self.columns["arm_qpos"].append(np.array(self.data.qpos[self._arm_qpos_ids]))
        self.columns["arm_qvel"].append(np.array(self.data.qvel[self._arm_dof_ids]))
        self.columns["arm_action"].append(task.arm.command)
        self.columns["gripper_travel"].append(task.gripper.travel)
        self.columns["gripper_action"].append(float(self.data.ctrl[task.gripper.actuator_id]))
        self.columns["ee_pos"].append(np.array(self.data.site_xpos[self._ee_site_id]))
        self.columns["ee_quat"].append(ee_quat)
        self.columns["touch"].append(np.array(task.gripper.touch_forces()))
        self.columns["cube_pos"].append(np.array(self.data.xpos[self._cube_body_ids]))

        if (
            self._wrist_renderer is not None
            and len(self.wrist_frames) < MAX_WRIST_FRAMES
            and (len(self.columns["time"]) % WRIST_EVERY_N_ROWS) == 0
        ):
            self._wrist_renderer.update_scene(self.data, camera=self.cfg.wrist_camera)
            self.wrist_frames.append(self._wrist_renderer.render())

        self._rows += 1

    # ------------------------------------------------------------------ #

    def finish(self, run_dir: Path, metadata: Dict[str, object]) -> Dict[str, str]:
        """Close the video, write the dataset and the run report."""
        artifacts: Dict[str, str] = {}

        if self._writer is not None:
            self._writer.close()
            self._writer = None
            if self._video_path is not None:
                artifacts["video"] = str(self._video_path)
        for renderer in (self._scene_renderer, self._wrist_renderer):
            if renderer is not None:
                renderer.close()
        self._scene_renderer = None
        self._wrist_renderer = None

        dataset_path = run_dir / "trajectory.npz"
        arrays = {key: np.asarray(value) for key, value in self.columns.items() if value}
        if self.wrist_frames:
            arrays["wrist_rgb"] = np.asarray(self.wrist_frames, dtype=np.uint8)
        np.savez_compressed(dataset_path, **arrays)
        artifacts["dataset"] = str(dataset_path)

        report_path = run_dir / "report.json"
        report = dict(metadata)
        report["dataset_rows"] = int(len(self.columns["time"]))
        report["state_ids"] = {state.value: index for state, index in STATE_IDS.items()}
        report["columns"] = sorted(arrays.keys())
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        artifacts["report"] = str(report_path)

        return artifacts
