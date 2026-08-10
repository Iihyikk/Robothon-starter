"""Low-level controllers for the sorting cell.

Two small controllers sit between the task state machine and MuJoCo:

* ArmController      - rate limits joint targets so the arm moves smoothly and
                       the position actuators never receive a step input.
* GripperController  - drives the single gripper actuator and reports whether
                       an object is actually held, using the jaw touch sensors.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import mujoco
import numpy as np

from .config import GripperConfig


def _sensor_address(model, name: str) -> int:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"unknown sensor {name!r}")
    return int(model.sensor_adr[sensor_id])


class ArmController:
    """Track a joint-space target with a bounded joint speed."""

    def __init__(
        self,
        model,
        data,
        joint_names: Sequence[str],
        actuator_names: Sequence[str],
        max_joint_speed: float = 1.7,
    ) -> None:
        self.model = model
        self.data = data
        self.max_joint_speed = float(max_joint_speed)

        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names],
            dtype=int,
        )
        if np.any(self.joint_ids < 0):
            raise ValueError("one or more arm joints were not found in the model")
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]

        self.actuator_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names],
            dtype=int,
        )
        if np.any(self.actuator_ids < 0):
            raise ValueError("one or more arm actuators were not found in the model")

        self._command = np.array(data.qpos[self.qpos_ids], dtype=float)
        self._target = self._command.copy()

    # ------------------------------------------------------------------ #

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    def measured(self) -> np.ndarray:
        return np.array(self.data.qpos[self.qpos_ids], dtype=float)

    def reset(self, q: Sequence[float]) -> None:
        self._command = np.asarray(q, dtype=float).copy()
        self._target = self._command.copy()
        self.data.ctrl[self.actuator_ids] = self._command

    def set_target(self, q_target: Sequence[float]) -> None:
        self._target = np.asarray(q_target, dtype=float).copy()

    def step(self, dt: float) -> np.ndarray:
        """Advance the commanded joint vector by at most max_joint_speed * dt."""
        limit = self.max_joint_speed * float(dt)
        delta = np.clip(self._target - self._command, -limit, limit)
        self._command = self._command + delta
        self.data.ctrl[self.actuator_ids] = self._command
        return self._command.copy()

    def command_reached(self, tolerance: float = 1e-3) -> bool:
        return bool(np.max(np.abs(self._target - self._command)) < tolerance)

    def settled(self, tolerance: float = 0.015) -> bool:
        """True when the measured joints have caught up with the target."""
        return self.command_reached() and bool(
            np.max(np.abs(self._target - self.measured())) < tolerance
        )


class GripperController:
    """Parallel-jaw gripper driven by a single position actuator.

    The two jaws are coupled by an equality constraint in the MJCF, so a single
    control signal (the travel of the left jaw) opens and closes both fingers.
    """

    def __init__(self, model, data, config: GripperConfig | None = None) -> None:
        self.model = model
        self.data = data
        self.cfg = config or GripperConfig()

        self.actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_jaw_joint")
        if self.actuator_id < 0 or joint_id < 0:
            raise ValueError("gripper actuator or joint missing from the model")
        self.qpos_id = int(model.jnt_qposadr[joint_id])

        self._left_touch = _sensor_address(model, "left_pad_touch")
        self._right_touch = _sensor_address(model, "right_pad_touch")

        self._goal = self.cfg.open_travel
        self._command = float(data.qpos[self.qpos_id])

    # ------------------------------------------------------------------ #

    @property
    def travel(self) -> float:
        """Current single-jaw travel in metres (0 = fully closed)."""
        return float(self.data.qpos[self.qpos_id])

    @property
    def opening(self) -> float:
        """Distance between the two jaw pads in metres."""
        return 2.0 * (0.007 + self.travel)

    def touch_forces(self) -> Tuple[float, float]:
        return (
            float(self.data.sensordata[self._left_touch]),
            float(self.data.sensordata[self._right_touch]),
        )

    def reset(self, opened: bool = True) -> None:
        self._goal = self.cfg.open_travel if opened else self.cfg.closed_travel
        self._command = self._goal
        self.data.ctrl[self.actuator_id] = self._command

    def open(self) -> None:
        self._goal = self.cfg.open_travel

    def close(self) -> None:
        self._goal = self.cfg.closed_travel

    @property
    def closing(self) -> bool:
        return self._goal <= self.cfg.closed_travel + 1e-9

    def step(self, dt: float) -> float:
        seconds = self.cfg.close_seconds if self.closing else self.cfg.open_seconds
        span = max(self.cfg.open_travel - self.cfg.closed_travel, 1e-6)
        limit = span / max(seconds, 1e-3) * float(dt)
        delta = np.clip(self._goal - self._command, -limit, limit)
        self._command = float(self._command + delta)
        self.data.ctrl[self.actuator_id] = self._command
        return self._command

    def command_reached(self, tolerance: float = 1e-4) -> bool:
        return abs(self._goal - self._command) < tolerance

    def holding_object(self) -> bool:
        """Both pads loaded and the jaws stopped well before full closure."""
        left, right = self.touch_forces()
        threshold = self.cfg.contact_force_threshold
        squeezing = left > threshold and right > threshold
        return bool(squeezing and self.travel > self.cfg.min_grasp_travel)
