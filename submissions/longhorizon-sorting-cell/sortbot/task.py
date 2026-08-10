"""The long-horizon sorting task, expressed as a finite state machine.

One episode asks the arm to clear six cubes from the workbench into the bin
that matches their colour. Every cube requires an approach, a top-down grasp,
a lift, a transfer and a release, so a full episode chains roughly fifty
sub-goals together. Grasps are verified with the jaw touch sensors and a
failed grasp is retried, which is what makes the task long-horizon rather than
a scripted replay.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .config import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    BIN_SITE_BY_COLOR,
    CUBE_BODIES,
    CUBE_HALF_SIZE,
    EE_SITE,
    GRIPPER_JOINT,
    HOME_QPOS,
    TABLE_SURFACE_Z,
    GripperConfig,
    IKConfig,
    SimConfig,
    TaskConfig,
    color_of,
)
from .controller import ArmController, GripperController
from .kinematics import DifferentialIK, top_down_quat


class TaskState(enum.Enum):
    """States of the sorting state machine."""

    HOME = "home"
    SELECT = "select"
    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSFER = "transfer"
    ALIGN = "align"
    RELEASE = "release"
    RETREAT = "retreat"
    RECOVER = "recover"
    DONE = "done"


# Per-state watchdog in seconds. Exceeding it sends the machine to RECOVER so a
# single bad grasp can never stall a whole episode.
STATE_TIMEOUTS: Dict[TaskState, float] = {
    TaskState.HOME: 8.0,
    TaskState.APPROACH: 10.0,
    TaskState.DESCEND: 8.0,
    TaskState.GRASP: 4.0,
    TaskState.LIFT: 8.0,
    TaskState.TRANSFER: 12.0,
    TaskState.ALIGN: 8.0,
    TaskState.RELEASE: 3.0,
    TaskState.RETREAT: 8.0,
    TaskState.RECOVER: 10.0,
}


@dataclass
class CubeStatus:
    """Bookkeeping for one cube across the whole episode."""

    body: str
    color: str
    attempts: int = 0
    sorted: bool = False
    abandoned: bool = False

    @property
    def pending(self) -> bool:
        return not self.sorted and not self.abandoned


@dataclass
class TaskEvent:
    """A timestamped transition, used for the run report and the dataset."""

    time: float
    state: str
    detail: str = ""
    cube: str = ""


class SortingTask:
    """Drives the arm through the colour-sorting task."""

    def __init__(
        self,
        model,
        data,
        sim_config: Optional[SimConfig] = None,
        task_config: Optional[TaskConfig] = None,
        ik_config: Optional[IKConfig] = None,
        gripper_config: Optional[GripperConfig] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.sim_cfg = sim_config or SimConfig()
        self.cfg = task_config or TaskConfig()

        self.ik = DifferentialIK(
            model,
            ARM_JOINTS,
            EE_SITE,
            rest_posture=HOME_QPOS,
            config=ik_config or IKConfig(),
        )
        self.arm = ArmController(model, data, ARM_JOINTS, ARM_ACTUATORS)
        self.gripper = GripperController(model, data, gripper_config or GripperConfig())

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self._body_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in CUBE_BODIES
        }
        self._bin_site_ids = {
            color: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            for color, site in BIN_SITE_BY_COLOR.items()
        }

        self.cubes: Dict[str, CubeStatus] = {
            name: CubeStatus(body=name, color=color_of(name)) for name in CUBE_BODIES
        }
        self.events: List[TaskEvent] = []

        self.state = TaskState.HOME
        self.current: Optional[str] = None
        self.grasp_yaw = 0.0
        self.elapsed = 0.0
        self._state_time = 0.0
        self._hold_time = 0.0
        self._ik_position_error = 0.0

    # ------------------------------------------------------------------ #
    # Scene helpers
    # ------------------------------------------------------------------ #

    def _cube_position(self, body: str) -> np.ndarray:
        return np.array(self.data.xpos[self._body_ids[body]], dtype=float)

    def _cube_yaw(self, body: str) -> float:
        w, x, y, z = self.data.xquat[self._body_ids[body]]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # A cube is symmetric every 90 degrees, so fold the yaw into a range the
        # wrist can always reach.
        quarter = math.pi / 2.0
        return yaw - quarter * round(yaw / quarter)

    def _bin_position(self, color: str) -> np.ndarray:
        return np.array(self.data.site_xpos[self._bin_site_ids[color]], dtype=float)

    def _ee_position(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.site_id], dtype=float)

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset the simulation, optionally randomising the cube layout."""
        mujoco.mj_resetData(self.model, self.data)

        for joint_name, value in zip(ARM_JOINTS, HOME_QPOS):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value

        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
        self.data.qpos[self.model.jnt_qposadr[gripper_id]] = self.gripper.cfg.open_travel

        if rng is not None and self.sim_cfg.randomize_layout:
            self._randomize_cubes(rng)

        mujoco.mj_forward(self.model, self.data)

        self.arm.reset(HOME_QPOS)
        self.gripper.reset(opened=True)

        for status in self.cubes.values():
            status.attempts = 0
            status.sorted = False
            status.abandoned = False

        self.events.clear()
        self.state = TaskState.HOME
        self.current = None
        self.elapsed = 0.0
        self._state_time = 0.0
        self._hold_time = 0.0

    def _randomize_cubes(self, rng: np.random.Generator) -> None:
        jitter = self.sim_cfg.layout_jitter_xy
        yaw_jitter = self.sim_cfg.layout_jitter_yaw
        for name in CUBE_BODIES:
            body_id = self._body_ids[name]
            joint_id = int(self.model.body_jntadr[body_id])
            if joint_id < 0:
                continue
            adr = int(self.model.jnt_qposadr[joint_id])
            base = np.array(self.model.body_pos[body_id], dtype=float)
            self.data.qpos[adr + 0] = base[0] + rng.uniform(-jitter, jitter)
            self.data.qpos[adr + 1] = base[1] + rng.uniform(-jitter, jitter)
            self.data.qpos[adr + 2] = TABLE_SURFACE_Z + CUBE_HALF_SIZE + 0.001
            yaw = float(rng.uniform(-yaw_jitter, yaw_jitter))
            self.data.qpos[adr + 3] = math.cos(yaw / 2.0)
            self.data.qpos[adr + 4] = 0.0
            self.data.qpos[adr + 5] = 0.0
            self.data.qpos[adr + 6] = math.sin(yaw / 2.0)

    # ------------------------------------------------------------------ #
    # Motion helpers
    # ------------------------------------------------------------------ #

    def _go_to(self, position: np.ndarray, yaw: float) -> None:
        target_quat = top_down_quat(yaw)
        q_target, position_error, _ = self.ik.solve(
            self.data, position, target_quat, q_init=self.arm.target
        )
        self._ik_position_error = position_error
        self.arm.set_target(q_target)

    def _reached(self, position: np.ndarray, tolerance: float) -> bool:
        return bool(np.linalg.norm(self._ee_position() - position) < tolerance)

    def _transition(self, state: TaskState, detail: str = "") -> None:
        self.events.append(
            TaskEvent(
                time=round(self.elapsed, 3),
                state=state.value,
                detail=detail,
                cube=self.current or "",
            )
        )
        self.state = state
        self._state_time = 0.0

    # ------------------------------------------------------------------ #
    # Main update
    # ------------------------------------------------------------------ #

    @property
    def finished(self) -> bool:
        return self.state is TaskState.DONE

    @property
    def sorted_count(self) -> int:
        return sum(1 for status in self.cubes.values() if status.sorted)

    def update(self, dt: float) -> None:
        """Advance the state machine by one control period."""
        self.elapsed += dt
        self._state_time += dt

        timeout = STATE_TIMEOUTS.get(self.state)
        if timeout is not None and self._state_time > timeout and self.state is not TaskState.DONE:
            self._fail_current("state timeout")
            return

        handler = getattr(self, f"_on_{self.state.value}")
        handler(dt)

        self.arm.step(dt)
        self.gripper.step(dt)

    # -- individual states ---------------------------------------------- #

    def _on_home(self, dt: float) -> None:
        self.arm.set_target(HOME_QPOS)
        self.gripper.open()
        if self.arm.settled(tolerance=0.05):
            self._transition(TaskState.SELECT)

    def _on_select(self, dt: float) -> None:
        pending = [status for status in self.cubes.values() if status.pending]
        if not pending:
            self.current = None
            self._transition(TaskState.DONE, "all cubes handled")
            return

        ee = self._ee_position()
        pending.sort(key=lambda s: float(np.linalg.norm(self._cube_position(s.body) - ee)))
        self.current = pending[0].body
        self.grasp_yaw = self._cube_yaw(self.current)
        self._transition(TaskState.APPROACH, "target selected")

    def _on_approach(self, dt: float) -> None:
        cube = self._cube_position(self.current)
        goal = np.array([cube[0], cube[1], TABLE_SURFACE_Z + self.cfg.hover_height])
        self._go_to(goal, self.grasp_yaw)
        self.gripper.open()
        if self._reached(goal, 0.012) and self.arm.command_reached():
            self._transition(TaskState.DESCEND)

    def _on_descend(self, dt: float) -> None:
        cube = self._cube_position(self.current)
        goal = np.array([cube[0], cube[1], cube[2] + self.cfg.grasp_offset])
        self._go_to(goal, self.grasp_yaw)
        if self._reached(goal, 0.006) and self.arm.command_reached():
            self._transition(TaskState.GRASP)

    def _on_grasp(self, dt: float) -> None:
        self.gripper.close()
        if not self.gripper.command_reached():
            return
        self._hold_time += dt
        if self._hold_time < self.cfg.settle_seconds:
            return
        self._hold_time = 0.0
        if self.gripper.holding_object():
            self._transition(TaskState.LIFT, "grasp confirmed")
        else:
            self._fail_current("grasp not detected")

    def _on_lift(self, dt: float) -> None:
        cube = self._cube_position(self.current)
        goal = np.array([cube[0], cube[1], TABLE_SURFACE_Z + self.cfg.lift_height])
        self._go_to(goal, self.grasp_yaw)
        if not self.gripper.holding_object():
            self._fail_current("object slipped during lift")
            return
        if self._reached(goal, 0.02):
            self._transition(TaskState.TRANSFER)

    def _on_transfer(self, dt: float) -> None:
        color = self.cubes[self.current].color
        target = self._bin_position(color)
        goal = np.array([target[0], target[1], TABLE_SURFACE_Z + self.cfg.lift_height])
        self._go_to(goal, 0.0)
        if not self.gripper.holding_object():
            self._fail_current("object slipped during transfer")
            return
        if self._reached(goal, 0.02):
            self._transition(TaskState.ALIGN)

    def _on_align(self, dt: float) -> None:
        color = self.cubes[self.current].color
        target = self._bin_position(color)
        goal = np.array([target[0], target[1], TABLE_SURFACE_Z + self.cfg.release_height])
        self._go_to(goal, 0.0)
        if self._reached(goal, 0.012):
            self._transition(TaskState.RELEASE)

    def _on_release(self, dt: float) -> None:
        self.gripper.open()
        if not self.gripper.command_reached():
            return
        self._hold_time += dt
        if self._hold_time < self.cfg.settle_seconds:
            return
        self._hold_time = 0.0
        status = self.cubes[self.current]
        status.sorted = True
        self._transition(TaskState.RETREAT, "cube released")

    def _on_retreat(self, dt: float) -> None:
        color = self.cubes[self.current].color if self.current else "red"
        target = self._bin_position(color)
        goal = np.array([target[0], target[1], TABLE_SURFACE_Z + self.cfg.lift_height])
        self._go_to(goal, 0.0)
        if self._reached(goal, 0.03):
            self.current = None
            self._transition(TaskState.HOME)

    def _on_recover(self, dt: float) -> None:
        self.gripper.open()
        self.arm.set_target(HOME_QPOS)
        if self.arm.settled(tolerance=0.08):
            self._transition(TaskState.SELECT, "recovered")

    def _on_done(self, dt: float) -> None:
        self.arm.set_target(HOME_QPOS)
        self.gripper.open()

    # ------------------------------------------------------------------ #

    def _fail_current(self, reason: str) -> None:
        if self.current is not None:
            status = self.cubes[self.current]
            status.attempts += 1
            if status.attempts >= self.cfg.max_grasp_attempts:
                status.abandoned = True
        self._hold_time = 0.0
        self._transition(TaskState.RECOVER, reason)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def evaluate(self) -> Dict[str, object]:
        """Geometric check of where every cube actually ended up."""
        placed: Dict[str, bool] = {}
        for name, status in self.cubes.items():
            cube = self._cube_position(name)
            target = self._bin_position(status.color)
            in_plane = float(np.linalg.norm(cube[:2] - target[:2]))
            placed[name] = bool(in_plane < self.cfg.success_radius and cube[2] < target[2] + 0.06)
        success = sum(1 for ok in placed.values() if ok)
        return {
            "cubes_total": len(placed),
            "cubes_in_correct_bin": success,
            "success_rate": round(success / max(len(placed), 1), 3),
            "per_cube": placed,
            "attempts": {name: st.attempts for name, st in self.cubes.items()},
            "abandoned": [name for name, st in self.cubes.items() if st.abandoned],
            "episode_seconds": round(self.elapsed, 2),
        }
