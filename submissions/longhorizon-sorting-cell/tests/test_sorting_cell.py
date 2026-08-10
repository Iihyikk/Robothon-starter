"""Smoke tests for the Long-Horizon Sorting Cell.

Run them with:

    python -m pip install pytest
    python -m pytest submissions/longhorizon-sorting-cell/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sortbot.config import (  # noqa: E402
    ARM_ACTUATORS,
    ARM_JOINTS,
    CUBE_BODIES,
    EE_SITE,
    HOME_QPOS,
    SCENE_PATH,
)
from sortbot.kinematics import DifferentialIK, quat_error, top_down_quat  # noqa: E402
from sortbot.task import SortingTask, TaskState  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


def _name_exists(model, obj_type, name) -> bool:
    return mujoco.mj_name2id(model, obj_type, name) >= 0


def test_scene_compiles_with_expected_interface(model):
    assert model.nu == len(ARM_ACTUATORS) + 1
    assert _name_exists(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    for joint in ARM_JOINTS:
        assert _name_exists(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    for actuator in ARM_ACTUATORS:
        assert _name_exists(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
    for cube in CUBE_BODIES:
        assert _name_exists(model, mujoco.mjtObj.mjOBJ_BODY, cube)
    for camera in ("overview", "front", "topdown", "wrist_cam"):
        assert _name_exists(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)


def test_top_down_quat_points_the_gripper_down():
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, top_down_quat(0.0))
    approach_axis = matrix.reshape(3, 3)[:, 2]
    assert approach_axis[2] == pytest.approx(-1.0, abs=1e-6)


def test_quat_error_is_zero_for_identical_orientations():
    quat = top_down_quat(0.37)
    assert np.allclose(quat_error(quat, quat), np.zeros(3), atol=1e-9)


def test_ik_round_trip(model):
    data = mujoco.MjData(model)
    solver = DifferentialIK(model, ARM_JOINTS, EE_SITE, rest_posture=HOME_QPOS)

    rng = np.random.default_rng(3)
    reference = np.asarray(HOME_QPOS) + rng.uniform(-0.12, 0.12, size=len(HOME_QPOS))
    target_pos, target_quat = solver.forward(data, reference)

    solution, position_error, rotation_error = solver.solve(
        data, target_pos, target_quat, q_init=HOME_QPOS
    )

    assert position_error < 2e-3
    assert rotation_error < 3e-2

    achieved_pos, _ = solver.forward(data, solution)
    assert np.linalg.norm(achieved_pos - target_pos) < 2e-3


def test_reset_places_every_cube_on_the_table(model):
    data = mujoco.MjData(model)
    task = SortingTask(model, data)
    task.reset(np.random.default_rng(0))

    assert task.state is TaskState.HOME
    assert task.sorted_count == 0
    for cube in CUBE_BODIES:
        position = task._cube_position(cube)
        assert 0.20 < position[0] < 0.72
        assert -0.44 < position[1] < 0.44
        assert 0.36 < position[2] < 0.42


def test_state_machine_advances_out_of_home(model):
    data = mujoco.MjData(model)
    task = SortingTask(model, data)
    task.reset(np.random.default_rng(1))

    control_dt = 1.0 / task.sim_cfg.control_hz
    steps = max(1, int(round(control_dt / model.opt.timestep)))
    for _ in range(400):
        task.update(control_dt)
        for _ in range(steps):
            mujoco.mj_step(model, data)
        if task.state not in (TaskState.HOME, TaskState.SELECT):
            break

    assert task.state not in (TaskState.HOME, TaskState.SELECT)
