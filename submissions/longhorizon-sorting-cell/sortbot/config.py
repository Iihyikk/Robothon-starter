"""Typed configuration for the Long-Horizon Sorting Cell.

Every tunable number used by the demo lives here, so the behaviour of the
workcell can be changed without touching control or task logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "scene" / "sorting_cell.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------- #
# Model naming. Keep in sync with scene/arm.xml and scene/sorting_cell.xml.
# --------------------------------------------------------------------------- #

ARM_JOINTS: Tuple[str, ...] = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)

ARM_ACTUATORS: Tuple[str, ...] = tuple(f"act_{name}" for name in ARM_JOINTS)
GRIPPER_ACTUATOR = "act_gripper"
GRIPPER_JOINT = "left_jaw_joint"
EE_SITE = "ee_site"

# Nominal posture: gripper hovering above the middle of the workbench with the
# jaws pointing straight down. Also used as the null-space bias for the IK.
HOME_QPOS: Tuple[float, ...] = (0.00, 0.30, 1.15, 0.55, 0.00, 1.14)

CUBE_BODIES: Tuple[str, ...] = (
    "cube_red_1",
    "cube_green_1",
    "cube_blue_1",
    "cube_red_2",
    "cube_green_2",
    "cube_blue_2",
)

BIN_SITE_BY_COLOR: Dict[str, str] = {
    "red": "bin_red_site",
    "green": "bin_green_site",
    "blue": "bin_blue_site",
}

TABLE_SURFACE_Z: float = 0.37
CUBE_HALF_SIZE: float = 0.0125


def color_of(body_name: str) -> str:
    """Return the colour label encoded in a cube body name."""
    parts = body_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"unexpected cube body name: {body_name!r}")
    return parts[1]


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimConfig:
    """Simulation loop settings."""

    scene_path: Path = SCENE_PATH
    control_hz: float = 50.0
    max_episode_seconds: float = 300.0
    seed: int = 0
    randomize_layout: bool = True
    layout_jitter_xy: float = 0.025
    layout_jitter_yaw: float = 0.45


@dataclass(frozen=True)
class IKConfig:
    """Damped least-squares differential inverse kinematics settings."""

    damping: float = 0.08
    max_iterations: int = 150
    position_tolerance: float = 1.2e-3
    rotation_tolerance: float = 1.5e-2
    step_scale: float = 0.65
    max_joint_step: float = 0.25
    nullspace_gain: float = 0.035
    rotation_weight: float = 0.6


@dataclass(frozen=True)
class GripperConfig:
    """Parallel-jaw gripper settings, expressed as a single-jaw travel."""

    open_travel: float = 0.032
    closed_travel: float = 0.0015
    close_seconds: float = 0.45
    open_seconds: float = 0.30
    contact_force_threshold: float = 1.2
    min_grasp_travel: float = 0.0035


@dataclass(frozen=True)
class TaskConfig:
    """Long-horizon sorting task settings."""

    hover_height: float = 0.14
    grasp_offset: float = 0.001
    lift_height: float = 0.24
    release_height: float = 0.13
    move_speed: float = 0.32
    settle_seconds: float = 0.35
    max_grasp_attempts: int = 3
    success_radius: float = 0.085


@dataclass(frozen=True)
class RecordConfig:
    """Video and dataset recording settings."""

    enabled: bool = True
    camera: str = "overview"
    width: int = 1280
    height: int = 720
    fps: int = 30
    wrist_camera: str = "wrist_cam"
    wrist_width: int = 320
    wrist_height: int = 240
    record_wrist_frames: bool = True
    dataset_stride: int = 2
    output_dir: Path = DEFAULT_OUTPUT_DIR
