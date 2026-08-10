"""Long-Horizon Sorting Cell - a MuJoCo workcell for FFAI Robothon 2026.

The package is intentionally small and dependency-light:

* :mod:`sortbot.config`      - dataclass configuration for every stage
* :mod:`sortbot.kinematics`  - damped least-squares differential IK
* :mod:`sortbot.controller`  - joint-space and gripper controllers
* :mod:`sortbot.task`        - the long-horizon sorting state machine
* :mod:`sortbot.recorder`    - video and trajectory dataset recording
"""

from .config import (
    GripperConfig,
    IKConfig,
    RecordConfig,
    SimConfig,
    TaskConfig,
)
from .controller import ArmController, GripperController
from .kinematics import DifferentialIK, quat_error
from .recorder import EpisodeRecorder
from .task import SortingTask, TaskState

__all__ = [
    "ArmController",
    "DifferentialIK",
    "EpisodeRecorder",
    "GripperConfig",
    "GripperController",
    "IKConfig",
    "RecordConfig",
    "SimConfig",
    "SortingTask",
    "TaskConfig",
    "TaskState",
    "quat_error",
]

__version__ = "1.0.0"
