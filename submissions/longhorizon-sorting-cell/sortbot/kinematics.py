"""Damped least-squares differential inverse kinematics.

The solver runs on a scratch mujoco.MjData object so it never disturbs the
live simulation state. It returns a joint-space target that the controller
tracks with the position actuators declared in scene/arm.xml.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

from .config import IKConfig


def quat_error(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    """Rotation vector that rotates current_quat onto target_quat."""
    conjugate = np.zeros(4)
    mujoco.mju_negQuat(conjugate, np.asarray(current_quat, dtype=float))
    delta = np.zeros(4)
    mujoco.mju_mulQuat(delta, np.asarray(target_quat, dtype=float), conjugate)
    rotation = np.zeros(3)
    mujoco.mju_quat2Vel(rotation, delta, 1.0)
    return rotation


def top_down_quat(yaw: float = 0.0) -> np.ndarray:
    """Orientation whose local +Z axis points straight down, spun by yaw."""
    flip = np.zeros(4)
    mujoco.mju_axisAngle2Quat(flip, np.array([0.0, 1.0, 0.0]), np.pi)
    spin = np.zeros(4)
    mujoco.mju_axisAngle2Quat(spin, np.array([0.0, 0.0, 1.0]), float(yaw))
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, spin, flip)
    return result


class DifferentialIK:
    """Cartesian target -> joint target, using a damped pseudo-inverse.

    Notes
    -----
    * Joint limits are respected by clipping after every update.
    * A null-space term biases the arm towards a rest posture, which keeps the
      elbow away from singular configurations during long sorting sequences.
    """

    def __init__(
        self,
        model,
        joint_names: Sequence[str],
        site_name: str,
        rest_posture: Optional[Sequence[float]] = None,
        config: Optional[IKConfig] = None,
    ) -> None:
        self.model = model
        self.cfg = config or IKConfig()

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"unknown site {site_name!r}")

        joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"unknown joint {name!r}")
            joint_ids.append(jid)

        self.joint_ids = np.asarray(joint_ids, dtype=int)
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        self.lower = model.jnt_range[self.joint_ids, 0].copy()
        self.upper = model.jnt_range[self.joint_ids, 1].copy()

        if rest_posture is None:
            rest_posture = np.zeros(len(joint_ids))
        self.rest_posture = np.asarray(rest_posture, dtype=float)

        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------------ #

    @property
    def n_joints(self) -> int:
        return int(self.joint_ids.size)

    def forward(self, data, q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics of the end-effector site for joint vector q."""
        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        scratch.qpos[self.qpos_ids] = np.asarray(q, dtype=float)
        mujoco.mj_kinematics(self.model, scratch)
        position = np.array(scratch.site_xpos[self.site_id])
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, scratch.site_xmat[self.site_id])
        return position, quaternion

    def solve(
        self,
        data,
        target_pos: Sequence[float],
        target_quat: Optional[Sequence[float]] = None,
        q_init: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Solve IK and return (q_target, position_error, rotation_error)."""
        cfg = self.cfg
        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0

        if q_init is None:
            q = np.array(data.qpos[self.qpos_ids], dtype=float)
        else:
            q = np.asarray(q_init, dtype=float).copy()

        target_pos = np.asarray(target_pos, dtype=float)
        target_quat = None if target_quat is None else np.asarray(target_quat, dtype=float)

        identity = np.eye(self.n_joints)
        position_error = float("inf")
        rotation_error = 0.0

        for _ in range(cfg.max_iterations):
            scratch.qpos[self.qpos_ids] = q
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)

            delta_pos = target_pos - scratch.site_xpos[self.site_id]
            position_error = float(np.linalg.norm(delta_pos))

            if target_quat is None:
                delta_rot = np.zeros(3)
                rotation_error = 0.0
            else:
                site_quat = np.zeros(4)
                mujoco.mju_mat2Quat(site_quat, scratch.site_xmat[self.site_id])
                delta_rot = quat_error(target_quat, site_quat)
                rotation_error = float(np.linalg.norm(delta_rot))

            if (position_error < cfg.position_tolerance
                    and rotation_error < cfg.rotation_tolerance):
                break

            mujoco.mj_jacSite(self.model, scratch, self._jacp, self._jacr, self.site_id)
            jacobian = np.vstack(
                (
                    self._jacp[:, self.dof_ids],
                    cfg.rotation_weight * self._jacr[:, self.dof_ids],
                )
            )
            residual = np.concatenate((delta_pos, cfg.rotation_weight * delta_rot))

            hessian = jacobian.T @ jacobian + (cfg.damping ** 2) * identity
            gradient = jacobian.T @ residual

            if cfg.nullspace_gain > 0.0:
                projector = identity - np.linalg.pinv(jacobian) @ jacobian
                gradient = gradient + cfg.nullspace_gain * projector @ (self.rest_posture - q)

            step = np.linalg.solve(hessian, gradient)
            step = np.clip(cfg.step_scale * step, -cfg.max_joint_step, cfg.max_joint_step)
            q = np.clip(q + step, self.lower, self.upper)

        return q, position_error, rotation_error
