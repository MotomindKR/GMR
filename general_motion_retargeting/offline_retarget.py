"""Shared profile-controlled offline retargeting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .motion_retarget import GeneralMotionRetargeting


@dataclass(frozen=True)
class OfflineRetargetResult:
    qpos: np.ndarray
    retargeter: GeneralMotionRetargeting
    smoothing_alpha: np.ndarray
    use_velocity_limit: bool
    passes_per_frame: int
    initial_settle_passes: int
    joint_smoothing_kernel: np.ndarray
    root_orientation_smoothing_kernel: np.ndarray
    joint_smoothing_passes: int


def _validated_kernel(values) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if (
        weights.ndim != 1
        or len(weights) % 2 != 1
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError(
            "joint smoothing kernel must be finite, non-negative, odd-length, "
            "and sum to one"
        )
    return weights


def _minimum_contact_distance(
    data: mujoco.MjData,
    monitored_geom_pairs: set[tuple[int, int]] | None,
) -> float:
    distances = [
        contact.dist
        for contact in data.contact
        if monitored_geom_pairs is None
        or tuple(sorted((int(contact.geom[0]), int(contact.geom[1]))))
        in monitored_geom_pairs
    ]
    return min(0.0, min(distances)) if distances else 0.0


def _body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    body_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data.xpos[body_ids].copy(), data.xquat[body_ids].copy()


def _smooth_root_orientations(
    trajectory: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    if len(weights) == 1:
        return trajectory[:, 3:7].copy()
    rotations = Rotation.from_quat(trajectory[:, 3:7], scalar_first=True)
    radius = len(weights) // 2
    candidate = []
    for frame, center in enumerate(rotations):
        indices = np.clip(
            np.arange(frame - radius, frame + radius + 1),
            0,
            len(trajectory) - 1,
        )
        relative = (center.inv() * rotations[indices]).as_rotvec()
        candidate.append(
            (center * Rotation.from_rotvec(weights @ relative)).as_quat(
                scalar_first=True
            )
        )
    return np.asarray(candidate)


def smooth_trajectory(
    trajectory: np.ndarray,
    model: mujoco.MjModel,
    config: dict,
    monitored_geom_pairs: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    joint_weights = _validated_kernel(
        config.get("joint_smoothing_kernel", [1.0])
    )
    root_weights = _validated_kernel(
        config.get("root_orientation_smoothing_kernel", [1.0])
    )
    if len(joint_weights) == 1 and len(root_weights) == 1:
        return trajectory, np.ones(len(trajectory), dtype=float)

    hinge_qpos = np.asarray(
        [
            model.jnt_qposadr[joint_id]
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ],
        dtype=int,
    )
    candidate = trajectory.copy()
    if len(joint_weights) > 1:
        radius = len(joint_weights) // 2
        padded = np.pad(
            trajectory[:, hinge_qpos],
            ((radius, radius), (0, 0)),
            mode="edge",
        )
        candidate[:, hinge_qpos] = sum(
            weight * padded[offset : offset + len(trajectory)]
            for offset, weight in enumerate(joint_weights)
        )
    candidate[:, 3:7] = _smooth_root_orientations(trajectory, root_weights)

    protected = config.get("protected_bodies", [])
    body_ids = np.asarray(
        [model.body(item["body_name"]).id for item in protected], dtype=int
    )
    position_limits = np.asarray(
        [item["max_position_delta"] for item in protected], dtype=float
    )
    orientation_limits = np.asarray(
        [item["max_orientation_delta_radians"] for item in protected], dtype=float
    )
    if np.any(position_limits <= 0.0) or np.any(orientation_limits <= 0.0):
        raise ValueError("protected-body smoothing limits must be positive")

    data = mujoco.MjData(model)
    alpha = np.ones(len(trajectory), dtype=float)
    collision_tolerance = float(
        config.get("max_additional_collision_penetration", 0.0)
    )
    if collision_tolerance < 0.0:
        raise ValueError("additional collision penetration must be non-negative")
    maximum_collision_penetration = config.get(
        "maximum_collision_penetration"
    )
    if maximum_collision_penetration is not None:
        maximum_collision_penetration = float(maximum_collision_penetration)
        if maximum_collision_penetration < 0.0:
            raise ValueError("maximum collision penetration must be non-negative")

    for frame in range(len(trajectory)):
        raw_position, raw_quaternion = _body_pose(
            model, data, trajectory[frame], body_ids
        )
        raw_contact_distance = _minimum_contact_distance(
            data, monitored_geom_pairs
        )
        candidate_position, candidate_quaternion = _body_pose(
            model, data, candidate[frame], body_ids
        )
        if len(body_ids):
            position_delta = np.linalg.norm(
                candidate_position - raw_position, axis=1
            )
            quaternion_dot = np.abs(
                np.sum(candidate_quaternion * raw_quaternion, axis=1)
            )
            orientation_delta = 2.0 * np.arccos(
                np.clip(quaternion_dot, -1.0, 1.0)
            )
            ratios = np.concatenate(
                (
                    position_limits / np.maximum(position_delta, 1e-12),
                    orientation_limits / np.maximum(orientation_delta, 1e-12),
                )
            )
            alpha[frame] = min(1.0, float(np.min(ratios)))

        for _ in range(9):
            blended = trajectory[frame].copy()
            blended[hinge_qpos] += alpha[frame] * (
                candidate[frame, hinge_qpos] - trajectory[frame, hinge_qpos]
            )
            root_rotation = Rotation.from_quat(
                trajectory[frame, 3:7], scalar_first=True
            )
            candidate_root = Rotation.from_quat(
                candidate[frame, 3:7], scalar_first=True
            )
            root_delta = (root_rotation.inv() * candidate_root).as_rotvec()
            blended[3:7] = (
                root_rotation
                * Rotation.from_rotvec(alpha[frame] * root_delta)
            ).as_quat(scalar_first=True)
            _body_pose(model, data, blended, body_ids)
            minimum_allowed_contact_distance = (
                raw_contact_distance - collision_tolerance
            )
            if maximum_collision_penetration is not None:
                minimum_allowed_contact_distance = max(
                    minimum_allowed_contact_distance,
                    -maximum_collision_penetration,
                )
            if _minimum_contact_distance(
                data, monitored_geom_pairs
            ) >= minimum_allowed_contact_distance:
                candidate[frame] = blended
                break
            alpha[frame] *= 0.5
        else:
            candidate[frame] = trajectory[frame]
            alpha[frame] = 0.0

    return candidate, alpha


def retarget_offline_frames(
    human_frames,
    *,
    src_human: str,
    tgt_robot: str,
    actual_human_height: float,
    task_profile: str | None = None,
    use_velocity_limit: bool | None = None,
    passes_per_frame: int | None = None,
    initial_settle_passes: int | None = None,
    verbose: bool = False,
) -> OfflineRetargetResult:
    retargeter = GeneralMotionRetargeting(
        src_human=src_human,
        tgt_robot=tgt_robot,
        actual_human_height=actual_human_height,
        verbose=verbose,
        use_velocity_limit=use_velocity_limit,
        task_profile=task_profile,
    )
    config = retargeter.offline_solver_config
    resolved_velocity_limit = retargeter.use_velocity_limit
    resolved_passes = (
        int(config.get("passes_per_frame", 1))
        if passes_per_frame is None
        else passes_per_frame
    )
    resolved_settle = (
        int(config.get("initial_settle_passes", 5))
        if initial_settle_passes is None
        else initial_settle_passes
    )
    if resolved_passes < 1 or resolved_settle < 1:
        raise ValueError("offline IK pass counts must be positive")

    trajectory = []
    for frame_index, frame in enumerate(human_frames):
        iterations = resolved_settle if frame_index == 0 else resolved_passes
        for _ in range(iterations):
            qpos = retargeter.retarget(frame)
        trajectory.append(qpos)
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or not np.all(np.isfinite(trajectory)):
        raise ValueError("retargeting produced an invalid trajectory")

    collision_limit = retargeter.collision_avoidance_limit
    monitored_geom_pairs = (
        {
            tuple(sorted((int(geom1), int(geom2))))
            for geom1, geom2 in collision_limit.geom_id_pairs
        }
        if collision_limit is not None
        else None
    )
    smoothing_passes = int(config.get("joint_smoothing_passes", 1))
    if smoothing_passes < 1:
        raise ValueError("joint smoothing passes must be positive")
    smoothing_alpha = np.ones(len(trajectory), dtype=float)
    for _ in range(smoothing_passes):
        trajectory, pass_alpha = smooth_trajectory(
            trajectory,
            retargeter.model,
            config,
            monitored_geom_pairs,
        )
        smoothing_alpha = np.minimum(smoothing_alpha, pass_alpha)
    return OfflineRetargetResult(
        qpos=trajectory,
        retargeter=retargeter,
        smoothing_alpha=smoothing_alpha,
        use_velocity_limit=resolved_velocity_limit,
        passes_per_frame=resolved_passes,
        initial_settle_passes=resolved_settle,
        joint_smoothing_kernel=_validated_kernel(
            config.get("joint_smoothing_kernel", [1.0])
        ),
        root_orientation_smoothing_kernel=_validated_kernel(
            config.get("root_orientation_smoothing_kernel", [1.0])
        ),
        joint_smoothing_passes=smoothing_passes,
    )
