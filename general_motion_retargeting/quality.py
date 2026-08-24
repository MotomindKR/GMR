"""Quantitative quality checks for retargeted robot trajectories."""

from __future__ import annotations

from collections import Counter

import mujoco
import numpy as np
from scipy.signal import savgol_filter

from .motion_retarget import GeneralMotionRetargeting


def joint_jitter_metrics(qpos: np.ndarray, model: mujoco.MjModel) -> dict:
    hinge_qpos = np.asarray(
        [
            model.jnt_qposadr[joint_id]
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ],
        dtype=int,
    )
    joints = np.asarray(qpos[:, hinge_qpos], dtype=float)
    window = min(7, len(joints) if len(joints) % 2 else len(joints) - 1)
    if window >= 3:
        trend = savgol_filter(
            joints,
            window_length=window,
            polyorder=min(2, window - 1),
            axis=0,
            mode="interp",
        )
        residual_degrees = np.rad2deg(joints - trend)
    else:
        residual_degrees = np.zeros_like(joints)
    steps_degrees = np.rad2deg(np.diff(joints, axis=0))
    if len(steps_degrees):
        maximum_step = float(np.max(np.abs(steps_degrees)))
        reversals = (
            (steps_degrees[:-1] * steps_degrees[1:] < 0.0)
            & (np.abs(steps_degrees[:-1]) > 2.0)
            & (np.abs(steps_degrees[1:]) > 2.0)
        )
    else:
        maximum_step = 0.0
        reversals = np.zeros((0, joints.shape[1]), dtype=bool)
    return {
        "rms_degrees": float(np.sqrt(np.mean(residual_degrees**2))),
        "p99_degrees": float(np.percentile(np.abs(residual_degrees), 99)),
        "maximum_degrees": float(np.max(np.abs(residual_degrees))),
        "maximum_step_degrees": maximum_step,
        "isolated_reversals": int(reversals.sum()),
        "joint_samples": int(joints.size),
    }


def joint_limit_saturation_metrics(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    *,
    margin_degrees: float = 1.0,
    minimum_fraction: float = 0.05,
) -> dict:
    """Report joint ranges that a trajectory persistently presses against."""
    if margin_degrees < 0.0:
        raise ValueError("joint-limit diagnostic margin cannot be negative")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("joint-limit diagnostic fraction must be in [0, 1]")

    margin_radians = np.deg2rad(margin_degrees)
    saturated_ranges = []
    for joint_id in range(model.njnt):
        if (
            model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
            or not model.jnt_limited[joint_id]
        ):
            continue
        address = model.jnt_qposadr[joint_id]
        values = np.asarray(qpos[:, address], dtype=float)
        lower, upper = model.jnt_range[joint_id]
        near_lower_fraction = float(np.mean(values - lower <= margin_radians))
        near_upper_fraction = float(np.mean(upper - values <= margin_radians))
        if max(near_lower_fraction, near_upper_fraction) < minimum_fraction:
            continue
        if (
            near_lower_fraction >= minimum_fraction
            and near_upper_fraction >= minimum_fraction
        ):
            limiting_bound = "both"
        elif near_lower_fraction >= minimum_fraction:
            limiting_bound = "lower"
        else:
            limiting_bound = "upper"
        saturated_ranges.append(
            {
                "joint_name": model.joint(joint_id).name,
                "range_degrees": [
                    float(np.rad2deg(lower)),
                    float(np.rad2deg(upper)),
                ],
                "observed_range_degrees": [
                    float(np.rad2deg(np.min(values))),
                    float(np.rad2deg(np.max(values))),
                ],
                "limiting_bound": limiting_bound,
                "near_lower_limit_percent": 100.0 * near_lower_fraction,
                "near_upper_limit_percent": 100.0 * near_upper_fraction,
            }
        )
    saturated_ranges.sort(
        key=lambda item: max(
            item["near_lower_limit_percent"],
            item["near_upper_limit_percent"],
        ),
        reverse=True,
    )
    return {
        "margin_degrees": float(margin_degrees),
        "minimum_reported_percent": 100.0 * minimum_fraction,
        "saturated_joint_ranges": saturated_ranges,
    }


def model_trajectory_metrics(
    qpos: np.ndarray,
    retargeter: GeneralMotionRetargeting,
    fps: float,
) -> dict:
    model = retargeter.model
    data = mujoco.MjData(model)
    collision_pairs = Counter()
    minimum_contact_distance = 0.0
    sole_heights = {
        model.geom(geom_id).name: []
        for geom_id in retargeter.ground_clearance_geom_ids
    }
    sole_positions = {name: [] for name in sole_heights}

    for configuration in qpos:
        data.qpos[:] = configuration
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            pair = tuple(
                sorted(
                    (
                        model.geom(contact.geom[0]).name,
                        model.geom(contact.geom[1]).name,
                    )
                )
            )
            collision_pairs[pair] += 1
            minimum_contact_distance = min(
                minimum_contact_distance, float(contact.dist)
            )
        for geom_id in retargeter.ground_clearance_geom_ids:
            name = model.geom(geom_id).name
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            half_height = np.sum(
                np.abs(rotation[2]) * model.geom_size[geom_id]
            )
            sole_heights[name].append(data.geom_xpos[geom_id, 2] - half_height)
            sole_positions[name].append(data.geom_xpos[geom_id, :2].copy())

    joint_limit_margin = np.inf
    for joint_id in range(model.njnt):
        if (
            model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
            or not model.jnt_limited[joint_id]
        ):
            continue
        address = model.jnt_qposadr[joint_id]
        low, high = model.jnt_range[joint_id]
        joint_limit_margin = min(
            joint_limit_margin,
            float(np.min(qpos[:, address] - low)),
            float(np.min(high - qpos[:, address])),
        )

    feet = {}
    for name, heights in sole_heights.items():
        heights = np.asarray(heights)
        positions = np.asarray(sole_positions[name])
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps
        support = heights[:-1] <= 0.01
        support_speed = speed[support]
        feet[name] = {
            "minimum_sole_height_meters": float(np.min(heights)),
            "support_frame_count": int(support.sum()),
            "support_sliding_p95_meters_per_second": (
                float(np.percentile(support_speed, 95))
                if len(support_speed)
                else 0.0
            ),
        }

    return {
        "jitter": joint_jitter_metrics(qpos, model),
        "collision_contact_samples": int(sum(collision_pairs.values())),
        "collision_pair_count": len(collision_pairs),
        "minimum_contact_distance_meters": minimum_contact_distance,
        "most_common_collision_pairs": [
            {"geoms": list(pair), "samples": samples}
            for pair, samples in collision_pairs.most_common(10)
        ],
        "minimum_joint_limit_margin_radians": float(max(0.0, joint_limit_margin)),
        "joint_limit_diagnostics": joint_limit_saturation_metrics(qpos, model),
        "feet": feet,
    }


def task_tracking_metrics(
    qpos: np.ndarray,
    human_frames,
    retargeter: GeneralMotionRetargeting,
) -> dict:
    requested = {
        "pelvis",
        "spine3",
        "left_knee",
        "right_knee",
        "left_foot",
        "right_foot",
        "left_wrist",
        "right_wrist",
    }
    tasks = {
        human_name: task
        for human_name, task in retargeter.human_body_to_task2.items()
        if human_name in requested
    }
    errors = {
        human_name: {"position": [], "orientation": []}
        for human_name in tasks
    }
    for frame, configuration in zip(human_frames, qpos, strict=True):
        retargeter.update_targets(frame)
        retargeter.configuration.update(configuration)
        for human_name, task in tasks.items():
            error = task.compute_error(retargeter.configuration)
            position_mask = task.cost[:3] > 0.0
            orientation_mask = task.cost[3:] > 0.0
            if np.any(position_mask):
                errors[human_name]["position"].append(
                    np.linalg.norm(error[:3][position_mask])
                )
            if np.any(orientation_mask):
                errors[human_name]["orientation"].append(
                    np.linalg.norm(error[3:][orientation_mask])
                )

    report = {}
    for human_name, components in errors.items():
        metrics = {}
        if components["position"]:
            position = np.asarray(components["position"])
            metrics.update(
                position_median_meters=float(np.median(position)),
                position_p95_meters=float(np.percentile(position, 95)),
            )
        if components["orientation"]:
            orientation = np.rad2deg(components["orientation"])
            metrics.update(
                orientation_median_degrees=float(np.median(orientation)),
                orientation_p95_degrees=float(np.percentile(orientation, 95)),
            )
        report[human_name] = metrics
    return report


def evaluate_retargeted_motion(
    qpos: np.ndarray,
    human_frames,
    retargeter: GeneralMotionRetargeting,
    fps: float,
) -> dict:
    if len(qpos) != len(human_frames):
        raise ValueError(
            f"frame-count mismatch: robot={len(qpos)}, human={len(human_frames)}"
        )
    report = {
        "frame_count": len(qpos),
        "fps": float(fps),
        "task_profile": retargeter.task_profile,
        "model": model_trajectory_metrics(qpos, retargeter, fps),
        "tracking": task_tracking_metrics(qpos, human_frames, retargeter),
    }
    report["quality_gate"] = evaluate_quality_gate(
        report, retargeter.quality_thresholds
    )
    return report


def evaluate_quality_gate(report: dict, thresholds: dict) -> dict:
    violations = []
    model = report["model"]
    tracking = report["tracking"]

    def maximum_tracking_value(suffix: str, names: tuple[str, ...]) -> float:
        values = [
            tracking[name][suffix]
            for name in names
            if name in tracking and suffix in tracking[name]
        ]
        return max(values) if values else 0.0

    checks = (
        (
            "jitter_rms_degrees",
            model["jitter"]["rms_degrees"],
            thresholds.get("maximum_jitter_rms_degrees"),
        ),
        (
            "collision_penetration_meters",
            -model["minimum_contact_distance_meters"],
            thresholds.get("maximum_collision_penetration_meters"),
        ),
        (
            "wrist_position_p95_meters",
            maximum_tracking_value(
                "position_p95_meters", ("left_wrist", "right_wrist")
            ),
            thresholds.get("maximum_wrist_position_p95_meters"),
        ),
        (
            "wrist_orientation_p95_degrees",
            maximum_tracking_value(
                "orientation_p95_degrees", ("left_wrist", "right_wrist")
            ),
            thresholds.get("maximum_wrist_orientation_p95_degrees"),
        ),
        (
            "foot_orientation_p95_degrees",
            maximum_tracking_value(
                "orientation_p95_degrees", ("left_foot", "right_foot")
            ),
            thresholds.get("maximum_foot_orientation_p95_degrees"),
        ),
    )
    for name, value, maximum in checks:
        if maximum is not None and value > maximum:
            violations.append(
                {"metric": name, "value": float(value), "maximum": float(maximum)}
            )
    return {"passed": not violations, "violations": violations}
