#!/usr/bin/env python3
"""Replay a retargeted Bello motion through the versioned reference service."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import signal
import time

import numpy as np

from general_motion_retargeting import BelloReferenceServer


GMR_BELLO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "neck_yaw_joint",
    "head_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
)


def load_motion(
    path: Path, expected_joint_names: tuple[str, ...]
) -> tuple[np.ndarray, float]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            fps = float(np.asarray(archive["fps"]).reshape(()))
            joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
        return _reorder_qpos(qpos, joint_names, expected_joint_names), fps
    with path.open("rb") as source:
        document = pickle.load(source)  # noqa: S301 - explicit trusted local input
    if not isinstance(document, dict):
        raise TypeError("GMR pickle must contain a motion mapping")
    root_position = np.asarray(document["root_pos"], dtype=np.float64)
    root_xyzw = np.asarray(document["root_rot"], dtype=np.float64)
    joint_position = np.asarray(document["dof_pos"], dtype=np.float64)
    if joint_position.shape[1] == len(GMR_BELLO_JOINT_NAMES):
        joint_names = GMR_BELLO_JOINT_NAMES
    elif joint_position.shape[1] == len(expected_joint_names):
        joint_names = expected_joint_names
    else:
        raise ValueError(
            "GMR Bello joint count does not match the fixed-head or legacy schema"
        )
    qpos = np.concatenate(
        (root_position, root_xyzw[:, (3, 0, 1, 2)], joint_position), axis=1
    )
    return (
        _reorder_qpos(qpos, joint_names, expected_joint_names),
        float(document["fps"]),
    )


def _reorder_qpos(
    qpos: np.ndarray,
    joint_names: tuple[str, ...],
    expected_joint_names: tuple[str, ...],
) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[1] != 7 + len(joint_names):
        raise ValueError("motion qpos shape does not match its joint schema")
    columns = {name: index for index, name in enumerate(joint_names)}
    missing = sorted(set(expected_joint_names) - set(columns))
    if missing:
        raise ValueError(f"motion is missing Bello joints: {missing}")
    reordered = np.concatenate(
        (
            qpos[:, :7],
            np.stack(
                [qpos[:, 7 + columns[name]] for name in expected_joint_names],
                axis=1,
            ),
        ),
        axis=1,
    )
    norms = np.linalg.norm(reordered[:, 3:7], axis=1, keepdims=True)
    if np.any(norms < 1.0e-8) or not np.all(np.isfinite(reordered)):
        raise ValueError("motion contains invalid root quaternions or values")
    reordered[:, 3:7] /= norms
    return reordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--listen", default="127.0.0.1:50053")
    parser.add_argument("--source-id", default="gmr-replay")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--hold-final", action="store_true")
    args = parser.parse_args()

    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with BelloReferenceServer(
        args.robot_xml,
        listen=args.listen,
        source_id=args.source_id,
    ) as server:
        qpos, fps = load_motion(args.motion, server.joint_names)
        period = 1.0 / fps
        frame = 0
        last_qpos = qpos[0]
        deadline = time.monotonic()
        while running:
            last_qpos = qpos[frame]
            server.publish_qpos(last_qpos, mode="active")
            frame += 1
            if frame == len(qpos):
                if not args.loop:
                    break
                frame = 0
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
        if args.hold_final and running:
            while running:
                last_qpos = qpos[-1]
                server.publish_qpos(last_qpos, mode="hold")
                time.sleep(period)
        server.publish_qpos(last_qpos, mode="stop")


if __name__ == "__main__":
    main()
