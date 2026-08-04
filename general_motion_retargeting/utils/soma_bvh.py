"""SOMA BVH loading and conversion to GMR's SMPL-X body convention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .lafan1_smplx import (
    Y_UP_TO_Z_UP,
    load_smplx_rest_skeleton,
    read_fps,
    sha256,
    solve_minimal_twist_smpl_rotations,
)
from .lafan_vendor import utils


SMPLX_JOINT_TO_SOMA = {
    "pelvis": "Hips",
    "left_hip": "LeftLeg",
    "right_hip": "RightLeg",
    "spine1": "Spine1",
    "left_knee": "LeftShin",
    "right_knee": "RightShin",
    "spine2": "Spine2",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Chest",
    "left_foot": "LeftToeBase",
    "right_foot": "RightToeBase",
    "neck": "Neck2",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}


@dataclass(frozen=True)
class SomaAnimation:
    """Local transforms and hierarchy read from a SOMA BVH file."""

    quaternions: np.ndarray
    positions: np.ndarray
    parents: np.ndarray
    names: tuple[str, ...]


def read_soma_bvh(path: Path) -> SomaAnimation:
    """Read BVH files that contain translation channels below the root.

    SOMA exports a zero-valued dummy ``Root`` followed by a six-channel
    ``Hips`` joint. The LAFAN reader assumes only the root is translated and
    cannot parse this layout.
    """

    path = path.resolve()
    lines = path.read_text(errors="strict").splitlines()
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    active = -1
    pending: int | None = None
    stack: list[int] = []
    in_end_site = False
    motion_line = None

    for line_number, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "MOTION":
            motion_line = line_number
            break
        joint = re.fullmatch(r"(?:ROOT|JOINT)\s+(\w+)", stripped)
        if joint is not None:
            names.append(joint.group(1))
            parents.append(active)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            pending = len(names) - 1
            continue
        if stripped == "End Site":
            pending = None
            in_end_site = True
            continue
        if stripped == "{":
            stack.append(active)
            if pending is not None:
                active = pending
            pending = None
            continue
        if stripped == "}":
            if not stack:
                raise ValueError(f"unbalanced BVH hierarchy in {path}")
            active = stack.pop()
            in_end_site = False
            continue
        offset = re.fullmatch(
            r"OFFSET\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)",
            stripped,
        )
        if offset is not None and active >= 0 and not in_end_site:
            offsets[active] = [float(value) for value in offset.groups()]
            continue
        channel_line = re.fullmatch(r"CHANNELS\s+(\d+)\s+(.+)", stripped)
        if channel_line is not None and active >= 0:
            joint_channels = channel_line.group(2).split()
            if len(joint_channels) != int(channel_line.group(1)):
                raise ValueError(f"invalid CHANNELS line in {path}: {stripped}")
            channels[active] = joint_channels

    if motion_line is None:
        raise ValueError(f"missing MOTION section in {path}")
    if stack:
        raise ValueError(f"unbalanced BVH hierarchy in {path}")

    frames_match = re.fullmatch(r"Frames:\s*(\d+)", lines[motion_line + 1].strip())
    if frames_match is None:
        raise ValueError(f"missing frame count in {path}")
    frame_count = int(frames_match.group(1))
    frame_lines = lines[motion_line + 3 : motion_line + 3 + frame_count]
    values = np.asarray(
        [[float(value) for value in line.split()] for line in frame_lines],
        dtype=float,
    )
    expected_channels = sum(len(joint_channels) for joint_channels in channels)
    if values.shape != (frame_count, expected_channels):
        raise ValueError(
            f"motion data in {path} has shape {values.shape}; "
            f"expected {(frame_count, expected_channels)}"
        )

    positions = np.broadcast_to(
        np.asarray(offsets, dtype=float), (frame_count, len(names), 3)
    ).copy()
    quaternions = np.zeros((frame_count, len(names), 4), dtype=float)
    cursor = 0
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    for joint_index, joint_channels in enumerate(channels):
        rotation_channels = [
            channel for channel in joint_channels if channel.endswith("rotation")
        ]
        rotation_values = []
        for channel in joint_channels:
            channel_values = values[:, cursor]
            cursor += 1
            axis = axis_index[channel[0]]
            if channel.endswith("position"):
                # Match established BVH loaders: animated translations replace
                # the static OFFSET along that axis.
                positions[:, joint_index, axis] = channel_values
            elif channel.endswith("rotation"):
                rotation_values.append(channel_values)
            else:
                raise ValueError(f"unsupported BVH channel {channel!r} in {path}")
        if len(rotation_channels) != 3:
            raise ValueError(
                f"joint {names[joint_index]} must have three rotation channels"
            )
        order = "".join(channel[0].lower() for channel in rotation_channels)
        euler = np.stack(rotation_values, axis=-1)
        quaternions[:, joint_index] = utils.euler_to_quat(
            np.radians(euler), order=order
        )

    quaternions = utils.remove_quat_discontinuities(quaternions)
    return SomaAnimation(
        quaternions=quaternions,
        positions=positions,
        parents=np.asarray(parents, dtype=int),
        names=tuple(names),
    )


def convert_soma_bvh_to_smplx(path: Path, body_model_dir: Path) -> tuple[dict, dict]:
    """Convert a SOMA BVH body trajectory to SMPL-X body parameters."""

    path = path.resolve()
    animation = read_soma_bvh(path)
    bone_index = {name: index for index, name in enumerate(animation.names)}
    missing = set(SMPLX_JOINT_TO_SOMA.values()) - bone_index.keys()
    if missing:
        raise ValueError(f"{path} is missing SOMA bones: {sorted(missing)}")

    mapped_indices = [bone_index[name] for name in SMPLX_JOINT_TO_SOMA.values()]
    _, global_positions = utils.quat_fk(
        animation.quaternions, animation.positions, animation.parents
    )
    mapped_positions_y_up = global_positions[:, mapped_indices] / 100.0
    mapped_positions_z_up = Y_UP_TO_Z_UP.apply(mapped_positions_y_up)
    rest_joints, parents = load_smplx_rest_skeleton(body_model_dir.resolve())
    root_orient, pose_body = solve_minimal_twist_smpl_rotations(
        mapped_positions_z_up, rest_joints, parents
    )

    root_position = mapped_positions_z_up[:, 0].copy()
    root_position[:, :2] -= root_position[0, :2]
    fps = read_fps(path)
    data = {
        "gender": np.asarray("neutral"),
        "mocap_frame_rate": np.asarray(fps),
        "betas": np.zeros(16),
        "root_orient": root_orient,
        "pose_body": pose_body.reshape(len(animation.quaternions), -1),
        "trans": root_position - rest_joints[0],
        "orientation_method": np.asarray("minimal_twist_skeletal_ik_v1"),
    }
    metadata = {
        "source_file": path.name,
        "source_sha256": sha256(path),
        "frames": len(animation.quaternions),
        "fps": fps,
        "duration_seconds": len(animation.quaternions) / fps,
        "orientation_method": "minimal_twist_skeletal_ik_v1",
    }
    return data, metadata
