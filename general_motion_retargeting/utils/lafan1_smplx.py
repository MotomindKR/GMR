"""Convert LAFAN1 BVH skeletons into the SMPL-X convention used by GMR."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation as R
import smplx
import torch

from .lafan_vendor.extract import read_bvh
from .lafan_vendor.utils import quat_fk


SMPLX_JOINT_TO_LAFAN1 = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "spine1": "Spine",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "spine2": "Spine1",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Spine2",
    "left_foot": "LeftToe",
    "right_foot": "RightToe",
    "neck": "Neck",
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
Y_UP_TO_Z_UP = R.from_matrix(
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fps(path: Path) -> float:
    with path.open(errors="strict") as source:
        for line in source:
            match = re.match(r"\s*Frame Time:\s*([\d.]+)", line)
            if match is not None:
                return 1.0 / float(match.group(1))
    raise ValueError(f"missing Frame Time in {path}")


@lru_cache(maxsize=None)
def load_smplx_rest_skeleton(body_model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    model = smplx.create(body_model_dir, "smplx", gender="neutral", use_pca=False)
    output = model(
        betas=torch.zeros(1, 16),
        global_orient=torch.zeros(1, 3),
        body_pose=torch.zeros(1, 63),
        transl=torch.zeros(1, 3),
        left_hand_pose=torch.zeros(1, 45),
        right_hand_pose=torch.zeros(1, 45),
        jaw_pose=torch.zeros(1, 3),
        leye_pose=torch.zeros(1, 3),
        reye_pose=torch.zeros(1, 3),
    )
    joint_count = len(SMPLX_JOINT_TO_LAFAN1)
    joints = output.joints[0, :joint_count].detach().numpy()
    parents = model.parents[:joint_count].detach().numpy()
    return joints, parents


def normalized(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-8):
        raise ValueError("cannot derive an orientation from a zero-length bone")
    return vectors / lengths


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> R:
    source, target = normalized(np.stack((source, target)))
    cross = np.cross(source, target)
    cross_norm = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cross_norm > 1e-8:
        return R.from_rotvec(cross / cross_norm * np.arctan2(cross_norm, cosine))
    if cosine > 0.0:
        return R.identity()

    fallback_axis = np.eye(3)[np.argmin(np.abs(source))]
    rotation_axis = normalized(np.cross(source, fallback_axis)[None])[0]
    return R.from_rotvec(np.pi * rotation_axis)


def solve_minimal_twist_smpl_rotations(
    target_positions: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit bone directions while choosing zero unobservable axial twist."""
    target_positions = np.asarray(target_positions, dtype=float)
    rest_joints = np.asarray(rest_joints, dtype=float)
    parents = np.asarray(parents, dtype=int)
    if target_positions.ndim != 3 or target_positions.shape[2] != 3:
        raise ValueError("target positions must have shape (frames, joints, 3)")
    if target_positions.shape[1:] != rest_joints.shape:
        raise ValueError("target and rest skeletons must contain the same joints")
    if len(parents) != len(rest_joints) or parents[0] != -1:
        raise ValueError("invalid SMPL-X parent hierarchy")

    children = [np.flatnonzero(parents == joint) for joint in range(len(parents))]
    root_orientations = []
    body_poses = []
    for frame_positions in target_positions:
        global_rotations: list[R | None] = [None] * len(parents)
        local_rotations = []
        for joint, joint_children in enumerate(children):
            reference = R.identity() if joint == 0 else global_rotations[parents[joint]]
            assert reference is not None
            if len(joint_children) == 0:
                global_rotation = reference
            else:
                rest_vectors = normalized(
                    rest_joints[joint_children] - rest_joints[joint]
                )
                target_vectors = normalized(
                    frame_positions[joint_children] - frame_positions[joint]
                )
                reference_vectors = reference.apply(rest_vectors)
                if len(joint_children) == 1:
                    correction = rotation_between_vectors(
                        reference_vectors[0], target_vectors[0]
                    )
                else:
                    correction, _ = R.align_vectors(target_vectors, reference_vectors)
                global_rotation = correction * reference

            global_rotations[joint] = global_rotation
            if joint == 0:
                root_orientations.append(global_rotation.as_rotvec())
            else:
                parent_rotation = global_rotations[parents[joint]]
                assert parent_rotation is not None
                local_rotations.append(
                    (parent_rotation.inv() * global_rotation).as_rotvec()
                )
        body_poses.append(local_rotations)

    frame_count = len(target_positions)
    return (
        np.asarray(root_orientations).reshape(frame_count, 3),
        np.asarray(body_poses).reshape(frame_count, len(parents) - 1, 3),
    )


def convert_bvh_to_smplx(path: Path, body_model_dir: Path) -> tuple[dict, dict]:
    """Convert every frame of one LAFAN1 BVH file to SMPL-X parameters."""
    path = path.resolve()
    animation = read_bvh(path)
    bone_index = {name: index for index, name in enumerate(animation.bones)}
    missing = set(SMPLX_JOINT_TO_LAFAN1.values()) - bone_index.keys()
    if missing:
        raise ValueError(f"{path} is missing LAFAN1 bones: {sorted(missing)}")
    mapped_indices = [bone_index[name] for name in SMPLX_JOINT_TO_LAFAN1.values()]
    _, global_positions = quat_fk(animation.quats, animation.pos, animation.parents)
    mapped_positions_y_up = global_positions[:, mapped_indices] / 100.0
    mapped_positions_z_up = Y_UP_TO_Z_UP.apply(mapped_positions_y_up)
    rest_joints, parents = load_smplx_rest_skeleton(body_model_dir.resolve())
    root_orient, pose_body = solve_minimal_twist_smpl_rotations(
        mapped_positions_z_up, rest_joints, parents
    )

    root_position = mapped_positions_z_up[:, 0].copy()
    root_position[:, :2] -= root_position[0, :2]
    translation = root_position - rest_joints[0]
    fps = read_fps(path)
    data = {
        "gender": np.asarray("neutral"),
        "mocap_frame_rate": np.asarray(fps),
        "betas": np.zeros(16),
        "root_orient": root_orient,
        "pose_body": pose_body.reshape(len(animation.quats), -1),
        "trans": translation,
        "orientation_method": np.asarray("minimal_twist_skeletal_ik_v1"),
    }
    metadata = {
        "source_file": path.name,
        "source_sha256": sha256(path),
        "frames": len(animation.quats),
        "fps": fps,
        "duration_seconds": len(animation.quats) / fps,
        "orientation_method": "minimal_twist_skeletal_ik_v1",
    }
    return data, metadata


def load_smplx_joint_frames(
    path: Path, body_model_dir: Path
) -> tuple[list[dict[str, tuple[np.ndarray, np.ndarray]]], float]:
    """Load the 22 body-joint transforms without constructing an SMPL-X mesh."""
    with np.load(path) as data:
        root_orient = np.asarray(data["root_orient"], dtype=float)
        body_pose = np.asarray(data["pose_body"], dtype=float).reshape(-1, 21, 3)
        translation = np.asarray(data["trans"], dtype=float)
        fps = float(data["mocap_frame_rate"])
    frame_count = len(root_orient)
    if body_pose.shape != (frame_count, 21, 3):
        raise ValueError(f"invalid SMPL-X body pose shape in {path}: {body_pose.shape}")
    if translation.shape != (frame_count, 3):
        raise ValueError(
            f"invalid SMPL-X translation shape in {path}: {translation.shape}"
        )

    rest_joints, parents = load_smplx_rest_skeleton(body_model_dir.resolve())
    joint_count = len(rest_joints)
    global_rotations: list[R] = [R.identity()] * joint_count
    positions = np.empty((frame_count, joint_count, 3), dtype=float)
    global_rotations[0] = R.from_rotvec(root_orient)
    positions[:, 0] = translation + rest_joints[0]
    for joint in range(1, joint_count):
        parent = int(parents[joint])
        parent_rotation = global_rotations[parent]
        positions[:, joint] = positions[:, parent] + parent_rotation.apply(
            np.broadcast_to(
                rest_joints[joint] - rest_joints[parent], (frame_count, 3)
            )
        )
        global_rotations[joint] = parent_rotation * R.from_rotvec(
            body_pose[:, joint - 1]
        )

    names = tuple(SMPLX_JOINT_TO_LAFAN1)
    quaternions = [
        rotation.as_quat(scalar_first=True) for rotation in global_rotations
    ]
    frames = [
        {
            name: (positions[frame, joint], quaternions[joint][frame])
            for joint, name in enumerate(names)
        }
        for frame in range(frame_count)
    ]
    return frames, fps
