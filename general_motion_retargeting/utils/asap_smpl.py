"""Validate ASAP's AMASS-style SMPL clips for GMR's SMPL-X body loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {"betas", "gender", "mocap_framerate", "poses", "trans"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_asap_smpl_to_smplx(path: Path) -> tuple[dict, dict]:
    """Map one ASAP SMPL clip onto the shared 22-joint SMPL-X body subset.

    SMPL and SMPL-X use the same ordering for the pelvis and first 21 body
    joints. ASAP's final two SMPL hand joints are intentionally omitted because
    GMR targets the wrists, not articulated hands.
    """
    path = path.resolve()
    with np.load(path, allow_pickle=False) as source:
        missing = REQUIRED_KEYS - set(source.files)
        if missing:
            raise ValueError(f"{path} is missing ASAP SMPL keys: {sorted(missing)}")

        poses = np.asarray(source["poses"], dtype=np.float64)
        translation = np.asarray(source["trans"], dtype=np.float64)
        betas = np.asarray(source["betas"], dtype=np.float64).reshape(-1)
        gender_value = np.asarray(source["gender"])
        frame_rate_value = np.asarray(source["mocap_framerate"])

    if poses.ndim != 2 or poses.shape[1] < 66:
        raise ValueError(
            f"invalid ASAP SMPL poses shape in {path}: {poses.shape}; "
            "expected (frames, at least 66)"
        )
    frame_count = poses.shape[0]
    if frame_count < 2:
        raise ValueError(f"ASAP SMPL clip must contain at least two frames: {path}")
    if translation.shape != (frame_count, 3):
        raise ValueError(
            f"invalid ASAP SMPL translation shape in {path}: {translation.shape}"
        )
    if betas.shape == (10,):
        betas = np.pad(betas, (0, 6))
    elif betas.shape != (16,):
        raise ValueError(
            f"invalid ASAP SMPL betas shape in {path}: {betas.shape}; "
            "expected 10 or 16 coefficients"
        )
    if gender_value.shape != ():
        raise ValueError(f"ASAP SMPL gender must be scalar in {path}")
    gender = str(gender_value.item()).lower()
    if gender not in {"female", "male", "neutral"}:
        raise ValueError(f"unsupported ASAP SMPL gender {gender!r} in {path}")
    if frame_rate_value.shape != ():
        raise ValueError(f"ASAP SMPL mocap_framerate must be scalar in {path}")
    frame_rate = float(frame_rate_value.item())
    if not np.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError(f"invalid ASAP SMPL frame rate {frame_rate} in {path}")
    if not all(np.all(np.isfinite(values)) for values in (poses, translation, betas)):
        raise ValueError(f"ASAP SMPL clip contains nonfinite values: {path}")

    translation = translation.copy()
    translation[:, :2] -= translation[0, :2]
    source_sha256 = sha256(path)
    data = {
        "gender": np.asarray(gender),
        "mocap_frame_rate": np.asarray(frame_rate),
        "betas": betas,
        "root_orient": poses[:, :3],
        "pose_body": poses[:, 3:66],
        "trans": translation,
        "source_format": np.asarray("asap_amass_smpl_v1"),
        "source_sha256": np.asarray(source_sha256),
    }
    metadata = {
        "source_file": path.name,
        "source_sha256": source_sha256,
        "frames": frame_count,
        "fps": frame_rate,
        "duration_seconds": frame_count / frame_rate,
        "source_format": "asap_amass_smpl_v1",
    }
    return data, metadata
