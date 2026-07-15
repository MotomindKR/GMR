"""Extract LAFAN1 and retarget every sequence to the clean Bello model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import tempfile
import zipfile

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.lafan1_smplx import (
    convert_bvh_to_smplx,
    load_smplx_joint_frames,
    sha256,
)


DATASET_NAME = "Ubisoft La Forge Animation Dataset (LAFAN1)"
SOURCE_URL = (
    "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/"
    "raw/master/lafan1/lafan1.zip"
)
SOURCE_LICENSE = (
    "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/"
    "blob/master/license.txt"
)


def extract_bvh_archive(archive: Path, output_dir: Path) -> list[Path]:
    archive = archive.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as source:
        members = sorted(
            (
                member
                for member in source.infolist()
                if member.filename.lower().endswith(".bvh")
            ),
            key=lambda member: member.filename,
        )
        for member in members:
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
            destination = output_dir / member_path.name
            if not destination.exists() or destination.stat().st_size != member.file_size:
                with source.open(member) as input_file, destination.open("wb") as output_file:
                    while chunk := input_file.read(1024 * 1024):
                        output_file.write(chunk)
            extracted.append(destination)
    if not extracted:
        raise ValueError(f"no BVH files found in {archive}")
    if len({path.name for path in extracted}) != len(extracted):
        raise ValueError("archive contains duplicate BVH basenames")
    return extracted


def atomic_savez(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        np.savez_compressed(temporary, **data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_pickle(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pkl", delete=False) as tmp:
        temporary = Path(tmp.name)
        pickle.dump(data, tmp)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def retarget_smplx_to_bello(
    source: Path, output: Path, body_model_dir: Path
) -> dict:
    frames, fps = load_smplx_joint_frames(source, body_model_dir)
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
        use_velocity_limit=True,
        source_fps=fps,
    )
    previous = retargeter.retarget(frames[0])
    for _ in range(9):
        current = retargeter.retarget(frames[0])
        if np.max(np.abs(current - previous)) < 1e-4:
            break
        previous = current
    qpos = np.asarray([retargeter.retarget(frame) for frame in frames])
    motion = {
        "fps": float(fps),
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    atomic_pickle(output, motion)
    return {
        "frames": len(qpos),
        "fps": float(fps),
        "duration_seconds": len(qpos) / float(fps),
        "gmr_file": output.name,
        "gmr_sha256": sha256(output),
    }


def process_sequence(
    source: str,
    smplx_dir: str,
    output_dir: str,
    body_model_dir: str,
    overwrite: bool,
) -> dict:
    source_path = Path(source)
    smplx_path = Path(smplx_dir) / f"{source_path.stem}.npz"
    output_path = Path(output_dir) / f"{source_path.stem}.pkl"
    if overwrite or not smplx_path.exists():
        data, metadata = convert_bvh_to_smplx(source_path, Path(body_model_dir))
        atomic_savez(smplx_path, data)
    else:
        with np.load(smplx_path) as existing:
            frames = int(existing["pose_body"].shape[0])
            fps = float(existing["mocap_frame_rate"])
        metadata = {
            "source_file": source_path.name,
            "source_sha256": sha256(source_path),
            "frames": frames,
            "fps": fps,
            "duration_seconds": frames / fps,
            "orientation_method": "minimal_twist_skeletal_ik_v1",
        }
    metadata.update(
        {"smplx_file": smplx_path.name, "smplx_sha256": sha256(smplx_path)}
    )
    if overwrite or not output_path.exists():
        metadata.update(
            retarget_smplx_to_bello(smplx_path, output_path, Path(body_model_dir))
        )
    else:
        with output_path.open("rb") as motion_file:
            motion = pickle.load(motion_file)
        frames = len(motion["root_pos"])
        fps = float(motion["fps"])
        metadata.update(
            {
                "frames": frames,
                "fps": fps,
                "duration_seconds": frames / fps,
                "gmr_file": output_path.name,
                "gmr_sha256": sha256(output_path),
            }
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("motion_data/sources/lafan1.zip")
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path("motion_data/sources/lafan1")
    )
    parser.add_argument(
        "--smplx-dir", type=Path, default=Path("motion_data/lafan1_smplx")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("motion_data/lafan1_bello")
    )
    parser.add_argument(
        "--body-model-dir", type=Path, default=Path("assets/body_models")
    )
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    sources = extract_bvh_archive(args.archive, args.source_dir)
    if args.limit is not None:
        sources = sources[: args.limit]
    args.smplx_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motions = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_sequence,
                str(source.resolve()),
                str(args.smplx_dir.resolve()),
                str(args.output_dir.resolve()),
                str(args.body_model_dir.resolve()),
                args.overwrite,
            ): source
            for source in sources
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            try:
                metadata = future.result()
                motions.append(metadata)
                print(f"[{completed}/{len(futures)}] {source.stem}", flush=True)
            except Exception as error:
                failures.append({"source_file": source.name, "error": repr(error)})
                print(
                    f"[{completed}/{len(futures)}] FAILED {source.stem}: {error}",
                    flush=True,
                )

    manifest = {
        "source_dataset": DATASET_NAME,
        "source_url": SOURCE_URL,
        "source_license": SOURCE_LICENSE,
        "archive_sha256": sha256(args.archive),
        "robot": "bello",
        "robot_configuration": "fixed_waist_roll_pitch",
        "motions": sorted(motions, key=lambda item: item["source_file"]),
        "failures": sorted(failures, key=lambda item: item["source_file"]),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path} with {len(motions)} motions")
    if failures:
        raise SystemExit(f"{len(failures)} sequence(s) failed; see {manifest_path}")


if __name__ == "__main__":
    main()
