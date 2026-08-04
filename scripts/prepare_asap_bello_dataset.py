"""Convert ASAP SMPL clips and retarget them to fixed-head Bello."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import pickle
import subprocess
import tempfile

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.asap_smpl import (
    convert_asap_smpl_to_smplx,
    sha256,
)
from general_motion_retargeting.utils.lafan1_smplx import load_smplx_joint_frames


SOURCE_REPOSITORY = "https://github.com/LeCAR-Lab/ASAP"


def atomic_savez(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as tmp:
        temporary = Path(tmp.name)
    try:
        np.savez_compressed(temporary, **data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_pickle(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".pkl", delete=False
    ) as tmp:
        temporary = Path(tmp.name)
        pickle.dump(data, tmp)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def retarget_smplx_to_bello(source: Path, output: Path, body_model_dir: Path) -> dict:
    frames, fps = load_smplx_joint_frames(source, body_model_dir)
    with np.load(source, allow_pickle=False) as source_data:
        source_sha256 = str(np.asarray(source_data["source_sha256"]).item())
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
    if qpos.shape != (len(frames), 32):
        raise ValueError(
            f"fixed-head Bello returned qpos shape {qpos.shape}; "
            f"expected {(len(frames), 32)}"
        )
    motion = {
        "fps": float(fps),
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
        "source_sha256": source_sha256,
    }
    atomic_pickle(output, motion)
    return {
        "frames": len(qpos),
        "fps": float(fps),
        "duration_seconds": len(qpos) / float(fps),
        "gmr_file": output.name,
        "gmr_sha256": sha256(output),
        "gmr_dof_count": qpos.shape[1] - 7,
    }


def process_sequence(
    source: str,
    smplx_dir: str,
    output_dir: str,
    body_model_dir: str,
    overwrite: bool,
) -> dict:
    source_path = Path(source)
    smplx_path = Path(smplx_dir) / source_path.name
    output_path = Path(output_dir) / f"{source_path.stem}.pkl"
    smplx_data, metadata = convert_asap_smpl_to_smplx(source_path)
    if overwrite or not smplx_path.exists():
        atomic_savez(smplx_path, smplx_data)
    else:
        with np.load(smplx_path, allow_pickle=False) as existing:
            existing_sha256 = str(np.asarray(existing["source_sha256"]).item())
        if existing_sha256 != metadata["source_sha256"]:
            raise ValueError(
                f"stale SMPL-X output {smplx_path}; rerun with --overwrite"
            )
    metadata.update({"smplx_file": smplx_path.name, "smplx_sha256": sha256(smplx_path)})
    if overwrite or not output_path.exists():
        metadata.update(
            retarget_smplx_to_bello(smplx_path, output_path, Path(body_model_dir))
        )
    else:
        with output_path.open("rb") as motion_file:
            motion = pickle.load(motion_file)  # noqa: S301 - trusted local output
        frames = len(motion["root_pos"])
        fps = float(motion["fps"])
        dof_count = int(np.asarray(motion["dof_pos"]).shape[1])
        source_sha256 = motion.get("source_sha256")
        if (
            frames != metadata["frames"]
            or dof_count != 25
            or source_sha256 != metadata["source_sha256"]
        ):
            raise ValueError(
                f"stale Bello output {output_path}; rerun with --overwrite"
            )
        metadata.update(
            {
                "frames": frames,
                "fps": fps,
                "duration_seconds": frames / fps,
                "gmr_file": output_path.name,
                "gmr_sha256": sha256(output_path),
                "gmr_dof_count": dof_count,
            }
        )
    return metadata


def git_revision(repository: Path) -> str | None:
    if not (repository / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo-dir", type=Path, default=Path("motion_data/sources/asap")
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--smplx-dir", type=Path, default=Path("motion_data/asap_smplx")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("motion_data/asap_bello_fixed_head"),
    )
    parser.add_argument(
        "--body-model-dir", type=Path, default=Path("assets/body_models")
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    source_dir = args.source_dir or (
        args.source_repo_dir / "humanoidverse/data/motions/raw_tairantestbed_smpl"
    )
    sources = sorted(source_dir.resolve().glob("*.npz"))
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        parser.error(f"no ASAP SMPL .npz files found in {source_dir}")

    args.smplx_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motions = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_sequence,
                str(source),
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
            except Exception as error:  # noqa: BLE001 - manifest every bad clip
                failures.append({"source_file": source.name, "error": repr(error)})
                print(
                    f"[{completed}/{len(futures)}] FAILED {source.stem}: {error}",
                    flush=True,
                )

    manifest = {
        "source_dataset": "ASAP TairanTestbed motions",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": git_revision(args.source_repo_dir.resolve()),
        "robot": "bello",
        "robot_configuration": "fixed_waist_roll_pitch_and_head",
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
