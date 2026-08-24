"""Retarget one SMPL-X motion without launching an interactive viewer."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from general_motion_retargeting import retarget_offline_frames
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplx-motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="bello")
    parser.add_argument(
        "--body-model-dir", type=Path, default=Path("assets/body_models")
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--initial-settle-passes",
        type=int,
        default=None,
        help="override the robot profile's first-frame settle passes",
    )
    parser.add_argument(
        "--passes-per-frame",
        type=int,
        default=None,
        help="override the robot profile's IK passes after the first frame",
    )
    parser.add_argument(
        "--use-velocity-limit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the robot profile's solver velocity-limit setting",
    )
    parser.add_argument(
        "--task-profile",
        default=None,
        help="override the robot's named task profile",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError("target frame rate must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0.0:
        raise ValueError("maximum duration must be positive")

    with torch.inference_mode():
        smplx_data, body_model, smplx_output, human_height = load_smplx_file(
            args.smplx_motion, args.body_model_dir
        )
        human_frames, fps = get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=args.fps,
        )
    del body_model, smplx_output
    gc.collect()

    if args.max_seconds is not None:
        human_frames = human_frames[: round(args.max_seconds * fps)]
    result = retarget_offline_frames(
        human_frames,
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=float(human_height),
        task_profile=args.task_profile,
        use_velocity_limit=args.use_velocity_limit,
        passes_per_frame=args.passes_per_frame,
        initial_settle_passes=args.initial_settle_passes,
    )
    trajectory = result.qpos

    timestamps = np.arange(len(trajectory), dtype=float) / fps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=trajectory,
        timestamp_seconds=timestamps,
        source_motion=np.asarray(str(args.smplx_motion)),
        use_velocity_limit=np.asarray(result.use_velocity_limit),
        passes_per_frame=np.asarray(result.passes_per_frame),
        initial_settle_passes=np.asarray(result.initial_settle_passes),
        joint_smoothing_kernel=result.joint_smoothing_kernel,
        root_orientation_smoothing_kernel=(
            result.root_orientation_smoothing_kernel
        ),
        joint_smoothing_passes=np.asarray(result.joint_smoothing_passes),
        smoothing_alpha=result.smoothing_alpha,
        task_profile=np.asarray(result.retargeter.task_profile or ""),
    )
    print(
        f"{args.output.resolve()} frames={len(trajectory)} "
        f"duration={len(trajectory) / fps:.3f}s"
    )


if __name__ == "__main__":
    main()
