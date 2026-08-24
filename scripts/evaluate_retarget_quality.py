"""Evaluate jitter, tracking, collision, limits, and foot contact quality."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

from general_motion_retargeting import (
    GeneralMotionRetargeting,
    evaluate_retargeted_motion,
)
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.quality import joint_jitter_metrics
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplx-motion", type=Path, required=True)
    parser.add_argument("--robot-motion", type=Path, required=True)
    parser.add_argument("--body-model-dir", type=Path, required=True)
    parser.add_argument("--robot", default="bello")
    parser.add_argument("--task-profile", default=None)
    parser.add_argument("--reference-motion", type=Path, default=None)
    parser.add_argument("--reference-robot", default="unitree_g1")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the robot profile's quality gate fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.robot_motion, allow_pickle=False) as motion:
        qpos = np.asarray(motion["qpos"], dtype=float)
        timestamps = np.asarray(motion["timestamp_seconds"], dtype=float)
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("robot timestamps must be strictly increasing")
    fps = 1.0 / float(np.median(np.diff(timestamps)))

    with torch.inference_mode():
        smplx_data, body_model, smplx_output, human_height = load_smplx_file(
            args.smplx_motion, args.body_model_dir
        )
        human_frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=fps,
        )
    del body_model, smplx_output
    gc.collect()
    if not np.isclose(fps, aligned_fps):
        raise ValueError(f"frame-rate mismatch: robot={fps}, human={aligned_fps}")
    if len(qpos) > len(human_frames):
        raise ValueError(
            f"robot trajectory is longer than its source: "
            f"robot={len(qpos)}, human={len(human_frames)}"
        )
    human_frames = human_frames[: len(qpos)]

    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=float(human_height),
        verbose=False,
        task_profile=args.task_profile,
    )
    report = evaluate_retargeted_motion(qpos, human_frames, retargeter, fps)
    if args.reference_motion is not None:
        with np.load(args.reference_motion, allow_pickle=False) as motion:
            reference_qpos = np.asarray(motion["qpos"], dtype=float)
        import mujoco

        reference_model = mujoco.MjModel.from_xml_path(
            str(ROBOT_XML_DICT[args.reference_robot])
        )
        reference_jitter = joint_jitter_metrics(reference_qpos, reference_model)
        report["reference_robot"] = args.reference_robot
        report["reference_jitter"] = reference_jitter
        report["jitter_rms_ratio"] = (
            report["model"]["jitter"]["rms_degrees"]
            / reference_jitter["rms_degrees"]
        )
    text = json.dumps(report, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    if args.strict and not report["quality_gate"]["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
