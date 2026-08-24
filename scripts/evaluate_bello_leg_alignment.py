"""Measure Bello knee-frame and toe-heading alignment for one SMPL-X clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplx-motion", type=Path, required=True)
    parser.add_argument("--bello-motion", type=Path, required=True)
    parser.add_argument(
        "--body-model-dir", type=Path, default=Path("assets/body_models")
    )
    return parser.parse_args()


def heading_error_degrees(actual: np.ndarray, target: np.ndarray) -> float:
    actual_xy = actual[:2]
    target_xy = target[:2]
    actual_xy /= np.linalg.norm(actual_xy)
    target_xy /= np.linalg.norm(target_xy)
    cross = actual_xy[0] * target_xy[1] - actual_xy[1] * target_xy[0]
    return abs(float(np.rad2deg(np.arctan2(cross, np.dot(actual_xy, target_xy)))))


def percentiles(values: list[float]) -> str:
    result = np.percentile(values, [50, 95, 100])
    return "/".join(f"{value:.2f}" for value in result)


def main() -> None:
    args = parse_args()
    with torch.inference_mode():
        smplx_data, body_model, smplx_output, _ = load_smplx_file(
            args.smplx_motion, args.body_model_dir
        )
        human_frames, fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30.0
        )
    with np.load(args.bello_motion, allow_pickle=False) as motion:
        trajectory = np.asarray(motion["qpos"], dtype=float)
    if len(human_frames) != len(trajectory):
        raise ValueError(
            f"frame-count mismatch: human={len(human_frames)}, Bello={len(trajectory)}"
        )

    config = json.loads(Path(IK_CONFIG_DICT["smplx"]["bello"]).read_text())
    table = config["ik_match_table1"]
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["bello"]))
    data = mujoco.MjData(model)

    print(f"{args.smplx_motion.name} frames={len(trajectory)} fps={fps:g}")
    for side in ("left", "right"):
        knee_name = f"{side}_knee_link"
        foot_name = f"{side}_ankle_roll_link"
        knee_offset = Rotation.from_quat(table[knee_name][4], scalar_first=True)
        foot_offset = Rotation.from_quat(table[foot_name][4], scalar_first=True)
        knee_id = model.body(knee_name).id
        foot_id = model.body(foot_name).id

        knee_full = []
        knee_axial = []
        toe_heading = []
        toe_elevation_error = []
        for human_frame, qpos in zip(human_frames, trajectory, strict=True):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            actual_knee = Rotation.from_matrix(data.xmat[knee_id].reshape(3, 3))
            target_knee = (
                Rotation.from_quat(human_frame[f"{side}_knee"][1], scalar_first=True)
                * knee_offset
            )
            knee_error = (actual_knee.inv() * target_knee).as_rotvec()
            knee_full.append(float(np.rad2deg(np.linalg.norm(knee_error))))
            knee_axial.append(abs(float(np.rad2deg(knee_error[0]))))

            actual_foot = Rotation.from_matrix(data.xmat[foot_id].reshape(3, 3))
            target_foot = (
                Rotation.from_quat(human_frame[f"{side}_foot"][1], scalar_first=True)
                * foot_offset
            )
            actual_toe = actual_foot.apply(np.array([0.0, 0.0, 1.0]))
            target_toe = target_foot.apply(np.array([0.0, 0.0, 1.0]))
            toe_heading.append(heading_error_degrees(actual_toe, target_toe))
            toe_elevation_error.append(
                abs(
                    float(
                        np.rad2deg(
                            np.arcsin(np.clip(actual_toe[2], -1.0, 1.0))
                            - np.arcsin(np.clip(target_toe[2], -1.0, 1.0))
                        )
                    )
                )
            )
        print(
            f"{side}: median/p95/max deg | knee_full "
            f"{percentiles(knee_full)} | knee_axial {percentiles(knee_axial)} "
            f"| toe_heading {percentiles(toe_heading)} "
            f"| toe_elevation {percentiles(toe_elevation_error)}"
        )


if __name__ == "__main__":
    main()
