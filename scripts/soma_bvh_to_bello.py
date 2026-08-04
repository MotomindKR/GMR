"""Retarget a SOMA BVH clip to fixed-head Bello and optionally render it."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import imageio.v2 as imageio
import mujoco
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1_smplx import load_smplx_joint_frames
from general_motion_retargeting.utils.soma_bvh import convert_soma_bvh_to_smplx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-file", type=Path, required=True)
    parser.add_argument("--motion-output", type=Path, required=True)
    parser.add_argument("--smplx-output", type=Path)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def save_motion(path: Path, trajectory: np.ndarray, fps: float, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    motion = {
        "fps": float(fps),
        "root_pos": trajectory[:, :3],
        "root_rot": trajectory[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": trajectory[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
        "source_file": metadata["source_file"],
        "source_sha256": metadata["source_sha256"],
        "robot": "bello",
        "robot_configuration": "sphere_hands_waist_yaw_only_fixed_head",
    }
    with path.open("wb") as output:
        pickle.dump(motion, output)


def world_camera(trajectory: np.ndarray) -> mujoco.MjvCamera:
    root_positions = trajectory[:, :3]
    lower = root_positions.min(axis=0)
    upper = root_positions.max(axis=0)
    travel_width = upper[1] - lower[1]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = 0.5 * (lower + upper)
    cam.lookat[2] = np.mean(root_positions[:, 2]) + 0.18
    cam.distance = max(3.0, (travel_width + 1.5) / 1.45)
    cam.azimuth = 0.0
    cam.elevation = -8.0
    return cam


def add_geom(scene, geom_type, size, position, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("render scene exceeded its geometry budget")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def render_video(
    path: Path, trajectory: np.ndarray, fps: float, width: int, height: int
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["bello"]))
    model.vis.headlight.ambient[:] = 0.42
    model.vis.headlight.diffuse[:] = 0.82
    model.vis.headlight.specular[:] = 0.25
    model.vis.quality.offsamples = 4
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    sphere_bodies = (
        model.body("l_end_effector_sphere_link").id,
        model.body("r_end_effector_sphere_link").id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart"],
    )
    fixed_camera = world_camera(trajectory)
    try:
        for qpos in trajectory:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=fixed_camera)
            add_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_PLANE,
                (25.0, 25.0, 0.05),
                (qpos[0], qpos[1], -0.006),
                (0.19, 0.22, 0.25, 1.0),
            )
            for body_id, color in zip(
                sphere_bodies,
                ((0.95, 0.36, 0.18, 1.0), (0.16, 0.55, 0.96, 1.0)),
                strict=True,
            ):
                add_geom(
                    renderer.scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    (0.047, 0.047, 0.047),
                    data.xpos[body_id],
                    color,
                )
            writer.append_data(renderer.render().copy())
    finally:
        writer.close()
        renderer.close()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    body_model_dir = root / "assets" / "body_models"
    smplx_output = args.smplx_output or args.motion_output.with_suffix(".smplx.npz")
    smplx_output.parent.mkdir(parents=True, exist_ok=True)

    smplx_data, metadata = convert_soma_bvh_to_smplx(
        args.bvh_file, body_model_dir
    )
    np.savez_compressed(smplx_output, **smplx_data)
    frames, fps = load_smplx_joint_frames(smplx_output, body_model_dir)
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
    trajectory = np.asarray([retargeter.retarget(frame) for frame in frames])
    if trajectory.shape != (len(frames), 32):
        raise ValueError(
            f"Bello returned trajectory shape {trajectory.shape}; "
            f"expected {(len(frames), 32)}"
        )
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("Bello trajectory contains non-finite values")
    save_motion(args.motion_output, trajectory, fps, metadata)
    print(f"Wrote {len(trajectory)} frames to {args.motion_output}")
    if args.video_output is not None:
        render_video(args.video_output, trajectory, fps, args.width, args.height)
        print(f"Wrote {args.width}x{args.height} video to {args.video_output}")


if __name__ == "__main__":
    main()
