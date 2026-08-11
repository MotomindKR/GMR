"""Render synchronized SMPL-X and retargeted Bello motion side by side."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import smplx
import torch

from general_motion_retargeting.params import ROBOT_XML_DICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplx-motion", type=Path, required=True)
    parser.add_argument("--bello-motion", type=Path, required=True)
    parser.add_argument("--video-output", type=Path, required=True)
    parser.add_argument(
        "--body-model-dir", type=Path, default=Path("assets/body_models")
    )
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--secondary-azimuth", type=float, default=None)
    parser.add_argument("--robot-xml", type=Path, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--render-fps", type=float, default=25.0)
    return parser.parse_args()


def load_human_meshes(
    path: Path, body_model_dir: Path, target_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        gender = str(np.asarray(data["gender"]).item())
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)
        body_pose = np.asarray(data["pose_body"], dtype=np.float32)
        translation = np.asarray(data["trans"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
        fps = float(data["mocap_frame_rate"])
    source_frame_count = len(root_orient)
    if body_pose.shape != (source_frame_count, 63):
        raise ValueError(f"invalid SMPL-X body pose shape: {body_pose.shape}")
    if translation.shape != (source_frame_count, 3):
        raise ValueError(f"invalid SMPL-X translation shape: {translation.shape}")
    if fps <= 0.0:
        raise ValueError("SMPL-X frame rate must be positive")
    source_times = np.arange(source_frame_count, dtype=np.float64) / fps
    target_times = np.clip(target_times, source_times[0], source_times[-1])
    root_orient = interpolate_rotvec(source_times, root_orient, target_times)
    body_pose = interpolate_rotvec(
        source_times, body_pose.reshape(source_frame_count, -1, 3), target_times
    ).reshape(len(target_times), 63)
    translation = np.stack(
        [
            np.interp(target_times, source_times, translation[:, axis])
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.float32)
    frame_count = len(target_times)

    body_model = smplx.create(
        body_model_dir,
        "smplx",
        gender=gender,
        use_pca=False,
        num_betas=len(betas),
    )
    vertices = []
    with torch.inference_mode():
        for start in range(0, frame_count, 128):
            stop = min(start + 128, frame_count)
            chunk_frames = stop - start
            repeated_betas = np.broadcast_to(
                betas, (chunk_frames, len(betas))
            ).copy()
            zero_pose = torch.zeros((chunk_frames, 3), dtype=torch.float32)
            output = body_model(
                betas=torch.from_numpy(repeated_betas),
                global_orient=torch.from_numpy(root_orient[start:stop]),
                body_pose=torch.from_numpy(body_pose[start:stop]),
                left_hand_pose=torch.zeros((chunk_frames, 45), dtype=torch.float32),
                right_hand_pose=torch.zeros((chunk_frames, 45), dtype=torch.float32),
                jaw_pose=zero_pose,
                leye_pose=zero_pose,
                reye_pose=zero_pose,
                expression=torch.zeros((chunk_frames, 10), dtype=torch.float32),
                transl=torch.from_numpy(translation[start:stop]),
            )
            vertices.append(output.vertices.detach().cpu().numpy())
    vertices = np.concatenate(vertices, axis=0)
    vertices[:, :, :2] -= translation[:, None, :2]
    vertices[:, :, 2] -= np.min(vertices[:, :, 2])
    faces = np.asarray(body_model.faces, dtype=np.int32)
    return vertices, faces


def interpolate_rotvec(
    source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape((len(source_times), -1, 3))
    result = np.empty((len(target_times), flat.shape[1], 3), dtype=np.float32)
    for joint in range(flat.shape[1]):
        rotations = Rotation.from_rotvec(flat[:, joint])
        result[:, joint] = Slerp(source_times, rotations)(target_times).as_rotvec()
    return result.reshape((len(target_times),) + values.shape[1:])


def load_bello_trajectory(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as motion:
            trajectory = np.asarray(motion["qpos"], dtype=float)
            timestamps = np.asarray(motion["timestamp_seconds"], dtype=float)
            source_timestamps = np.asarray(
                motion["source_timestamp_seconds"]
                if "source_timestamp_seconds" in motion.files
                else timestamps,
                dtype=float,
            )
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("Bello NPZ timestamps must be increasing")
        fps = 1.0 / float(np.median(np.diff(timestamps)))
        return trajectory, fps, source_timestamps
    with path.open("rb") as motion_file:
        motion = pickle.load(motion_file)  # noqa: S301 - trusted local output
    root_position = np.asarray(motion["root_pos"], dtype=float)
    root_xyzw = np.asarray(motion["root_rot"], dtype=float)
    joint_position = np.asarray(motion["dof_pos"], dtype=float)
    if root_xyzw.shape != (len(root_position), 4):
        raise ValueError(f"invalid Bello root rotation shape: {root_xyzw.shape}")
    if joint_position.shape != (len(root_position), 25):
        raise ValueError(f"invalid Bello joint position shape: {joint_position.shape}")
    trajectory = np.concatenate(
        (root_position, root_xyzw[:, [3, 0, 1, 2]], joint_position), axis=1
    )
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("Bello trajectory contains non-finite values")
    fps = float(motion["fps"])
    source_timestamps = np.arange(len(trajectory), dtype=float) / fps
    return trajectory, fps, source_timestamps


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def make_human_model(
    vertices: np.ndarray, faces: np.ndarray, width: int, height: int
) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.add_mesh(
        name="smplx",
        uservert=vertices.reshape(-1),
        userface=faces.reshape(-1),
        smoothnormal=1,
    )
    spec.worldbody.add_geom(
        name="smplx",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="smplx",
        rgba=(0.72, 0.48, 0.35, 1.0),
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(25.0, 25.0, 0.05),
        rgba=(0.19, 0.22, 0.25, 1.0),
        contype=0,
        conaffinity=0,
    )
    model = spec.compile()
    model.vis.headlight.ambient[:] = 0.42
    model.vis.headlight.diffuse[:] = 0.82
    model.vis.headlight.specular[:] = 0.25
    model.vis.quality.offsamples = 4
    return model


def make_camera(
    lookat: np.ndarray, azimuth: float, distance: float = 3.0
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = -8.0
    return camera


def add_floor(scene: mujoco.MjvScene, center: np.ndarray) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_PLANE,
        np.asarray((25.0, 25.0, 0.05)),
        np.asarray((center[0], center[1], -0.006)),
        np.eye(3).reshape(-1),
        np.asarray((0.19, 0.22, 0.25, 1.0), dtype=np.float32),
    )
    scene.ngeom += 1


def render_comparison(
    output_path: Path,
    human_vertices: np.ndarray,
    human_faces: np.ndarray,
    bello_trajectory: np.ndarray,
    fps: float,
    width: int,
    height: int,
    azimuth: float,
    secondary_azimuth: float | None,
    robot_xml: Path,
    render_fps: float,
) -> None:
    if len(human_vertices) != len(bello_trajectory):
        raise ValueError(
            f"frame-count mismatch: human={len(human_vertices)}, "
            f"Bello={len(bello_trajectory)}"
        )
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")

    human_model = make_human_model(human_vertices[0], human_faces, width, height)
    human_data = mujoco.MjData(human_model)
    human_renderer = mujoco.Renderer(human_model, height=height, width=width)
    mesh_position = human_model.mesh_pos[0].copy()
    mesh_rotation = Rotation.from_quat(human_model.mesh_quat[0], scalar_first=True)

    bello_model = mujoco.MjModel.from_xml_path(str(robot_xml))
    bello_model.vis.global_.offwidth = width
    bello_model.vis.global_.offheight = height
    bello_model.vis.headlight.ambient[:] = 0.42
    bello_model.vis.headlight.diffuse[:] = 0.82
    bello_model.vis.headlight.specular[:] = 0.25
    bello_model.vis.quality.offsamples = 4
    bello_data = mujoco.MjData(bello_model)
    bello_renderer = mujoco.Renderer(bello_model, height=height, width=width)
    torso_id = bello_model.body("torso_link").id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if render_fps <= 0.0 or render_fps > fps:
        raise ValueError("render frame rate must be positive and no greater than motion")
    frame_stride = max(1, round(fps / render_fps))
    writer = imageio.get_writer(
        output_path,
        fps=fps / frame_stride,
        codec="libx264",
        quality=9,
        macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart"],
    )
    try:
        for frame in range(0, len(bello_trajectory), frame_stride):
            vertices = human_vertices[frame]
            qpos = bello_trajectory[frame]
            human_model.mesh_vert[:] = mesh_rotation.inv().apply(
                vertices - mesh_position
            )
            human_model.mesh_normal[:] = mesh_rotation.inv().apply(
                vertex_normals(vertices, human_faces)
            )
            human_renderer._gl_context.make_current()
            mujoco.mjr_uploadMesh(human_model, human_renderer._mjr_context, 0)
            mujoco.mj_forward(human_model, human_data)
            human_lookat = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
            human_renderer.update_scene(
                human_data, camera=make_camera(human_lookat, azimuth)
            )
            human_image = human_renderer.render().copy()
            if secondary_azimuth is not None:
                human_renderer.update_scene(
                    human_data,
                    camera=make_camera(human_lookat, secondary_azimuth),
                )
                human_secondary = human_renderer.render().copy()

            bello_data.qpos[:] = qpos
            mujoco.mj_forward(bello_model, bello_data)
            bello_renderer.update_scene(
                bello_data,
                camera=make_camera(bello_data.xpos[torso_id], azimuth),
            )
            add_floor(bello_renderer.scene, qpos[:3])
            bello_image = bello_renderer.render().copy()
            primary = np.concatenate((human_image, bello_image), axis=1)
            if secondary_azimuth is None:
                writer.append_data(primary)
                continue
            bello_renderer.update_scene(
                bello_data,
                camera=make_camera(
                    bello_data.xpos[torso_id], secondary_azimuth
                ),
            )
            add_floor(bello_renderer.scene, qpos[:3])
            bello_secondary = bello_renderer.render().copy()
            secondary = np.concatenate(
                (human_secondary, bello_secondary), axis=1
            )
            writer.append_data(np.concatenate((primary, secondary), axis=0))
    finally:
        writer.close()
        human_renderer.close()
        bello_renderer.close()


def main() -> None:
    args = parse_args()
    bello_trajectory, bello_fps, source_times = load_bello_trajectory(
        args.bello_motion
    )
    if args.max_seconds is not None:
        if args.max_seconds <= 0.0:
            raise ValueError("maximum duration must be positive")
        frame_count = min(
            len(bello_trajectory), int(np.floor(args.max_seconds * bello_fps)) + 1
        )
        bello_trajectory = bello_trajectory[:frame_count]
        source_times = source_times[:frame_count]
    human_vertices, human_faces = load_human_meshes(
        args.smplx_motion, args.body_model_dir, source_times
    )
    robot_xml = args.robot_xml or Path(ROBOT_XML_DICT["bello"])
    render_comparison(
        args.video_output,
        human_vertices,
        human_faces,
        bello_trajectory,
        bello_fps,
        args.panel_width,
        args.height,
        args.azimuth,
        args.secondary_azimuth,
        robot_xml,
        args.render_fps,
    )
    print(args.video_output.resolve())


if __name__ == "__main__":
    main()
