"""Render synchronized SMPL-X, Bello, and optional G1 motion side by side."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import pickle

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp
import smplx
from smplx.joint_names import JOINT_NAMES
import torch

from general_motion_retargeting.params import ROBOT_XML_DICT


BELLO_FRAME_NAMES = (
    "l_end_effector_sphere_link",
    "r_end_effector_sphere_link",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplx-motion", type=Path, required=True)
    parser.add_argument("--bello-motion", type=Path, required=True)
    parser.add_argument("--g1-motion", type=Path, default=None)
    parser.add_argument("--video-output", type=Path, required=True)
    parser.add_argument(
        "--body-model-dir", type=Path, default=Path("assets/body_models")
    )
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--secondary-azimuth", type=float, default=None)
    parser.add_argument("--robot-xml", type=Path, default=None)
    parser.add_argument("--g1-xml", type=Path, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--render-fps", type=float, default=25.0)
    parser.add_argument("--camera-distance", type=float, default=2.3)
    parser.add_argument("--motion-label", default="RETARGETED MOTION")
    parser.add_argument(
        "--show-endpoint-frames",
        action="store_true",
        help="draw anatomical palm-frame overlays on every model",
    )
    parser.add_argument(
        "--show-leg-orientation",
        action="store_true",
        help="draw knee-bend and toe-heading arrows on every model",
    )
    return parser.parse_args()


def load_human_meshes(
    path: Path, body_model_dir: Path, target_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    wrist_positions = []
    wrist_indices = [JOINT_NAMES.index(name) for name in ("left_wrist", "right_wrist")]
    leg_joint_names = (
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_foot",
        "left_big_toe",
        "left_small_toe",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_foot",
        "right_big_toe",
        "right_small_toe",
    )
    leg_joint_indices = [JOINT_NAMES.index(name) for name in leg_joint_names]
    leg_joint_positions = []
    with torch.inference_mode():
        for start in range(0, frame_count, 64):
            stop = min(start + 64, frame_count)
            chunk_frames = stop - start
            repeated_betas = np.broadcast_to(betas, (chunk_frames, len(betas))).copy()
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
            wrist_positions.append(
                output.joints[:, wrist_indices].detach().cpu().numpy()
            )
            leg_joint_positions.append(
                output.joints[:, leg_joint_indices].detach().cpu().numpy()
            )
            del output
    vertices = np.concatenate(vertices, axis=0)
    wrist_positions = np.concatenate(wrist_positions, axis=0)
    leg_joint_positions = np.concatenate(leg_joint_positions, axis=0)
    full_pose = np.concatenate(
        (root_orient[:, None], body_pose.reshape(frame_count, -1, 3)), axis=1
    )
    global_rotations = []
    for joint_index in range(full_pose.shape[1]):
        local_rotation = Rotation.from_rotvec(full_pose[:, joint_index])
        if joint_index == 0:
            global_rotations.append(local_rotation)
            continue
        parent_index = int(body_model.parents[joint_index])
        global_rotations.append(global_rotations[parent_index] * local_rotation)
    wrist_rotations = np.stack(
        [global_rotations[index].as_matrix() for index in wrist_indices], axis=1
    )
    ground_height = float(np.min(vertices[:, :, 2]))
    vertices[:, :, :2] -= translation[:, None, :2]
    vertices[:, :, 2] -= ground_height
    wrist_positions[:, :, :2] -= translation[:, None, :2]
    wrist_positions[:, :, 2] -= ground_height
    leg_joint_positions[:, :, :2] -= translation[:, None, :2]
    leg_joint_positions[:, :, 2] -= ground_height
    faces = np.asarray(body_model.faces, dtype=np.int32)
    del body_model
    gc.collect()
    return vertices, faces, wrist_positions, wrist_rotations, leg_joint_positions


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


def load_robot_trajectory(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
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
            raise ValueError("robot NPZ timestamps must be increasing")
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


def add_connector(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: tuple[float, float, float, float],
) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.full(3, width),
        start,
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        start,
        end,
    )
    scene.ngeom += 1


def add_frame_overlay(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rotation: np.ndarray,
) -> None:
    palm = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        palm,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray((0.006, 0.052, 0.075)),
        position,
        rotation.reshape(-1),
        np.asarray((1.0, 0.72, 0.15, 0.48), dtype=np.float32),
    )
    scene.ngeom += 1
    add_connector(
        scene,
        position,
        position + 0.15 * rotation[:, 0],
        0.009,
        (1.0, 0.08, 0.06, 1.0),
    )
    add_connector(
        scene,
        position,
        position - 0.15 * rotation[:, 2],
        0.008,
        (0.0, 0.88, 1.0, 1.0),
    )


def add_bello_endpoint_overlays(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    for frame_name in BELLO_FRAME_NAMES:
        frame_id = model.body(frame_name).id
        add_frame_overlay(
            scene,
            data.xpos[frame_id],
            data.xmat[frame_id].reshape(3, 3),
        )


def anatomical_frame_rotation(
    wrist_rotation: np.ndarray, side: str, source: str
) -> np.ndarray:
    if source == "human":
        palm_normal = np.array([0.0, -1.0, 0.0])
        distal = np.array([1.0 if side == "left" else -1.0, 0.0, 0.0])
    elif source == "g1":
        palm_normal = np.array([0.0, -1.0 if side == "left" else 1.0, 0.0])
        distal = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError(f"unsupported anatomical frame source: {source}")
    frame_x = wrist_rotation @ palm_normal
    frame_z = -(wrist_rotation @ distal)
    frame_y = np.cross(frame_z, frame_x)
    return np.column_stack((frame_x, frame_y, frame_z))


def add_human_endpoint_overlays(
    scene: mujoco.MjvScene,
    wrist_positions: np.ndarray,
    wrist_rotations: np.ndarray,
) -> None:
    for index, side in enumerate(("left", "right")):
        rotation = anatomical_frame_rotation(wrist_rotations[index], side, "human")
        position = wrist_positions[index] - 0.08 * rotation[:, 2]
        add_frame_overlay(scene, position, rotation)


def add_g1_endpoint_overlays(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    for side in ("left", "right"):
        frame_id = model.body(f"{side}_wrist_yaw_link").id
        wrist_rotation = data.xmat[frame_id].reshape(3, 3)
        rotation = anatomical_frame_rotation(wrist_rotation, side, "g1")
        position = data.xpos[frame_id] + 0.08 * wrist_rotation[:, 0]
        add_frame_overlay(scene, position, rotation)


def normalized_direction(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 1e-6:
        return vector / norm
    fallback_norm = np.linalg.norm(fallback)
    if fallback_norm <= 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return fallback / fallback_norm


def knee_bend_direction(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    hip_to_ankle = ankle - hip
    denominator = float(hip_to_ankle @ hip_to_ankle)
    if denominator <= 1e-12:
        return normalized_direction(fallback, np.array([1.0, 0.0, 0.0]))
    fraction = float((knee - hip) @ hip_to_ankle) / denominator
    closest = hip + np.clip(fraction, 0.0, 1.0) * hip_to_ankle
    return normalized_direction(knee - closest, fallback)


def add_leg_orientation_overlay(
    scene: mujoco.MjvScene,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    foot: np.ndarray,
    toe_heading: np.ndarray,
) -> None:
    toe_direction = normalized_direction(
        toe_heading.copy(), np.array([1.0, 0.0, 0.0])
    )
    planar_toe_heading = toe_heading.copy()
    planar_toe_heading[2] = 0.0
    planar_toe_heading = normalized_direction(
        planar_toe_heading, np.array([1.0, 0.0, 0.0])
    )
    knee_heading = knee_bend_direction(hip, knee, ankle, planar_toe_heading)
    add_connector(
        scene,
        knee,
        knee + 0.18 * knee_heading,
        0.008,
        (0.30, 1.0, 0.10, 1.0),
    )
    toe_origin = foot + np.array([0.0, 0.0, 0.035])
    add_connector(
        scene,
        toe_origin,
        toe_origin + 0.20 * toe_direction,
        0.008,
        (1.0, 0.10, 0.78, 1.0),
    )


def add_human_leg_orientation_overlays(
    scene: mujoco.MjvScene,
    joints: np.ndarray,
) -> None:
    for offset in (0, 6):
        hip, knee, ankle, foot, big_toe, small_toe = joints[offset : offset + 6]
        toe_center = 0.5 * (big_toe + small_toe)
        add_leg_orientation_overlay(
            scene,
            hip,
            knee,
            ankle,
            foot,
            toe_center - foot,
        )


def add_robot_leg_orientation_overlays(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: str,
) -> None:
    if robot == "bello":
        toe_axis = 2
    elif robot == "g1":
        toe_axis = 0
    else:
        raise ValueError(f"unsupported leg-overlay robot: {robot}")
    for side in ("left", "right"):
        hip_id = model.body(f"{side}_hip_yaw_link").id
        knee_id = model.body(f"{side}_knee_link").id
        ankle_id = model.body(f"{side}_ankle_roll_link").id
        foot_rotation = data.xmat[ankle_id].reshape(3, 3)
        add_leg_orientation_overlay(
            scene,
            data.xpos[hip_id],
            data.xpos[knee_id],
            data.xpos[ankle_id],
            data.xpos[ankle_id],
            foot_rotation[:, toe_axis],
        )


def load_font(size: int) -> ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if font_path.exists():
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def annotate_comparison(
    image: np.ndarray,
    motion_label: str,
    panel_width: int,
    panel_height: int,
    has_secondary_view: bool,
    show_endpoint_frames: bool,
    show_leg_orientation: bool,
    include_g1: bool,
) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    label_font = load_font(19)
    small_font = load_font(17)
    draw.rectangle((0, 0, canvas.width, 82), fill=(11, 13, 17))
    if show_endpoint_frames and show_leg_orientation:
        legend = "RED PALM NORMAL  |  CYAN TOOL  |  GREEN KNEE  |  MAGENTA TOE"
        subtitle = f"{motion_label}  |  colored arrows are diagnostic overlays only"
    elif show_endpoint_frames:
        legend = "RED = PALM NORMAL     CYAN = DISTAL / TOOL DIRECTION"
        subtitle = (
            f"{motion_label}  |  gold plates and arrows are diagnostic overlays only"
        )
    elif show_leg_orientation:
        legend = "GREEN = KNEE BEND DIRECTION     MAGENTA = TOE HEADING"
        subtitle = f"{motion_label}  |  arrows are diagnostic overlays only"
    else:
        legend = "SYNCHRONIZED HUMAN / BELLO / G1 RETARGETING COMPARISON"
        subtitle = motion_label
    draw.text((20, 12), legend, fill=(245, 247, 250), font=title_font)
    draw.text((20, 49), subtitle, fill=(255, 205, 93), font=label_font)

    model_names = ["HUMAN", "BELLO"]
    if include_g1:
        model_names.append("G1")
    labels = tuple(
        (f"{name} / SIDE", index * panel_width)
        for index, name in enumerate(model_names)
    )
    for label, x_position in labels:
        draw.rectangle(
            (x_position + 12, 92, x_position + 210, 122),
            fill=(11, 13, 17),
        )
        draw.text(
            (x_position + 18, 96),
            label,
            fill=(245, 247, 250),
            font=small_font,
        )
    if has_secondary_view:
        y_position = panel_height + 12
        labels = tuple(
            (f"{name} / OBLIQUE", index * panel_width)
            for index, name in enumerate(model_names)
        )
        for label, x_position in labels:
            draw.rectangle(
                (
                    x_position + 12,
                    y_position,
                    x_position + 210,
                    y_position + 30,
                ),
                fill=(11, 13, 17),
            )
            draw.text(
                (x_position + 18, y_position + 4),
                label,
                fill=(245, 247, 250),
                font=small_font,
            )
    return np.asarray(canvas)


def render_robot_panel(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    floor_center: np.ndarray,
    show_endpoint_frames: bool,
    show_leg_orientation: bool,
    robot: str,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    add_floor(renderer.scene, floor_center)
    if show_endpoint_frames:
        if robot == "bello":
            add_bello_endpoint_overlays(renderer.scene, model, data)
        elif robot == "g1":
            add_g1_endpoint_overlays(renderer.scene, model, data)
        else:
            raise ValueError(f"unsupported overlay robot: {robot}")
    if show_leg_orientation:
        add_robot_leg_orientation_overlays(renderer.scene, model, data, robot)
    return renderer.render().copy()


def render_comparison(
    output_path: Path,
    human_vertices: np.ndarray,
    human_faces: np.ndarray,
    human_wrist_positions: np.ndarray,
    human_wrist_rotations: np.ndarray,
    human_leg_joint_positions: np.ndarray,
    bello_trajectory: np.ndarray,
    g1_trajectory: np.ndarray | None,
    fps: float,
    width: int,
    height: int,
    azimuth: float,
    secondary_azimuth: float | None,
    robot_xml: Path,
    g1_xml: Path | None,
    render_fps: float,
    camera_distance: float,
    motion_label: str,
    show_endpoint_frames: bool,
    show_leg_orientation: bool,
) -> None:
    if len(human_vertices) != len(bello_trajectory):
        raise ValueError(
            f"frame-count mismatch: human={len(human_vertices)}, "
            f"Bello={len(bello_trajectory)}"
        )
    if g1_trajectory is not None and len(g1_trajectory) != len(bello_trajectory):
        raise ValueError(
            f"frame-count mismatch: Bello={len(bello_trajectory)}, "
            f"G1={len(g1_trajectory)}"
        )
    if len(human_wrist_positions) != len(human_vertices) or len(
        human_wrist_rotations
    ) != len(human_vertices):
        raise ValueError("human wrist-frame count must match human mesh count")
    if len(human_leg_joint_positions) != len(human_vertices):
        raise ValueError("human leg-joint count must match human mesh count")
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

    g1_model = None
    g1_data = None
    g1_renderer = None
    g1_torso_id = None
    if g1_trajectory is not None:
        if g1_xml is None:
            raise ValueError("G1 XML is required with a G1 trajectory")
        g1_model = mujoco.MjModel.from_xml_path(str(g1_xml))
        if g1_trajectory.shape[1] != g1_model.nq:
            raise ValueError(
                f"expected G1 qpos width {g1_model.nq}, got {g1_trajectory.shape[1]}"
            )
        g1_model.vis.global_.offwidth = width
        g1_model.vis.global_.offheight = height
        g1_model.vis.headlight.ambient[:] = 0.42
        g1_model.vis.headlight.diffuse[:] = 0.82
        g1_model.vis.headlight.specular[:] = 0.25
        g1_model.vis.quality.offsamples = 4
        g1_data = mujoco.MjData(g1_model)
        g1_renderer = mujoco.Renderer(g1_model, height=height, width=width)
        g1_torso_id = g1_model.body("torso_link").id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if render_fps <= 0.0 or render_fps > fps:
        raise ValueError(
            "render frame rate must be positive and no greater than motion"
        )
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
                human_data,
                camera=make_camera(human_lookat, azimuth + 180.0, camera_distance),
            )
            if show_endpoint_frames:
                add_human_endpoint_overlays(
                    human_renderer.scene,
                    human_wrist_positions[frame],
                    human_wrist_rotations[frame],
                )
            if show_leg_orientation:
                add_human_leg_orientation_overlays(
                    human_renderer.scene,
                    human_leg_joint_positions[frame],
                )
            human_image = human_renderer.render().copy()
            if secondary_azimuth is not None:
                human_renderer.update_scene(
                    human_data,
                    camera=make_camera(
                        human_lookat,
                        secondary_azimuth + 180.0,
                        camera_distance,
                    ),
                )
                if show_endpoint_frames:
                    add_human_endpoint_overlays(
                        human_renderer.scene,
                        human_wrist_positions[frame],
                        human_wrist_rotations[frame],
                    )
                if show_leg_orientation:
                    add_human_leg_orientation_overlays(
                        human_renderer.scene,
                        human_leg_joint_positions[frame],
                    )
                human_secondary = human_renderer.render().copy()

            bello_data.qpos[:] = qpos
            mujoco.mj_forward(bello_model, bello_data)
            bello_image = render_robot_panel(
                bello_renderer,
                bello_model,
                bello_data,
                make_camera(bello_data.xpos[torso_id], azimuth, camera_distance),
                qpos[:3],
                show_endpoint_frames,
                show_leg_orientation,
                "bello",
            )
            primary_images = [human_image, bello_image]
            g1_secondary = None
            if g1_trajectory is not None:
                assert g1_model is not None
                assert g1_data is not None
                assert g1_renderer is not None
                assert g1_torso_id is not None
                g1_qpos = g1_trajectory[frame]
                g1_data.qpos[:] = g1_qpos
                mujoco.mj_forward(g1_model, g1_data)
                g1_image = render_robot_panel(
                    g1_renderer,
                    g1_model,
                    g1_data,
                    make_camera(g1_data.xpos[g1_torso_id], azimuth, camera_distance),
                    g1_qpos[:3],
                    show_endpoint_frames,
                    show_leg_orientation,
                    "g1",
                )
                primary_images.append(g1_image)
                if secondary_azimuth is not None:
                    g1_secondary = render_robot_panel(
                        g1_renderer,
                        g1_model,
                        g1_data,
                        make_camera(
                            g1_data.xpos[g1_torso_id],
                            secondary_azimuth,
                            camera_distance,
                        ),
                        g1_qpos[:3],
                        show_endpoint_frames,
                        show_leg_orientation,
                        "g1",
                    )
            primary = np.concatenate(primary_images, axis=1)
            if secondary_azimuth is None:
                writer.append_data(
                    annotate_comparison(
                        primary,
                        motion_label,
                        width,
                        height,
                        False,
                        show_endpoint_frames,
                        show_leg_orientation,
                        g1_trajectory is not None,
                    )
                )
                continue
            bello_secondary = render_robot_panel(
                bello_renderer,
                bello_model,
                bello_data,
                make_camera(
                    bello_data.xpos[torso_id],
                    secondary_azimuth,
                    camera_distance,
                ),
                qpos[:3],
                show_endpoint_frames,
                show_leg_orientation,
                "bello",
            )
            secondary_images = [human_secondary, bello_secondary]
            if g1_secondary is not None:
                secondary_images.append(g1_secondary)
            secondary = np.concatenate(secondary_images, axis=1)
            writer.append_data(
                annotate_comparison(
                    np.concatenate((primary, secondary), axis=0),
                    motion_label,
                    width,
                    height,
                    True,
                    show_endpoint_frames,
                    show_leg_orientation,
                    g1_trajectory is not None,
                )
            )
    finally:
        writer.close()
        human_renderer.close()
        bello_renderer.close()
        if g1_renderer is not None:
            g1_renderer.close()


def main() -> None:
    args = parse_args()
    bello_trajectory, bello_fps, source_times = load_robot_trajectory(args.bello_motion)
    g1_trajectory = None
    if args.g1_motion is not None:
        g1_trajectory, g1_fps, g1_source_times = load_robot_trajectory(args.g1_motion)
        if not np.isclose(g1_fps, bello_fps):
            raise ValueError(f"frame-rate mismatch: Bello={bello_fps}, G1={g1_fps}")
        if len(g1_source_times) != len(source_times) or not np.allclose(
            g1_source_times, source_times
        ):
            raise ValueError("Bello and G1 source timestamps must match")
    if args.max_seconds is not None:
        if args.max_seconds <= 0.0:
            raise ValueError("maximum duration must be positive")
        frame_count = min(
            len(bello_trajectory), int(np.floor(args.max_seconds * bello_fps)) + 1
        )
        bello_trajectory = bello_trajectory[:frame_count]
        source_times = source_times[:frame_count]
        if g1_trajectory is not None:
            g1_trajectory = g1_trajectory[:frame_count]
    (
        human_vertices,
        human_faces,
        human_wrist_positions,
        human_wrist_rotations,
        human_leg_joint_positions,
    ) = load_human_meshes(args.smplx_motion, args.body_model_dir, source_times)
    robot_xml = args.robot_xml or Path(ROBOT_XML_DICT["bello"])
    g1_xml = args.g1_xml
    if g1_trajectory is not None and g1_xml is None:
        g1_xml = Path(ROBOT_XML_DICT["unitree_g1"])
    render_comparison(
        args.video_output,
        human_vertices,
        human_faces,
        human_wrist_positions,
        human_wrist_rotations,
        human_leg_joint_positions,
        bello_trajectory,
        g1_trajectory,
        bello_fps,
        args.panel_width,
        args.height,
        args.azimuth,
        args.secondary_azimuth,
        robot_xml,
        g1_xml,
        args.render_fps,
        args.camera_distance,
        args.motion_label,
        args.show_endpoint_frames,
        args.show_leg_orientation,
    )
    print(args.video_output.resolve())


if __name__ == "__main__":
    main()
