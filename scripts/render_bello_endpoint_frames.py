"""Render Bello's attachment-neutral endpoint frames over robot motion."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from general_motion_retargeting.params import ROBOT_XML_DICT


FRAME_NAMES = (
    "l_end_effector_sphere_link",
    "r_end_effector_sphere_link",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--video-output", type=Path, required=True)
    parser.add_argument("--default-pose-output", type=Path, default=None)
    parser.add_argument("--robot-xml", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--primary-azimuth", type=float, default=90.0)
    parser.add_argument("--secondary-azimuth", type=float, default=52.0)
    parser.add_argument("--camera-distance", type=float, default=2.3)
    parser.add_argument("--lookat-height-offset", type=float, default=-0.08)
    parser.add_argument("--primary-label", default="FRONTAL")
    parser.add_argument("--secondary-label", default="OBLIQUE")
    parser.add_argument("--motion-label", default="RETARGETED MOTION")
    parser.add_argument("--default-pose-seconds", type=float, default=2.0)
    parser.add_argument("--max-motion-seconds", type=float, default=6.0)
    parser.add_argument(
        "--hold-lower-body-at-default",
        action="store_true",
        help="copy only arm joints from the input motion",
    )
    return parser.parse_args()


def load_trajectory(path: Path) -> tuple[np.ndarray, float]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as motion:
            qpos = np.asarray(motion["qpos"], dtype=float)
            timestamps = np.asarray(motion["timestamp_seconds"], dtype=float)
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("motion timestamps must be strictly increasing")
        fps = 1.0 / float(np.median(np.diff(timestamps)))
    else:
        with path.open("rb") as motion_file:
            motion = pickle.load(motion_file)  # noqa: S301 - trusted local output
        root_position = np.asarray(motion["root_pos"], dtype=float)
        root_xyzw = np.asarray(motion["root_rot"], dtype=float)
        joint_position = np.asarray(motion["dof_pos"], dtype=float)
        qpos = np.concatenate(
            (root_position, root_xyzw[:, [3, 0, 1, 2]], joint_position), axis=1
        )
        fps = float(motion["fps"])
    if qpos.ndim != 2 or not np.all(np.isfinite(qpos)):
        raise ValueError("motion qpos must be a finite two-dimensional array")
    if fps <= 0.0:
        raise ValueError("motion frame rate must be positive")
    return qpos, fps


def grounded_default_pose(model: mujoco.MjModel) -> np.ndarray:
    qpos = model.qpos0.copy()
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    minimum_height = np.inf
    for side in ("left", "right"):
        geom_id = model.geom(f"{side}_ankle_roll_link_collision_box_1").id
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        center = data.geom_xpos[geom_id]
        half_size = model.geom_size[geom_id]
        for signs in np.ndindex(2, 2, 2):
            corner = half_size * (2.0 * np.asarray(signs) - 1.0)
            minimum_height = min(
                minimum_height, float((center + rotation @ corner)[2])
            )
    qpos[2] -= minimum_height
    return qpos


def make_camera(
    lookat: np.ndarray, azimuth: float, distance: float
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = -7.0
    return camera


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


def add_endpoint_overlays(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    for frame_name in FRAME_NAMES:
        frame_id = model.body(frame_name).id
        position = data.xpos[frame_id]
        rotation = data.xmat[frame_id].reshape(3, 3)

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


def add_floor(scene: mujoco.MjvScene) -> None:
    floor = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        floor,
        mujoco.mjtGeom.mjGEOM_PLANE,
        np.asarray((25.0, 25.0, 0.05)),
        np.asarray((0.0, 0.0, -0.006)),
        np.eye(3).reshape(-1),
        np.asarray((0.16, 0.18, 0.21, 1.0), dtype=np.float32),
    )
    scene.ngeom += 1


def load_font(size: int) -> ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if font_path.exists():
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def annotate(
    image: np.ndarray,
    default_pose: bool,
    lower_body_held: bool,
    panel_width: int,
    primary_label: str,
    secondary_label: str,
    motion_label: str,
) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    label_font = load_font(19)
    draw.rectangle((0, 0, canvas.width, 82), fill=(11, 13, 17))
    draw.text(
        (20, 12),
        "RED +X = INWARD PALM NORMAL     CYAN -Z = DISTAL / TOOL DIRECTION",
        fill=(245, 247, 250),
        font=title_font,
    )
    if default_pose:
        phase = "URDF DEFAULT JOINT POSE"
    elif lower_body_held:
        phase = "ARM FRAME MOTION  |  lower body held at URDF default"
    else:
        phase = motion_label
    draw.text(
        (20, 49),
        f"{phase}  |  gold plates and arrows are diagnostic overlays only",
        fill=(255, 205, 93),
        font=label_font,
    )
    draw.text((18, 92), primary_label, fill=(245, 247, 250), font=label_font)
    draw.text(
        (panel_width + 18, 92),
        secondary_label,
        fill=(245, 247, 250),
        font=label_font,
    )
    return np.asarray(canvas)


def render_panel(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    add_floor(renderer.scene)
    add_endpoint_overlays(renderer.scene, model, data)
    return renderer.render().copy()


def render(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    fps: float,
    output_path: Path,
    default_pose_output: Path | None,
    width: int,
    height: int,
    default_pose_frames: int,
    lower_body_held: bool,
    primary_azimuth: float,
    secondary_azimuth: float,
    camera_distance: float,
    lookat_height_offset: float,
    primary_label: str,
    secondary_label: str,
    motion_label: str,
) -> None:
    if trajectory.shape[1] != model.nq:
        raise ValueError(f"expected qpos width {model.nq}, got {trajectory.shape[1]}")
    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")

    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.headlight.ambient[:] = 0.42
    model.vis.headlight.diffuse[:] = 0.82
    model.vis.headlight.specular[:] = 0.25
    model.vis.quality.offsamples = 4
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    torso_id = model.body("torso_link").id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart"],
    )
    try:
        for frame, qpos in enumerate(trajectory):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            lookat = data.xpos[torso_id].copy()
            lookat[2] += lookat_height_offset
            front = render_panel(
                renderer,
                model,
                data,
                make_camera(lookat, primary_azimuth, camera_distance),
            )
            oblique = render_panel(
                renderer,
                model,
                data,
                make_camera(lookat, secondary_azimuth, camera_distance),
            )
            composite = annotate(
                np.concatenate((front, oblique), axis=1),
                frame < default_pose_frames,
                lower_body_held,
                width,
                primary_label,
                secondary_label,
                motion_label,
            )
            if frame == 0 and default_pose_output is not None:
                default_pose_output.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(default_pose_output, composite)
            writer.append_data(composite)
    finally:
        writer.close()
        renderer.close()


def main() -> None:
    args = parse_args()
    robot_xml = args.robot_xml or Path(ROBOT_XML_DICT["bello"])
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    motion, fps = load_trajectory(args.motion)
    motion_frames = min(len(motion), round(args.max_motion_seconds * fps))
    default_pose_frames = round(args.default_pose_seconds * fps)
    default_pose = grounded_default_pose(model)
    if args.hold_lower_body_at_default:
        arm_motion = np.repeat(default_pose[None], len(motion), axis=0)
        for side in ("left", "right"):
            for joint in (
                "shoulder_pitch_joint",
                "shoulder_roll_joint",
                "shoulder_yaw_joint",
                "elbow_pitch_joint",
                "elbow_yaw_joint",
                "wrist_pitch_joint",
            ):
                qpos_address = model.joint(f"{side}_{joint}").qposadr[0]
                arm_motion[:, qpos_address] = motion[:, qpos_address]
        motion = arm_motion
    trajectory = np.concatenate(
        (
            np.repeat(default_pose[None], default_pose_frames, axis=0),
            motion[:motion_frames],
        ),
        axis=0,
    )
    render(
        model,
        trajectory,
        fps,
        args.video_output,
        args.default_pose_output,
        args.width,
        args.height,
        default_pose_frames,
        args.hold_lower_body_at_default,
        args.primary_azimuth,
        args.secondary_azimuth,
        args.camera_distance,
        args.lookat_height_offset,
        args.primary_label,
        args.secondary_label,
        args.motion_label,
    )
    print(args.video_output.resolve())


if __name__ == "__main__":
    main()
