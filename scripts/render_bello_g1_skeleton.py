import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


SKELETON_EDGES = (
    ("pelvis", "spine1", "center"),
    ("spine1", "spine2", "center"),
    ("spine2", "spine3", "center"),
    ("spine3", "neck", "center"),
    ("neck", "head", "center"),
    ("pelvis", "left_hip", "left"),
    ("left_hip", "left_knee", "left"),
    ("left_knee", "left_ankle", "left"),
    ("left_ankle", "left_foot", "left"),
    ("pelvis", "right_hip", "right"),
    ("right_hip", "right_knee", "right"),
    ("right_knee", "right_ankle", "right"),
    ("right_ankle", "right_foot", "right"),
    ("spine3", "left_collar", "left"),
    ("left_collar", "left_shoulder", "left"),
    ("left_shoulder", "left_elbow", "left"),
    ("left_elbow", "left_wrist", "left"),
    ("spine3", "right_collar", "right"),
    ("right_collar", "right_shoulder", "right"),
    ("right_shoulder", "right_elbow", "right"),
    ("right_elbow", "right_wrist", "right"),
)

COLORS = {
    "center": np.asarray((0.22, 0.88, 0.66, 1.0), dtype=np.float32),
    "left": np.asarray((1.0, 0.48, 0.20, 1.0), dtype=np.float32),
    "right": np.asarray((0.20, 0.58, 1.0, 1.0), dtype=np.float32),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--motion-name", default=None)
    return parser.parse_args()


def camera(root_pos, distance):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(root_pos) + np.asarray((0.0, 0.0, 0.08))
    cam.distance = distance
    cam.azimuth = 90.0
    cam.elevation = -8.0
    return cam


def render_robot(renderer, model, data, qpos, distance):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera(qpos[:3], distance))
    return renderer.render().copy()


def add_geom(scene, geom_type, size, pos, rgba):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Skeleton scene exceeded its geometry budget")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(-1),
        rgba,
    )
    scene.ngeom += 1
    return geom


def render_skeleton(renderer, data, frame):
    root = np.asarray(frame["pelvis"][0])
    renderer.update_scene(data, camera=camera(root, 2.8))
    scene = renderer.scene
    for first, second, color_name in SKELETON_EDGES:
        start = np.asarray(frame[first][0], dtype=np.float64)
        end = np.asarray(frame[second][0], dtype=np.float64)
        geom = add_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            (0.018, 0.0, 0.0),
            start,
            COLORS[color_name],
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.026,
            start,
            end,
        )
    joint_names = {name for edge in SKELETON_EDGES for name in edge[:2]}
    for name in joint_names:
        color_name = "left" if name.startswith("left") else "right"
        if not name.startswith(("left", "right")):
            color_name = "center"
        add_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (0.032, 0.032, 0.032),
            frame[name][0],
            COLORS[color_name],
        )
    return renderer.render().copy()


def label_panel(frame, label):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0]
    draw.rounded_rectangle(
        (12, 12, 28 + width, 42),
        radius=4,
        fill=(245, 245, 242),
    )
    draw.text((20, 19), label, fill=(20, 22, 24), font=font)
    return np.asarray(image)


def main():
    args = parse_args()
    body_model_root = Path(__file__).resolve().parents[1] / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.smplx_file, body_model_root
    )
    human_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=args.fps,
    )

    retargeters = {
        robot: GeneralMotionRetargeting(
            src_human="smplx",
            tgt_robot=robot,
            actual_human_height=human_height,
            source_fps=aligned_fps,
            verbose=False,
        )
        for robot in ("bello", "unitree_g1")
    }
    trajectories = {
        robot: np.asarray([retargeter.retarget(frame) for frame in human_frames])
        for robot, retargeter in retargeters.items()
    }

    robot_scenes = {}
    for robot in retargeters:
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
        model.vis.headlight.ambient[:] = 0.45
        model.vis.headlight.diffuse[:] = 0.85
        robot_scenes[robot] = (
            model,
            mujoco.MjData(model),
            mujoco.Renderer(model, args.height, args.panel_width),
        )
    skeleton_model = mujoco.MjModel.from_xml_string(
        "<mujoco><visual><global offwidth='480' offheight='480'/></visual>"
        "<worldbody><body name='origin'/></worldbody></mujoco>"
    )
    skeleton_data = mujoco.MjData(skeleton_model)
    skeleton_renderer = mujoco.Renderer(
        skeleton_model, args.height, args.panel_width
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    title = args.motion_name or args.smplx_file.stem
    writer = imageio.get_writer(
        args.output,
        fps=aligned_fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for index, human_frame in enumerate(human_frames):
            bello_model, bello_data, bello_renderer = robot_scenes["bello"]
            g1_model, g1_data, g1_renderer = robot_scenes["unitree_g1"]
            panels = (
                label_panel(
                    render_robot(
                        bello_renderer,
                        bello_model,
                        bello_data,
                        trajectories["bello"][index],
                        2.5,
                    ),
                    "BELLO",
                ),
                label_panel(
                    render_robot(
                        g1_renderer,
                        g1_model,
                        g1_data,
                        trajectories["unitree_g1"][index],
                        2.7,
                    ),
                    "UNITREE G1",
                ),
                label_panel(
                    render_skeleton(skeleton_renderer, skeleton_data, human_frame),
                    "SOURCE SKELETON",
                ),
            )
            comparison = Image.fromarray(np.concatenate(panels, axis=1))
            draw = ImageDraw.Draw(comparison)
            draw.text(
                (args.panel_width * 3 - 12, args.height - 12),
                title,
                anchor="rs",
                fill=(230, 230, 226),
                font=ImageFont.load_default(size=14),
            )
            writer.append_data(np.asarray(comparison))
    finally:
        writer.close()
        for _, _, renderer in robot_scenes.values():
            renderer.close()
        skeleton_renderer.close()

    print(f"Wrote {len(human_frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
