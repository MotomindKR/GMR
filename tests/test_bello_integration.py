import json
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
from general_motion_retargeting.quality import joint_limit_saturation_metrics
from general_motion_retargeting.utils.smpl import _target_frame_coordinates


EXPECTED_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
)


def test_smplx_resampling_uses_exact_target_rate() -> None:
    np.testing.assert_allclose(
        _target_frame_coordinates(121, 120.0, 50.0),
        np.arange(51) * 120.0 / 50.0,
    )
    np.testing.assert_allclose(
        _target_frame_coordinates(31, 30.0, 50.0),
        np.arange(51) * 30.0 / 50.0,
    )


def load_models():
    viewer = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["bello"]))
    box_path = Path(ROBOT_XML_DICT["bello"]).with_name("bello_full_body_boxes.xml")
    boxes = mujoco.MjModel.from_xml_path(str(box_path))
    return viewer, boxes


def test_source_generated_model_contract() -> None:
    viewer, boxes = load_models()
    assert (viewer.nq, viewer.nv, viewer.nu) == (32, 31, 25)
    assert (boxes.nq, boxes.nv, boxes.nu) == (32, 31, 25)

    model_joints = tuple(
        viewer.joint(joint_id).name for joint_id in range(1, viewer.njnt)
    )
    assert model_joints == EXPECTED_JOINTS
    assert "waist_roll_joint" not in model_joints
    assert "waist_pitch_joint" not in model_joints
    assert "neck_yaw_joint" not in model_joints
    assert "head_pitch_joint" not in model_joints
    assert viewer.body("waist_roll_link").id >= 0
    assert viewer.body("torso_link").id >= 0


def test_end_effector_frames_face_inward_in_default_pose() -> None:
    for model in load_models():
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        torso_position = data.xpos[model.body("torso_link").id]

        for prefix in ("l", "r"):
            frame_id = model.body(f"{prefix}_end_effector_sphere_link").id
            frame_rotation = data.xmat[frame_id].reshape(3, 3)
            inward = torso_position - data.xpos[frame_id]
            inward[2] = 0.0
            inward /= np.linalg.norm(inward)

            np.testing.assert_allclose(frame_rotation[:, 0], inward, atol=1e-8)
            np.testing.assert_allclose(
                -frame_rotation[:, 2], np.array([0.0, 0.0, -1.0]), atol=1e-8
            )


def test_elbow_yaw_ranges_are_anatomically_mirrored() -> None:
    for model in load_models():
        np.testing.assert_allclose(
            model.joint("left_elbow_yaw_joint").range,
            np.deg2rad([-120.0, 30.0]),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            model.joint("right_elbow_yaw_joint").range,
            np.deg2rad([-30.0, 120.0]),
            atol=1e-5,
        )


def test_box_model_uses_horizontal_primitive_soles() -> None:
    _, model = load_models()
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    for side in ("left", "right"):
        ankle_pitch = model.joint(f"{side}_ankle_pitch_joint")
        data.qpos[ankle_pitch.qposadr[0]] = np.deg2rad(15.0)
    mujoco.mj_forward(model, data)

    for side in ("left", "right"):
        prefix = f"{side}_ankle_roll_link_collision_"
        foot_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom(geom_id).name.startswith(prefix)
        ]
        assert len(foot_geom_ids) == 5
        assert (
            sum(
                model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                for geom_id in foot_geom_ids
            )
            == 3
        )
        assert (
            sum(
                model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
                for geom_id in foot_geom_ids
            )
            == 2
        )

        sole_id = model.geom(f"{prefix}box_1").id
        sole_normal = data.geom_xmat[sole_id].reshape(3, 3)[:, 2]
        assert abs(float(sole_normal[2])) > 0.9999


def test_bello_config_is_symmetric_and_references_model() -> None:
    model, _ = load_models()
    config = json.loads(Path(IK_CONFIG_DICT["smplx"]["bello"]).read_text())
    assert "ground_alignment_axes" not in config
    scales = config["human_scale_table"]
    for landmark in ("hip", "knee", "foot", "shoulder", "elbow", "wrist"):
        assert scales[f"left_{landmark}"] == scales[f"right_{landmark}"]

    table1 = config["ik_match_table1"]
    table2 = config["ik_match_table2"]
    assert {entry[0] for entry in table1.values()} == set(scales)
    assert {entry[0] for entry in table2.values()} == set(scales)
    for table in (table1, table2):
        for body_name, entry in table.items():
            assert model.body(body_name).id >= 0
            np.testing.assert_allclose(np.linalg.norm(entry[4]), 1.0, atol=1e-8)

    for left_name, right_name in (
        ("l_upper_arm_link", "r_upper_arm_link"),
        ("l_elbow_link", "r_elbow_link"),
        ("l_end_effector_sphere_link", "r_end_effector_sphere_link"),
    ):
        assert table1[left_name][1:3] == table1[right_name][1:3]
        assert table2[left_name][1:3] == table2[right_name][1:3]

    for prefix in ("l", "r"):
        frame_name = f"{prefix}_end_effector_sphere_link"
        assert table1[frame_name][1:3] == [0, 0]
        assert table2[frame_name][1] > 0
        assert table2[frame_name][2] == [30, 30, 30]
        assert table1[frame_name][3] == table2[frame_name][3]
        assert table1[frame_name][4] == table2[frame_name][4]
        frame_rotation = Rotation.from_quat(table1[frame_name][4], scalar_first=True)
        expected_distal = np.array([1.0, 0.0, 0.0])
        if prefix == "r":
            expected_distal *= -1.0
        np.testing.assert_allclose(
            frame_rotation.apply(np.array([1.0, 0.0, 0.0])),
            np.array([0.0, -1.0, 0.0]),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            frame_rotation.apply(np.array([0.0, 0.0, -1.0])),
            expected_distal,
            atol=1e-8,
        )

        elbow_name = f"{prefix}_elbow_link"
        assert table1[elbow_name][1:3] == [0, 0]
        assert table2[elbow_name][1] > 0
        assert table2[elbow_name][2] == 0

        upper_arm_name = f"{prefix}_upper_arm_link"
        assert table1[upper_arm_name][1] == table2[upper_arm_name][1] == 0
        assert table1[upper_arm_name][2] > 0
        assert table2[upper_arm_name][2] == 0

    for side in ("left", "right"):
        hip1 = table1[f"{side}_hip_roll_link"]
        hip2 = table2[f"{side}_hip_roll_link"]
        knee1 = table1[f"{side}_knee_link"]
        knee2 = table2[f"{side}_knee_link"]
        foot1 = table1[f"{side}_ankle_roll_link"]
        foot2 = table2[f"{side}_ankle_roll_link"]
        assert hip1[1] == knee1[1] == 0
        assert hip1[2] > 0
        assert knee1[2] == [4, 10, 10]
        assert 0 < knee1[2][0] < min(knee1[2][1:])
        assert hip2[1] > 0 and hip2[2] > 0
        assert knee2[1] > 0
        assert knee2[2] == [2, 5, 5]
        assert 0 < knee2[2][0] < min(knee2[2][1:])
        for foot in (foot1, foot2):
            assert foot[1] > 0
            assert foot[2] == [10, 3, 10]
            assert 0 < foot[2][1] < min(foot[2][0], foot[2][2])

    assert table1["bello_root"][2] == [0, 0, 10]
    assert table2["bello_root"][2] == [0, 0, 5]
    assert table1["torso_link"][1:3] == [0, [50, 50, 0]]
    assert table2["torso_link"][1:3] == [0, [30, 30, 0]]
    for table in (table1, table2):
        assert table["torso_link"][4] == table["bello_root"][4]

    yaw_task = config["planar_relative_yaw_task"]
    assert model.body(yaw_task["robot_frame_name"]).id >= 0
    assert model.body(yaw_task["robot_root_name"]).id >= 0
    assert model.joint(yaw_task["robot_joint_name"]).id >= 0
    assert yaw_task["human_frame_landmarks"] == [
        "left_shoulder",
        "right_shoulder",
    ]
    assert yaw_task["human_root_landmarks"] == ["left_hip", "right_hip"]
    assert yaw_task["orientation_cost"] > 0

    for geom_name in config["ground_clearance_geoms"]:
        assert model.geom(geom_name).id >= 0

    forbidden_solver_keys = {
        "fixed_iterations",
        "joint_acceleration_limits",
        "joint_position_limits",
        "max_joint_velocity",
        "posture_costs",
        "posture_targets",
        "use_velocity_limit",
    }
    assert forbidden_solver_keys.isdisjoint(config)
    collision = config["collision_avoidance"]
    assert collision["enabled"]
    assert collision["geom_pairs"]
    assert 0.0 < collision["gain"] <= 1.0
    assert collision["detection_distance"] > collision["minimum_distance"]

    profiles = config["task_profiles"]
    assert config["default_task_profile"] == "universal"
    assert set(profiles) == {"universal", "live_upper_body"}
    assert profiles["universal"] == {}
    for prefix in ("l", "r"):
        table = config["ik_match_table2"]
        assert table[f"{prefix}_upper_arm_link"][2] == 0
        assert table[f"{prefix}_elbow_link"][1] == 30
        assert table[f"{prefix}_end_effector_sphere_link"][1:3] == [
            50,
            [30, 30, 30],
        ]


def test_bello_wrist_position_targets_have_wrist_pitch_leverage() -> None:
    model, _ = load_models()
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    for side, prefix in (("left", "l"), ("right", "r")):
        frame_id = model.body(f"{prefix}_end_effector_sphere_link").id
        joint_id = model.joint(f"{side}_wrist_pitch_joint").id
        dof_id = int(model.jnt_dofadr[joint_id])
        position_jacobian = np.zeros((3, model.nv))
        rotation_jacobian = np.zeros((3, model.nv))

        mujoco.mj_jacBody(
            model,
            data,
            position_jacobian,
            rotation_jacobian,
            frame_id,
        )

        assert np.linalg.norm(position_jacobian[:, dof_id]) > 0.1


def test_bello_relative_waist_yaw_follows_shoulder_twist_direction() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
    )
    human_data = {
        "left_hip": [np.array([-1.0, 0.0, 0.0]), None],
        "right_hip": [np.array([1.0, 0.0, 0.0]), None],
    }
    address = retargeter.planar_relative_yaw_joint_qpos_address

    for angle_degrees in (30.0, -30.0):
        angle = np.deg2rad(angle_degrees)
        shoulder_axis = np.array([np.cos(angle), np.sin(angle), 0.0])
        human_data["left_shoulder"] = [-shoulder_axis, None]
        human_data["right_shoulder"] = [shoulder_axis, None]
        retargeter.update_planar_relative_yaw_target(human_data)

        np.testing.assert_allclose(
            retargeter.planar_relative_yaw_reference.data.qpos[address],
            angle,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            retargeter.planar_relative_yaw_task.compute_error(
                retargeter.planar_relative_yaw_reference
            ),
            np.zeros(6),
            atol=1e-8,
        )


def test_bello_joint_ranges_match_the_updated_robot_contract() -> None:
    models = load_models()
    expected_shoulder_pitch_ranges = {
        "left": np.deg2rad([-170.0, 45.0]),
        "right": np.deg2rad([-45.0, 170.0]),
    }
    for model in models:
        for side in ("left", "right"):
            np.testing.assert_allclose(
                model.joint(f"{side}_ankle_roll_joint").range,
                np.array([-0.1745, 0.1745]),
                atol=1e-7,
            )
        np.testing.assert_allclose(
            model.joint("waist_yaw_joint").range,
            np.array([-np.pi / 4.0, np.pi / 4.0]),
            atol=1e-8,
        )
        for side, shoulder_range in expected_shoulder_pitch_ranges.items():
            np.testing.assert_allclose(
                model.joint(f"{side}_shoulder_pitch_joint").range,
                shoulder_range,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                model.actuator(f"{side}_shoulder_pitch_joint_pos").ctrlrange,
                shoulder_range,
                atol=1e-6,
            )


def test_ground_alignment_projects_configured_local_axis_to_world_up() -> None:
    retargeter = object.__new__(GeneralMotionRetargeting)
    retargeter.ground_alignment_axes = {
        "left_foot": np.array([0.0, 1.0, 0.0]),
        "right_foot": np.array([0.0, -1.0, 0.0]),
    }
    input_rotation = mink.SO3.from_rpy_radians(0.4, -0.3, 1.2)
    human_data = {
        body_name: [np.zeros(3), input_rotation.wxyz]
        for body_name in retargeter.ground_alignment_axes
    }

    aligned = retargeter.align_human_data_to_ground(human_data)

    for body_name, local_axis in retargeter.ground_alignment_axes.items():
        rotation = mink.SO3(aligned[body_name][1])
        np.testing.assert_allclose(
            rotation.as_matrix() @ local_axis,
            np.array([0.0, 0.0, 1.0]),
            atol=1e-8,
        )


def test_bello_ground_clearance_only_raises_penetrating_soles() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
    )
    retargeter.enforce_ground_clearance()
    grounded_qpos = retargeter.configuration.data.qpos.copy()
    retargeter.enforce_ground_clearance()
    np.testing.assert_array_equal(retargeter.configuration.data.qpos, grounded_qpos)

    lowered_qpos = grounded_qpos.copy()
    lowered_qpos[retargeter.root_height_qpos_address] -= 1.0
    retargeter.configuration.update(lowered_qpos)
    retargeter.enforce_ground_clearance()

    for geom_id in retargeter.ground_clearance_geom_ids:
        sole_height = retargeter.configuration.data.geom_xpos[geom_id, 2] - np.sum(
            np.abs(retargeter.configuration.data.geom_xmat[geom_id].reshape(3, 3)[2])
            * retargeter.model.geom_size[geom_id]
        )
        assert sole_height >= -1e-10


def test_bello_uses_shared_retargeter() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
    )
    assert type(retargeter) is GeneralMotionRetargeting
    assert retargeter.max_iter == 10
    assert len(retargeter.ik_limits) == 2
    assert isinstance(retargeter.ik_limits[0], mink.ConfigurationLimit)
    assert isinstance(retargeter.ik_limits[1], mink.CollisionAvoidanceLimit)
    assert not any(
        isinstance(task, mink.PostureTask)
        for task in (*retargeter.tasks1, *retargeter.tasks2)
    )
    relative_tasks = [
        task for task in retargeter.tasks2 if isinstance(task, mink.RelativeFrameTask)
    ]
    assert relative_tasks == [retargeter.planar_relative_yaw_task]
    assert not hasattr(retargeter, "previous_output_qpos")
    assert not hasattr(retargeter, "fixed_iterations")

    for side in ("left", "right"):
        assert f"{side}_hip" in retargeter.rot_offsets1
        assert f"{side}_knee" in retargeter.rot_offsets1
        assert f"{side}_hip" in retargeter.human_body_to_task1
        assert f"{side}_knee" in retargeter.human_body_to_task1
        assert f"{side}_knee" in retargeter.human_body_to_task2


def test_velocity_limit_is_an_explicit_shared_solver_option() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
        use_velocity_limit=True,
    )
    assert len(retargeter.ik_limits) == 3
    assert isinstance(retargeter.ik_limits[0], mink.ConfigurationLimit)
    assert isinstance(retargeter.ik_limits[1], mink.VelocityLimit)
    assert isinstance(retargeter.ik_limits[2], mink.CollisionAvoidanceLimit)


def test_universal_profile_uses_one_arm_task_balance() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
        task_profile="universal",
    )
    for side in ("left", "right"):
        assert f"{side}_shoulder" not in retargeter.human_body_to_task2
        elbow_task = retargeter.human_body_to_task2[f"{side}_elbow"]
        np.testing.assert_allclose(elbow_task.cost, [30, 30, 30, 0, 0, 0])
        wrist_task = retargeter.human_body_to_task2[f"{side}_wrist"]
        np.testing.assert_allclose(wrist_task.cost, [50, 50, 50, 30, 30, 30])
    assert retargeter.profile_posture_task is None
    assert not any(
        isinstance(task, mink.PostureTask)
        for task in (*retargeter.tasks1, *retargeter.tasks2)
    )
    assert retargeter.offline_solver_config["initial_settle_passes"] == 20
    np.testing.assert_allclose(
        retargeter.offline_solver_config["joint_smoothing_kernel"],
        [0.0625, 0.25, 0.375, 0.25, 0.0625],
    )
    assert retargeter.offline_solver_config["joint_smoothing_passes"] == 3
    assert retargeter.offline_solver_config["maximum_collision_penetration"] == 0.03


def test_live_upper_body_profile_uses_only_observable_arm_positions() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
        task_profile="live_upper_body",
    )
    for side in ("left", "right"):
        assert f"{side}_shoulder" not in retargeter.human_body_to_task1
        assert f"{side}_shoulder" not in retargeter.human_body_to_task2
        elbow_task = retargeter.human_body_to_task2[f"{side}_elbow"]
        np.testing.assert_allclose(elbow_task.cost, [30, 30, 30, 0, 0, 0])
        wrist_task = retargeter.human_body_to_task2[f"{side}_wrist"]
        np.testing.assert_allclose(wrist_task.cost, [50, 50, 50, 0, 0, 0])
    assert not any(isinstance(task, mink.PostureTask) for task in retargeter.tasks1)
    posture_tasks = [
        task for task in retargeter.tasks2 if isinstance(task, mink.PostureTask)
    ]
    assert posture_tasks == [retargeter.profile_posture_task]
    np.testing.assert_array_equal(
        retargeter.profile_posture_task.target_q, retargeter.model.qpos0
    )
    expected_cost = np.zeros(retargeter.model.nv)
    for side in ("left", "right"):
        for joint_kind in ("shoulder_yaw", "elbow_yaw"):
            joint = retargeter.model.joint(f"{side}_{joint_kind}_joint")
            expected_cost[joint.dofadr[0]] = 10
    np.testing.assert_array_equal(retargeter.profile_posture_task.cost, expected_cost)


def test_live_upper_body_static_reachable_pose_does_not_accumulate_axial_twist() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
        use_velocity_limit=None,
        task_profile="live_upper_body",
    )
    assert retargeter.use_velocity_limit
    retargeter.max_iter = 0

    target_qpos = retargeter.model.qpos0.copy()
    for side, shoulder_pitch in (("left", -0.35), ("right", 0.35)):
        target_qpos[
            retargeter.model.joint(f"{side}_shoulder_pitch_joint").qposadr[0]
        ] = shoulder_pitch
        target_qpos[
            retargeter.model.joint(f"{side}_shoulder_roll_joint").qposadr[0]
        ] = 0.25
        target_qpos[
            retargeter.model.joint(f"{side}_elbow_pitch_joint").qposadr[0]
        ] = 0.75
    retargeter.configuration.update(target_qpos)
    retargeter.enforce_ground_clearance()

    scaled_positions = {}
    source_orientations = {}
    for frame_name, entry in retargeter.ik_match_table1.items():
        source_name = entry[0]
        body_id = retargeter.model.body(frame_name).id
        frame_rotation = Rotation.from_matrix(
            retargeter.configuration.data.xmat[body_id].reshape(3, 3)
        )
        scaled_positions[source_name] = (
            retargeter.configuration.data.xpos[body_id]
            - frame_rotation.apply(retargeter.pos_offsets1[source_name])
        )
        source_orientations[source_name] = (
            frame_rotation * retargeter.rot_offsets1[source_name].inv()
        ).as_quat(scalar_first=True)

    root_name = retargeter.human_root_name
    source_root = (
        scaled_positions[root_name] / retargeter.human_scale_table[root_name]
    )
    source_positions = {root_name: source_root}
    for source_name, scaled_position in scaled_positions.items():
        if source_name == root_name:
            continue
        source_positions[source_name] = source_root + (
            scaled_position - scaled_positions[root_name]
        ) / retargeter.human_scale_table[source_name]

    def standing_frame():
        return {
            source_name: [
                source_positions[source_name].copy(),
                source_orientations[source_name].copy(),
            ]
            for source_name in source_positions
        }

    retargeter.configuration.update(retargeter.model.qpos0)
    maximum_axial_twist = 0.0
    for _ in range(500):
        qpos = retargeter.retarget(standing_frame())
        for side in ("left", "right"):
            for joint_kind in ("shoulder_yaw", "elbow_yaw"):
                joint = retargeter.model.joint(f"{side}_{joint_kind}_joint")
                qpos_address = int(joint.qposadr[0])
                maximum_axial_twist = max(
                    maximum_axial_twist,
                    abs(qpos[qpos_address] - retargeter.model.qpos0[qpos_address]),
                )

    assert np.all(np.isfinite(qpos))
    assert maximum_axial_twist < 0.1
    for side in ("left", "right"):
        elbow_error = retargeter.human_body_to_task2[
            f"{side}_elbow"
        ].compute_error(retargeter.configuration)
        wrist_error = retargeter.human_body_to_task2[
            f"{side}_wrist"
        ].compute_error(retargeter.configuration)
        assert np.linalg.norm(elbow_error[:3]) < 0.04
        assert np.linalg.norm(wrist_error[:3]) < 0.015


def test_joint_limit_diagnostics_identify_bound_and_range() -> None:
    model, _ = load_models()
    qpos = np.repeat(model.qpos0[None], 20, axis=0)
    joint = model.joint("right_elbow_yaw_joint")
    qpos[:, joint.qposadr[0]] = joint.range[1]

    report = joint_limit_saturation_metrics(qpos, model)
    saturated = {
        item["joint_name"]: item
        for item in report["saturated_joint_ranges"]
    }
    elbow_yaw = saturated["right_elbow_yaw_joint"]
    assert elbow_yaw["limiting_bound"] == "upper"
    np.testing.assert_allclose(
        elbow_yaw["range_degrees"], [-30.0, 120.0], atol=1e-3
    )
    assert elbow_yaw["near_upper_limit_percent"] == 100.0


def test_bello_velocity_limit_maps_differential_ankle_tendons() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body><freejoint/><geom type="sphere" size=".01"/>
              <body><joint name="left_ankle_pitch_joint"/><geom size=".01"/></body>
              <body><joint name="left_ankle_roll_joint"/><geom size=".01"/></body>
              <body><joint name="right_ankle_pitch_joint"/><geom size=".01"/></body>
              <body><joint name="right_ankle_roll_joint"/><geom size=".01"/></body>
            </body>
          </worldbody>
          <tendon>
            <fixed name="left_ankle_motor_1"><joint joint="left_ankle_pitch_joint" coef="1"/></fixed>
            <fixed name="left_ankle_motor_2"><joint joint="left_ankle_roll_joint" coef="1"/></fixed>
            <fixed name="right_ankle_motor_1"><joint joint="right_ankle_pitch_joint" coef="1"/></fixed>
            <fixed name="right_ankle_motor_2"><joint joint="right_ankle_roll_joint" coef="1"/></fixed>
          </tendon>
          <actuator>
            <position name="left_1" tendon="left_ankle_motor_1"/>
            <position name="left_2" tendon="left_ankle_motor_2"/>
            <position name="right_1" tendon="right_ankle_motor_1"/>
            <position name="right_2" tendon="right_ankle_motor_2"/>
          </actuator>
        </mujoco>
        """
    )
    retargeter = object.__new__(GeneralMotionRetargeting)
    retargeter.model = model
    retargeter.tgt_robot = "bello"

    assert retargeter.velocity_limited_joint_names() == (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )


def test_bello_mapping_does_not_change_unitree_tasks() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=1.66,
        verbose=False,
    )
    assert retargeter.max_iter == 10
    assert not any(
        isinstance(task, mink.PostureTask)
        for task in (*retargeter.tasks1, *retargeter.tasks2)
    )


def test_simulation_collision_matrix_excludes_nearby_links() -> None:
    model, _ = load_models()
    graph = {body_id: set() for body_id in range(1, model.nbody)}
    for body_id in graph:
        parent_id = int(model.body_parentid[body_id])
        if parent_id in graph:
            graph[body_id].add(parent_id)
            graph[parent_id].add(body_id)

    expected_exclusions = set()
    for body1 in graph:
        nearby = set(graph[body1])
        for neighbor in graph[body1]:
            nearby.update(graph[neighbor])
        for body2 in nearby:
            if body1 < body2:
                expected_exclusions.add((body1 << 16) + body2)

    assert set(model.exclude_signature.tolist()) == expected_exclusions
    assert model.nexclude == 78
