import json
from pathlib import Path

import mink
import mujoco
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
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


def test_box_model_uses_horizontal_primitive_soles() -> None:
    _, model = load_models()
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
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

    expected_arm_weights = {
        "l_upper_arm_link": ((0, 10), (0, 5)),
        "l_elbow_link": ((0, 10), (50, 5)),
        "l_wrist_link": ((0, 2), (40, 1)),
        "r_upper_arm_link": ((0, 10), (0, 5)),
        "r_elbow_link": ((0, 10), (50, 5)),
        "r_wrist_link": ((0, 2), (40, 1)),
    }
    for body_name, (stage1, stage2) in expected_arm_weights.items():
        assert tuple(table1[body_name][1:3]) == stage1
        assert tuple(table2[body_name][1:3]) == stage2

    assert table1["left_ankle_roll_link"][1:3] == [100, [20, 0, 20]]
    assert table1["right_ankle_roll_link"][1:3] == [100, [20, 0, 20]]
    assert table2["left_ankle_roll_link"][1:3] == [100, [20, 0, 20]]
    assert table2["right_ankle_roll_link"][1:3] == [100, [20, 0, 20]]
    assert tuple(table2["left_hip_roll_link"][1:3]) == (20, 0)
    assert tuple(table2["right_hip_roll_link"][1:3]) == (20, 0)
    assert tuple(table1["bello_root"][1:3]) == (100, 10)
    assert tuple(table2["bello_root"][1:3]) == (100, 5)
    assert tuple(table1["left_knee_link"][1:3]) == (0, 10)
    assert tuple(table1["right_knee_link"][1:3]) == (0, 10)
    assert tuple(table2["left_knee_link"][1:3]) == (10, 5)
    assert tuple(table2["right_knee_link"][1:3]) == (10, 5)
    expected_knee_offsets = {
        "left_knee_link": np.array(
            [0.500926591475, 0.489716132629, 0.534047445216, 0.473332848696]
        ),
        "right_knee_link": np.array(
            [-0.395393636381, -0.601919070832, -0.605627097806, -0.33848651802]
        ),
    }
    for body_name, expected_offset in expected_knee_offsets.items():
        np.testing.assert_allclose(table1[body_name][4], expected_offset)
        np.testing.assert_allclose(table2[body_name][4], expected_offset)
    assert config["ground_alignment_axes"] == {
        "left_foot": [0.0, 1.0, 0.0],
        "right_foot": [0.0, -1.0, 0.0],
    }
    assert config["ground_clearance_geoms"] == [
        "left_ankle_roll_link_collision_box_1",
        "right_ankle_roll_link_collision_box_1",
    ]

    forbidden_solver_keys = {
        "collision_avoidance",
        "fixed_iterations",
        "joint_acceleration_limits",
        "joint_position_limits",
        "max_joint_velocity",
        "posture_costs",
        "posture_targets",
        "use_velocity_limit",
    }
    assert forbidden_solver_keys.isdisjoint(config)


def test_bello_ankle_roll_range_requires_bello_specific_lower_body_mapping() -> None:
    model, _ = load_models()
    expected_range = np.array([-0.0873, 0.0873])
    for side in ("left", "right"):
        np.testing.assert_allclose(
            model.joint(f"{side}_ankle_roll_joint").range,
            expected_range,
            atol=1e-7,
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
            np.abs(
                retargeter.configuration.data.geom_xmat[geom_id].reshape(3, 3)[2]
            )
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
    assert len(retargeter.ik_limits) == 1
    assert isinstance(retargeter.ik_limits[0], mink.ConfigurationLimit)
    assert not any(
        isinstance(task, mink.PostureTask)
        for task in (*retargeter.tasks1, *retargeter.tasks2)
    )
    assert not hasattr(retargeter, "previous_output_qpos")
    assert not hasattr(retargeter, "fixed_iterations")

    for side in ("left", "right"):
        assert f"{side}_hip" in retargeter.rot_offsets1
        assert f"{side}_knee" in retargeter.rot_offsets1
        assert f"{side}_hip" not in retargeter.human_body_to_task1
        assert f"{side}_knee" in retargeter.human_body_to_task1


def test_velocity_limit_is_an_explicit_shared_solver_option() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
        use_velocity_limit=True,
    )
    assert len(retargeter.ik_limits) == 2
    assert isinstance(retargeter.ik_limits[0], mink.ConfigurationLimit)
    assert isinstance(retargeter.ik_limits[1], mink.VelocityLimit)


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


def test_simulation_collision_matrix_is_not_an_ik_constraint() -> None:
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
