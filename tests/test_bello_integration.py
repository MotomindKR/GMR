import json
from pathlib import Path

import mink
import mujoco
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT


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

EXPECTED_POSTURE_COSTS = {
    "left_shoulder_pitch_joint": 1.0,
    "right_shoulder_pitch_joint": 1.0,
    "left_shoulder_roll_joint": 2.0,
    "right_shoulder_roll_joint": 2.0,
    "left_shoulder_yaw_joint": 3.0,
    "right_shoulder_yaw_joint": 3.0,
    "left_elbow_pitch_joint": 1.0,
    "right_elbow_pitch_joint": 1.0,
    "left_elbow_yaw_joint": 2.0,
    "right_elbow_yaw_joint": 2.0,
    "left_wrist_pitch_joint": 10.0,
    "right_wrist_pitch_joint": 10.0,
}


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
        assert sum(
            model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
            for geom_id in foot_geom_ids
        ) == 3
        assert sum(
            model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
            for geom_id in foot_geom_ids
        ) == 2

        sole_id = model.geom(f"{prefix}box_1").id
        sole_normal = data.geom_xmat[sole_id].reshape(3, 3)[:, 2]
        assert abs(float(sole_normal[2])) > 0.9999


def test_bello_config_is_symmetric_and_references_model() -> None:
    model, _ = load_models()
    config = json.loads(Path(IK_CONFIG_DICT["smplx"]["bello"]).read_text())
    scales = config["human_scale_table"]
    for landmark in ("hip", "knee", "foot", "shoulder", "elbow", "wrist"):
        assert scales[f"left_{landmark}"] == scales[f"right_{landmark}"]

    for table_name in ("ik_match_table1", "ik_match_table2"):
        for body_name in config[table_name]:
            assert model.body(body_name).id >= 0
    for body_name, entry in config["ik_match_table2"].items():
        if any(token in body_name for token in ("shoulder", "elbow", "wrist")):
            assert entry[2] == 0

    assert config["collision_avoidance"]["use_model_contact_matrix"] is True
    assert config["max_joint_velocity"] == 3.0
    assert config["max_iter"] == 3
    assert config["joint_limits"] == {
        "left_knee_joint": [0.3, 2.0944],
        "right_knee_joint": [0.3, 2.0944],
    }
    assert config["flat_orientation_geoms"] == {
        "left_foot": "left_ankle_roll_link_collision_box_1",
        "right_foot": "right_ankle_roll_link_collision_box_1",
    }
    assert scales["pelvis"] == 0.95
    assert scales["left_foot"] == scales["right_foot"] == 0.95
    assert config["posture_costs"] == EXPECTED_POSTURE_COSTS


def test_bello_uses_shared_retargeter() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
        use_velocity_limit=True,
    )
    assert type(retargeter) is GeneralMotionRetargeting
    for side in ("left", "right"):
        knee = retargeter.model.joint(f"{side}_knee_joint")
        np.testing.assert_allclose(knee.range, (0.3, 2.0944))
        assert retargeter.configuration.data.qpos[knee.qposadr[0]] == 0.3

        geom_name = retargeter.flat_orientation_geoms[f"{side}_foot"]
        body_quat = retargeter.flatten_geom_orientation(
            np.asarray((1.0, 0.0, 0.0, 0.0)), geom_name
        )
        body_to_geom = retargeter.model.geom(geom_name).quat
        body_rotation = mink.SO3(body_quat).as_matrix()
        geom_rotation = mink.SO3(body_to_geom).as_matrix()
        np.testing.assert_allclose(
            (body_rotation @ geom_rotation)[:, 2],
            (0.0, 0.0, 1.0),
            atol=1e-12,
        )

    posture_tasks = [
        task for task in retargeter.tasks1 if isinstance(task, mink.PostureTask)
    ]
    assert len(posture_tasks) == 1
    posture_cost = posture_tasks[0].cost
    expected_cost = np.zeros(retargeter.model.nv)
    for joint_name, cost in EXPECTED_POSTURE_COSTS.items():
        dof_address = retargeter.model.joint(joint_name).dofadr[0]
        expected_cost[dof_address] = cost
    np.testing.assert_array_equal(posture_cost, expected_cost)

    initial = retargeter.configuration.data.qpos.copy()
    retargeter.previous_output_qpos = initial.copy()
    candidate = initial.copy()
    candidate[retargeter.velocity_limited_qpos_addresses] += 1.0
    limited = retargeter.limit_output_velocity(candidate)
    max_delta = retargeter.max_joint_velocity / retargeter.source_fps
    np.testing.assert_allclose(
        limited[retargeter.velocity_limited_qpos_addresses]
        - initial[retargeter.velocity_limited_qpos_addresses],
        max_delta,
    )


def test_bello_posture_regularizer_does_not_change_unitree_tasks() -> None:
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=1.66,
        verbose=False,
    )
    assert not any(
        isinstance(task, mink.PostureTask)
        for task in (*retargeter.tasks1, *retargeter.tasks2)
    )


def test_collision_matrix_matches_depth_two_body_neighborhood() -> None:
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

    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=1.66,
        verbose=False,
    )
    collision_limit = retargeter.ik_limits[-1]
    ik_pairs = {
        frozenset(pair) for pair in collision_limit.geom_id_pairs
    }
    assert len(ik_pairs) == len(collision_limit.geom_id_pairs) == 669
    for pair in ik_pairs:
        geom1, geom2 = pair
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        signature = (min(body1, body2) << 16) + max(body1, body2)
        assert signature not in expected_exclusions
