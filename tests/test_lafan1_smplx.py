import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.lafan1_smplx import (
    Y_UP_TO_Z_UP,
    rotation_between_vectors,
    solve_minimal_twist_smpl_rotations,
)


def test_rotation_between_vectors_has_no_parallel_twist() -> None:
    rotation = rotation_between_vectors(
        np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(
        rotation.apply([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-8
    )
    assert np.isclose(rotation.magnitude(), np.pi / 2)


def test_neutral_skeleton_produces_zero_local_twist() -> None:
    rest_joints = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.2, -0.1, 0.0],
            [0.2, -0.1, 0.0],
            [0.0, 0.3, 0.0],
            [-0.2, -0.8, 0.0],
            [0.2, -0.8, 0.0],
            [0.0, 0.7, 0.0],
        ]
    )
    parents = np.array([-1, 0, 0, 0, 1, 2, 3])
    target = Y_UP_TO_Z_UP.apply(rest_joints)[None]
    root_orient, body_pose = solve_minimal_twist_smpl_rotations(
        target, rest_joints, parents
    )
    np.testing.assert_allclose(
        R.from_rotvec(root_orient[0]).as_matrix(),
        Y_UP_TO_Z_UP.as_matrix(),
        atol=1e-8,
    )
    np.testing.assert_allclose(body_pose, 0.0, atol=1e-8)
