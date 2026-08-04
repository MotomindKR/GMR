from pathlib import Path

import mujoco
import numpy as np

from general_motion_retargeting.params import ROBOT_XML_DICT


RIGHT_ARM_LIMITS = {
    "right_shoulder_pitch_joint": (-1.57, 1.57),
    "right_shoulder_roll_joint": (-1.57, 1.57),
    "right_shoulder_yaw_joint": (-1.57, 1.57),
    "right_elbow_pitch_joint": (-1.57, 1.57),
    "right_elbow_yaw_joint": (-0.523599, 2.0944),
    "right_wrist_pitch_joint": (-1.57, 1.57),
}


def test_bello_models_preserve_remote_right_arm_limits() -> None:
    viewer_path = Path(ROBOT_XML_DICT["bello"])
    model_paths = (viewer_path, viewer_path.with_name("bello_full_body_boxes.xml"))

    for model_path in model_paths:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        for joint_name, expected in RIGHT_ARM_LIMITS.items():
            joint = model.joint(joint_name)
            actuator = model.actuator(f"{joint_name}_pos")
            np.testing.assert_allclose(joint.range, expected)
            np.testing.assert_allclose(actuator.ctrlrange, expected)
