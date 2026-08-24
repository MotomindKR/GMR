from rich import print
from .params import IK_CONFIG_ROOT, ASSET_ROOT, ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from .motion_retarget import GeneralMotionRetargeting
from .offline_retarget import OfflineRetargetResult, retarget_offline_frames
from .quality import evaluate_retargeted_motion
from .robot_motion_viewer import RobotMotionViewer, draw_frame
from .data_loader import load_robot_motion
from .kinematics_model import KinematicsModel

from .neck_retarget import human_head_to_robot_neck

try:
    from .xrobot_utils import XRobotStreamer, XRobotRecorder
except ImportError:
    print("XRobotStreamer is not installed. Please install xrobotoolkit_sdk to use this feature.")
    XRobotStreamer = None
    XRobotRecorder = None

__all__ = [
    "ASSET_ROOT",
    "GeneralMotionRetargeting",
    "IK_CONFIG_DICT",
    "IK_CONFIG_ROOT",
    "KinematicsModel",
    "OfflineRetargetResult",
    "ROBOT_BASE_DICT",
    "ROBOT_XML_DICT",
    "RobotMotionViewer",
    "VIEWER_CAM_DISTANCE_DICT",
    "XRobotRecorder",
    "XRobotStreamer",
    "draw_frame",
    "evaluate_retargeted_motion",
    "human_head_to_robot_neck",
    "load_robot_motion",
    "retarget_offline_frames",
]
