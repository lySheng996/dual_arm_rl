import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .file_pose_command import FileBasedPoseCommand

@configclass
class FileBasedPoseCommandCfg(CommandTermCfg):
    """Configuration for file-based pose command generator."""

    class_type: type = FileBasedPoseCommand  # 注意：后面类名要一致

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""
    pose_data_file: str = MISSING  # <<< 新增字段：CSV 文件路径
    """Path to the CSV file containing pose commands in the format (x, y, z, qw, qx, qy, qz)."""
    make_quat_unique: bool = False

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)