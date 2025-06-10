from dataclasses import MISSING
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def spec_joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    joint_names=["right_j1","right_j2","right_j3","right_j4","right_j5","right_j6","right_j7"]
    _asset: Articulation = env.scene[asset_cfg.name]
    asset_cfg.joint_ids
    _joint_ids, _joint_names = _asset.find_joints(joint_names)
    return _asset.data.joint_pos[:, _joint_ids] - _asset.data.default_joint_pos[:,  _joint_ids]