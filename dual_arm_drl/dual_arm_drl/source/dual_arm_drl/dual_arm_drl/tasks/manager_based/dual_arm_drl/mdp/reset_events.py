from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_spec_joints_by_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_ranges: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = list(position_ranges.keys())
    _joint_ids, _joint_names = asset.find_joints(joint_names)
    joint_pos_radom = asset.data.default_joint_pos[:,_joint_ids].clone()
    #print("joint_pos_radom shape:", joint_pos_radom.shape)
    joint_pos_send = asset.data.default_joint_pos[:,_joint_ids].clone()
    # get default joint state
    for i, joint_name in enumerate(joint_names):
        if joint_name in position_ranges:
            pos_range = position_ranges[joint_name]
            joint_pos_radom[:, i] += math_utils.sample_uniform(*pos_range, joint_pos_radom[:, i].shape, joint_pos_radom.device)
    # clamp joint pos to limits
    for i in range(5):
        joint_pos_radom=joint_pos_radom/5
        joint_pos_send += joint_pos_radom
        asset.write_joint_position_to_sim(joint_pos_send,_joint_ids)
    # set into the physics simulation
   
