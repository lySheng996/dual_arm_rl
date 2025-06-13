# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import sys

# 追加路径到模块搜索路径末尾
sys.path.append('/home/user/sly/dual_arm_drl/dual_arm_drl/source')
import math
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from dual_arm_drl.dual_arm_drl.aseets.config.dual_arm_model import hechuan_dual_arm as DUAL_ARM_CFG  # isort:skip


##
# Scene definition
##


@configclass
class DualArmDrlSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robot
    robot: ArticulationCfg = DUAL_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Humanoid_dual_arm_platform")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##
@configclass
class CommandCfg:
    end_pose = mdp.FileBasedPoseCommandCfg(
        asset_name="robot",
        body_name="right_end",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        pose_data_file="/home/user/sly/arm_pose_target/ee_pose_commands.csv",
        )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointV2PActionCfg(
            asset_name="robot", joint_names=["right_j1","right_j2","right_j3","right_j4","right_j5","right_j6","right_j7"]
        )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.spec_joint_pos_rel,noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "end_pose"})
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_spec_joints_by_uniform,
        mode="reset",
        params={
            "position_ranges":{
            "right_j1": (-np.pi,np.pi),
            "right_j2": (-np.pi,np.pi),
            "right_j3": (-np.pi,np.pi),
            "right_j4": (0,0.05),
            "right_j5": (-np.pi,np.pi),
            "right_j6": (-0.02,0.02),
            "right_j7": (-0.02,0.02),
        }
        },
    )


@configclass
class RewardsCfg:
    """位置偏差."""
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_end"), "command_name": "end_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_end"), "std": 0.1, "command_name": "end_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_end"), "command_name": "end_pose"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )


##
# Environment configuration
##


@configclass
class DualArmDrlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DualArmDrlSceneCfg = DualArmDrlSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 12
        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation

class DualArmDrlEnvCfg_play(DualArmDrlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.replicate_physics=False
        # disable randomization for play
        self.observations.policy.enable_corruption = False