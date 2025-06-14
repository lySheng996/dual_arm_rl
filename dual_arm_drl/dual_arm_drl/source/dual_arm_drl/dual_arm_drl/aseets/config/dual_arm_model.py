import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os


usd_dir_path = "/home/user/sly/arm_usd/"
robot_usd = "Humanoid_dual_arm_platform.usd"

hechuan_dual_arm = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_dir_path + robot_usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link = True
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "right_j1": 0.0,
            "right_j2": 0.0,
            "right_j3": 0.0,
            "right_j4": 0.0,
            "right_j5": 0.0,
            "right_j6": 0.0,
            "right_j7": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["right_j1","right_j2","right_j3","right_j4","right_j5","right_j6","right_j7"],
            velocity_limit=1,
            effort_limit={
                "right_j1": 100,
                "right_j2": 100,
                "right_j3": 100,
                "right_j4": 400,
                "right_j5": 100,
                "right_j6": 200,
                "right_j7": 200,
            },
            stiffness={
                "right_j1": 200,
                "right_j2": 200,
                "right_j3": 200,
                "right_j4": 400,
                "right_j5": 200,
                "right_j6": 300,
                "right_j7": 300,
            },
            damping={
                "right_j1": 20,
                "right_j2": 20,
                "right_j3": 20,
                "right_j4": 40,
                "right_j5": 20,
                "right_j6": 30,
                "right_j7": 30,
            },
        ),
    },
)