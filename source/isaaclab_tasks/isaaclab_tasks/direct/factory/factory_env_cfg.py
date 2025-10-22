# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert, LighterTaskCfg
import os
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.sensors.contact_sensor import ContactSensorCfg

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "lighter_joints": 1,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "envs_mass": 1,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "lighter_joints": 1,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "envs_mass": 1,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = [
        1.5178e-03,
        -1.9651e-01,
        -1.4364e-03,
        -1.9761,
        -2.7717e-04,
        1.7796,
        7.8556e-01,
    ]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=2.0,replicate_physics = True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path_raw = os.path.join(
        current_dir,
        "..",
        "..",
        "..",
        "..",
        "..",
        "my_assets_new",
        "franka_tacsl_5.0",
        "franka_no_ins.usd",
        # "franka.usd",
    )
    #usd_path_raw = r"C:\onedrive\OneDrive - University of Virginia\Desktop\isaac\IsaacLab\my_assets_new\franka_tacsl_correct\franka.usd"
    # usd_path_raw = r"/p/langdiffuse/isaac_lab_xh/IsaacLab/my_assets_new/franka_tacsl_correct/franka.usd"
    robot_usd_path = os.path.normpath(usd_path_raw).replace("\\", "/")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=2.0, clone_in_fabric=False)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
            # usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            )
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )
    # gripper_camera : CameraCfg = CameraCfg(
    #     prim_path="/World/envs/env_0/Robot/panda_hand/Camera/gripper_camera",
    #     update_period=0.0,
    #     height=320,
    #     width=240,
    #     data_types=["rgb", "distance_to_image_plane"],  # 等效于 'depth'
    #     depth_clipping_behavior="none",  # (near_plane, far_plane)
    #     # offset=CameraCfg.OffsetCfg(
    #     #     pos=(0.0, 0.0, -0.02034),  # camera_dist
    #     #     rot=(0.0, 0.707, 0.707, 0.0),  # 将欧拉角 [pi/2, -pi/2, 0] 转换为四元数
    #     # ),
    #     offset=CameraCfg.OffsetCfg(
    #             pos=(0.13, 0.0, -0.15), rot=(0.7061377, -0.0370072, -0.0370072, 0.7061377), convention="ros"
    #             # pos=(0.13, 0.0, -0.15), rot=(0.247404, 0, 0, 0.9689124), convention="ros"
    #         ),
    #     spawn=sim_utils.PinholeCameraCfg(),
    #     debug_vis=True,
    #     update_latest_camera_pose = True
    # )

    # tactile_camera : CameraCfg = CameraCfg(
    #     prim_path="/World/envs/env_0/Robot/elastomer_tip_right/Camera/tactile_camera",
    #     update_period=0.0,
    #     height=320,
    #     width=240,
    #     data_types=["rgb", "distance_to_image_plane"],  # 等效于 'depth'
    #     depth_clipping_behavior="zero",  # (near_plane, far_plane)
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.0, -0.020342857142857145), rot=(0, 0, 0, 1), convention="ros"
    #         # pos=(0.0, 0.0, -0.021), rot=(0, 0, 0, 1), convention="ros"
    #     ),
    #     # offset=CameraCfg.OffsetCfg(
    #     #         pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
    #     #     ),
    #     spawn=sim_utils.PinholeCameraCfg(),
    #     debug_vis=True,
    #     update_latest_camera_pose = True
    # )

class LighterEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 29
    state_space = 73
    enable_global_camera = False
    enable_gripper_camera = False
    enable_tactile_camera = False
    enable_tactile = False
    mass_range = [0.01, 0.001]
    obs_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "lighter_joints",
        "fixed_pos",
        "fixed_quat",
        "envs_mass",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "lighter_joints",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "envs_mass",
    ]

    task_name: str = "lighter"  # peg_insert, gear_mesh, nut_thread
    task: LighterTaskCfg = LighterTaskCfg()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))
    sim: SimulationCfg = SimulationCfg(
        render_interval=1,
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),

    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=2.0, clone_in_fabric=False , replicate_physics = True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path_raw = os.path.join(
        current_dir,
        "..",
        "..",
        "..",
        "..",
        "..",
        "my_assets_new",
        "franka_tacsl_5.0",
        "franka_no_ins.usd",
        # "franka.usd",
    )
    robot_usd_path = os.path.normpath(usd_path_raw).replace("\\", "/")

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
            # usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            )
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=0.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    # elastomer_contact: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/franka_bak/elastomer_right",
    #     filter_prim_paths_expr=["/World/envs/env_.*/lighter/link_2"],
    #     history_length=3,
    # )
    elastomer_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/franka_bak/elastomer_right",
        filter_prim_paths_expr=["/World/envs/env_.*/lighter/link_2"],
        history_length=3,
    )
    gripper_camera : TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/franka_bak/panda_hand/Camera/gripper_camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],  # 等效于 'depth'
        depth_clipping_behavior="none",  # (near_plane, far_plane)
        # offset=CameraCfg.OffsetCfg(
        #     pos=(0.0, 0.0, -0.02034),  # camera_dist
        #     rot=(0.0, 0.707, 0.707, 0.0),  # 将欧拉角 [pi/2, -pi/2, 0] 转换为四元数
        # ),
        offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(0.7061377, -0.0370072, -0.0370072, 0.7061377), convention="ros"
                # pos=(0.13, 0.0, -0.15), rot=(0.247404, 0, 0, 0.9689124), convention="ros"
            ),
        spawn=sim_utils.PinholeCameraCfg(),
        debug_vis=True,
        update_latest_camera_pose = True
    )

    tactile_camera : TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/franka_bak/elastomer_tip_right/Camera/tactile_camera",
        update_period=0.0,
        height=320,
        width=240,
        data_types=["distance_to_image_plane"],  # 等效于 'depth'
        depth_clipping_behavior="zero",  # (near_plane, far_plane)
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.020342857142857145), rot=(0, 0, 0, 1), convention="ros"
            # pos=(0.0, 0.0, -0.021), rot=(0, 0, 0, 1), convention="ros"
        ),
        # offset=CameraCfg.OffsetCfg(
        #         pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        #     ),
        spawn=sim_utils.PinholeCameraCfg(),
        debug_vis=True,
        update_latest_camera_pose = True
    )

    global_camera: TiledCameraCfg = TiledCameraCfg(
        # 注意：不挂在 Robot 下，而是挂在每个 env 自己的 Cameras 目录
        prim_path="/World/envs/env_.*/Robot/franka_bak/Camera/global_camera",
        update_period=0.2,
        height=200,
        width=300,
        data_types=["rgb", "distance_to_image_plane"],  # 深度同等价
        depth_clipping_behavior="none",
        offset=CameraCfg.OffsetCfg(
            pos=(1.2, -0.5, 0.5),
            # rot=( 0.73254,0.46194, 0.19134, 0.46194),
            rot=(-0.46194, 0.73254, 0.46194, -0.19134),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(),
        debug_vis=True,
        # 设为 False：不去“更新到最新的跟随姿态”，确保纯静止
        update_latest_camera_pose=False,
    )

    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = None
    # (2) Terminate if the robot falls
    object_reached_goal = None


@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0
    terminations = TerminationsCfg(
        time_out=None,
        object_reached_goal=None,
    )


@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0
    terminations = TerminationsCfg(
        time_out=None,
        object_reached_goal=None,
    )


@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0

@configclass
class FactoryTaskLighterCfg(LighterEnvCfg):
    task_name = "lighter"
    task = LighterTaskCfg()
    episode_length_s = 10.0
    terminations = TerminationsCfg(
        time_out=None,
        object_reached_goal=None,
    )
