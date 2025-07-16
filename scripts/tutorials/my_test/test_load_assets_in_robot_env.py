# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator
and load custom assets into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller with custom assets.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

# -- Define and check asset paths globally --
# This logic is moved outside the scene config class.
lamp_base_path_raw = r"C:\onedrive\OneDrive - University of Virginia\Desktop\isaac\my_assets_new\lamp_base\lamp_base.usd"
lamp_bulb_path_raw = r"C:\onedrive\OneDrive - University of Virginia\Desktop\isaac\my_assets_new\lamp_bulb\lamp_bulb.usd"

# Convert paths for Isaac Sim and check existence
lamp_base_usd_path = os.path.normpath(lamp_base_path_raw).replace("\\", "/")
if not os.path.exists(lamp_base_usd_path):
    raise FileNotFoundError(f"File not found: {lamp_base_usd_path}")

lamp_bulb_usd_path = os.path.normpath(lamp_bulb_path_raw).replace("\\", "/")
if not os.path.exists(lamp_bulb_usd_path):
    raise FileNotFoundError(f"File not found: {lamp_bulb_usd_path}")


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a table-top scene with custom assets."""

    # -- Static Environment Assets --
    # Use AssetBaseCfg for static elements that are not part of the direct interaction.
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
        ),
    )

    # -- Dynamic/Interactive Assets --
    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    # Custom Assets as Rigid Objects
    # These now refer to the globally defined path variables.
    lamp_base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LampBase",
        spawn=sim_utils.UsdFileCfg(
            usd_path=lamp_base_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),  # Assign a mass
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.0)),  # Position on the table
    )

    lamp_bulb = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LampBulb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=lamp_bulb_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # Assign a small mass
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.15)),  # Position slightly above the base
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]
    lamp_base = scene["lamp_base"]
    lamp_bulb = scene["lamp_bulb"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.5, 0.3, 0.5, 0.707, 0, 0.707, 0],
        [0.5, -0.3, 0.5, 0.707, 0.707, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    robot_entity_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 300 == 0:
            # reset time and counters
            count = 0
            # reset robot state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset object state
            lamp_base.reset()
            lamp_bulb.reset()
            # reset controller
            current_goal_idx = 0
            ik_commands[:] = ee_goals[current_goal_idx]
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            print("----------------------------------------")
            print("[INFO]: Resetting robot and objects.")

        # change goal
        if count % 100 == 0 and count > 0:
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            ik_commands[:] = ee_goals[current_goal_idx]
            diff_ik_controller.set_command(ik_commands)

        # obtain quantities from simulation
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # compute the joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # update marker positions
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera to view the new objects
    sim.set_camera_view([2.0, 2.0, 1.8], [0.5, 0.0, 1.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()