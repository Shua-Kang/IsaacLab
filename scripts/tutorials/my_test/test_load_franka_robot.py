# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create an articulated robot from a USD file,
make it grasp a cube, and lift it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p <path_to_this_script>.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and controlling an articulated robot from a USD file.")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.actuators import ImplicitActuatorCfg

def design_scene():
    """Designs the scene by adding a ground plane, light, the Franka robot, and a cube."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # -- Robot Configuration --
    usd_path_raw = r"C:\onedrive\OneDrive - University of Virginia\Desktop\isaac\IsaacLab\my_assets_new\franka_tacsl_correct\franka.usd"
    robot_usd_path = os.path.normpath(usd_path_raw).replace("\\", "/")
    if not os.path.exists(robot_usd_path):
        raise FileNotFoundError(f"The specified USD file does not exist: {robot_usd_path}")

    franka_cfg = ArticulationCfg(
        prim_path="/World/Franka",
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot_usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0, "panda_joint2": -0.55, "panda_joint3": 0.0,
                "panda_joint4": -2.5, "panda_joint5": 0.0, "panda_joint6": 2.0,
                "panda_joint7": 0.75, "panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr="panda_joint[1-7]", stiffness=800.0, damping=40.0),
            "gripper": ImplicitActuatorCfg(joint_names_expr="panda_finger_joint[1-2]", stiffness=500.0, damping=10.0),
        },
    )
    robot = Articulation(cfg=franka_cfg)

    # -- Cube Configuration --
    # Add a cube for the robot to grasp
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05), # 5cm cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05), # 50g
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.025) # Position it in front of the robot
        )
    )
    cube = RigidObject(cfg=cube_cfg)


    # Return the scene information
    scene_entities = {"robot": robot, "cube": cube}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = entities["robot"]
    cube = entities["cube"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # --- Define states for the pick-and-place motion ---
    
    # Get the default joint positions from the initial state
    default_joint_pos = robot.data.default_joint_pos.clone()
    
    # Get actuator indices
    arm_joint_indices = robot.actuators["arm"].joint_indices
    gripper_joint_indices = robot.actuators["gripper"].joint_indices
    
    # Define joint configurations for each stage of the grasp
    # These are target joint angles that move the gripper to the desired poses.
    # You may need to tune these values for your specific gripper.
    # Pose 1: Pre-grasp (above the cube)
    pre_grasp_pos = torch.tensor([[0.0, 0.1, 0.0, -1.8, 0.0, 2.0, 0.75]], device=sim.device)
    # Pose 2: Grasp (at the cube level)
    grasp_pos = torch.tensor([[0.0, 0.6, 0.0, -1.2, 0.0, 1.8, 0.75]], device=sim.device)
    # Pose 3: Lift (up with the cube)
    lift_pos = torch.tensor([[0.0, 0.4, 0.0, -1.5, 0.0, 1.9, 0.75]], device=sim.device)
    
    # Define gripper states
    open_gripper_pos = 0.04
    closed_gripper_pos = 0.0
    
    # Simple state machine
    # 0: Reset, 1: Move to pre-grasp, 2: Lower to grasp, 3: Close gripper, 4: Lift, 5: Done
    state = 0
    
    # Simulate physics
    while simulation_app.is_running():
        # --- State Machine Logic ---
        target_joint_pos = robot.data.joint_pos.clone() # Start with current position
        
        if state == 0: # Reset state
            # Set arm to pre-grasp pose and open gripper
            target_joint_pos[:, arm_joint_indices] = pre_grasp_pos
            target_joint_pos[:, gripper_joint_indices] = open_gripper_pos
            # Reset cube to its initial position
            cube.write_root_pose_to_sim(cube.data.default_root_state[:, :7])
            cube.write_root_velocity_to_sim(cube.data.default_root_state[:, 7:])
            
            # Go to next state after a short delay
            if count > 50:
                state = 1
                print("[INFO] State -> 1: Moving to pre-grasp pose.")

        elif state == 1: # Move to pre-grasp
            target_joint_pos[:, arm_joint_indices] = pre_grasp_pos
            target_joint_pos[:, gripper_joint_indices] = open_gripper_pos
            # Check if arm is close to target
            if torch.allclose(robot.data.joint_pos[:, arm_joint_indices], pre_grasp_pos, atol=0.1):
                state = 2
                print("[INFO] State -> 2: Lowering to grasp pose.")

        elif state == 2: # Lower to grasp
            target_joint_pos[:, arm_joint_indices] = grasp_pos
            # Check if arm is close to target
            if torch.allclose(robot.data.joint_pos[:, arm_joint_indices], grasp_pos, atol=0.1):
                state = 3
                print("[INFO] State -> 3: Closing gripper.")

        elif state == 3: # Close gripper
            target_joint_pos[:, gripper_joint_indices] = closed_gripper_pos
            # Keep arm at grasp pose
            target_joint_pos[:, arm_joint_indices] = grasp_pos
            # Wait for gripper to close
            current_gripper_pos = robot.data.joint_pos[0, gripper_joint_indices[0]]
            if abs(current_gripper_pos - closed_gripper_pos) < 0.005:
                state = 4
                print("[INFO] State -> 4: Lifting cube.")

        elif state == 4: # Lift
            target_joint_pos[:, arm_joint_indices] = lift_pos
            # Keep gripper closed
            target_joint_pos[:, gripper_joint_indices] = closed_gripper_pos
            # Check if arm is close to target
            if torch.allclose(robot.data.joint_pos[:, arm_joint_indices], lift_pos, atol=0.1):
                state = 5
                print("[INFO] State -> 5: Task complete. Resetting in 3 seconds.")
                count = -150 # Set count for a 3-second pause before reset

        elif state == 5: # Done
            # Hold position until reset
            target_joint_pos[:, arm_joint_indices] = lift_pos
            target_joint_pos[:, gripper_joint_indices] = closed_gripper_pos
            if count >= 0:
                state = 0
                print("[INFO] Resetting task.")

        # Set the command for the robot
        robot.set_joint_position_target(target_joint_pos)
        
        # FIX: Write commands to the simulation BEFORE stepping the physics
        robot.write_data_to_sim()
        
        # Perform simulation step
        sim.step()
        count += 1
        
        # Update buffers from the simulation
        robot.update(sim_dt)
        cube.update(sim_dt)

def main():
    """Main function."""
    # Create a simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera view
    sim.set_camera_view(eye=[0.8, 0.8, 0.7], target=[0.4, 0.0, 0.4])
    
    # Design the scene
    scene_entities = design_scene()
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # Run the main function
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close sim app
        simulation_app.close()
