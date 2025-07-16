# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object from a USD file and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object from a USD file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # -- Here is the main change: We are loading the RigidObject from a USD file.

    # Convert the windows-style path to a path Isaac Sim can understand
    # Note: Using os.path.normpath and replacing backslashes is a robust way to handle paths.
    usd_path_raw = r"C:\onedrive\OneDrive - University of Virginia\Desktop\isaac\my_assets_new\lamp_base\lamp_base.usd"
    lighter_usd_path = os.path.normpath(usd_path_raw).replace("\\", "/")

    # Check if the file exists before proceeding
    if not os.path.exists(lighter_usd_path):
        raise FileNotFoundError(f"The specified USD file does not exist: {lighter_usd_path}")


    # Configuration for the lighter object
    lighter_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Lighter",
        # Use UsdFileCfg to specify the asset to load
        spawn=sim_utils.UsdFileCfg(
            usd_path=lighter_usd_path,
            # Define how the object should behave physically
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, # False means it's a dynamic object affected by physics
                disable_gravity=False,
            ),
            # Define the mass properties
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1), # Set a reasonable mass for a lighter
            # Enable collisions
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)), # Start it a bit higher
    )
    # Create the lighter object using the configuration
    lighter_object = RigidObject(cfg=lighter_cfg)

    # return the scene information
    scene_entities = {"lighter": lighter_object}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    lighter_object = entities["lighter"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = lighter_object.data.default_root_state.clone()
            # set the initial position of the lighters based on the origins
            root_state[:, :3] += origins
            # sample a random position on a cylinder around the origins
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 0.5), size=lighter_object.num_instances, device=lighter_object.device
            )
            # write root state to simulation
            lighter_object.write_root_pose_to_sim(root_state[:, :7])
            lighter_object.write_root_velocity_to_sim(root_state[:, 7:])
            # reset buffers
            lighter_object.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # apply sim data
        lighter_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        lighter_object.update(sim_dt)
        # print the root position
        if count % 50 == 0:
            print(f"Lighter root position (in world): {lighter_object.data.root_pos_w}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()