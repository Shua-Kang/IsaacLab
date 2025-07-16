# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the first step in creating a penalty-based tactile sensor
simulation in Isaac Lab using pysdf.

Step 1: Setting up the scene with a sphere and a cube.

.. code-block:: bash

    # Usage
    # Note: This script requires the pysdf and trimesh packages.
    # You can install them using pip:
    # ./isaaclab.sh -p --pip-install pysdf trimesh

    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on setting up a tactile simulation scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import trimesh
from pysdf import SDF

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
import isaaclab.utils.math as math_utils


class TactileSystem:
    """
    A class to manage the tactile sensing simulation between a sensor (cube) and an object (sphere).
    """

    def __init__(self, sim: sim_utils.SimulationContext):
        """
        Initializes the tactile system by creating the sensor and the object.
        """
        self.sim = sim
        self.cfg = sim.cfg

        # create the cube and sphere
        self._create_objects()

    def _create_objects(self):
        """Creates the cube (sensor) and the sphere (object to be sensed)."""
        # Cube configuration (our "tactile sensor")
        cube_cfg = RigidObjectCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
        self.cube = RigidObject(cfg=cube_cfg)

        # Sphere configuration (the object to be "felt")
        sphere_cfg = RigidObjectCfg(
            prim_path="/World/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=0.1,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
        self.sphere = RigidObject(cfg=sphere_cfg)

    def get_scene_entities(self):
        """Returns the scene entities created by this system."""
        return {"cube": self.cube, "sphere": self.sphere}


def design_scene(sim: sim_utils.SimulationContext):
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Tactile System
    tactile_system = TactileSystem(sim)

    # return the scene information
    return tactile_system.get_scene_entities()


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    cube = entities["cube"]
    sphere = entities["sphere"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state for cube
            root_state_cube = cube.data.default_root_state.clone()
            root_state_cube[:, 2] = 0.2
            cube.write_root_state_to_sim(root_state_cube)

            # reset root state for sphere
            root_state_sphere = sphere.data.default_root_state.clone()
            root_state_sphere[:, 2] = 0.5
            sphere.write_root_state_to_sim(root_state_sphere)

            # reset buffers
            cube.reset()
            sphere.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object states...")

        # apply actions (we will move the sphere down to touch the cube)
        if count > 100:
            # get current state of the sphere
            sphere_state = sphere.data.root_state_w.clone()
            # apply a downward force/velocity change
            sphere_state[:, 8] = -0.5 # vz = -0.5 m/s
            # write new state to simulation
            sphere.write_root_velocity_to_sim(sphere_state[:, 7:])


        # apply sim data
        cube.write_data_to_sim()
        sphere.write_data_to_sim()

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cube.update(sim_dt)
        sphere.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    # Note: pysdf works on CPU, so we ensure tensors are created on the correct device later.
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.2])
    # Design scene
    scene_entities = design_scene(sim)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # close sim app
        simulation_app.close()

