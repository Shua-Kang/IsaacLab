# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an alternative, single-process, non-blocking visualization
method using matplotlib.

WARNING: This approach is NOT recommended as it is highly unstable and can lead to
         fatal Python errors due to conflicts between Isaac Sim's and Matplotlib's
         event loops. The multiprocessing approach from the previous step is the
         correct and stable solution.

.. code-block:: bash

    # Usage
    # Note: This script requires pysdf, trimesh, and matplotlib packages.
    # You can install them using pip:
    # ./isaaclab.sh -p --pip-install pysdf trimesh matplotlib

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
import isaaclab.utils.math as math_utils
from isaacsim.core.utils.torch.transformations import tf_apply, tf_inverse


class MatplotlibVisualizer:
    """
    一个使用Matplotlib进行非阻塞式3D可视化的类（单进程，不稳定）。
    """
    def __init__(self):
        plt.ion()  # 开启交互模式以实现非阻塞绘图
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Tactile Force Field')
        # 设置固定的坐标轴范围以稳定视角
        self.ax.set_xlim([-0.15, 0.15])
        self.ax.set_ylim([-0.15, 0.15])
        self.ax.set_zlim([0.2, 0.4])
        # 首次显示图形
        plt.show()


    def update(self, contact_points_w, force_vectors_w):
        """更新并重新绘制3D力场图。"""
        try:
            self.ax.cla()  # 清除旧的箭头
            self.ax.set_xlim([-0.15, 0.15])
            self.ax.set_ylim([-0.15, 0.15])
            self.ax.set_zlim([0.2, 0.4])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('Tactile Force Field')

            if contact_points_w.shape[0] > 0:
                p = contact_points_w
                f = force_vectors_w
                # self.ax.quiver(p[:, 0], p[:, 1], p[:, 2],  # 箭头的起始点
                #                f[:, 0], f[:, 1], f[:, 2],  # 箭头的方向向量
                #                length=0.05, normalize=True, color='r')
                self.ax.quiver(p[:, 0], p[:, 1], p[:, 2],  # 箭头的起始点
                               f[:, 0], f[:, 1], f[:, 2],  # 箭头的方向向量
                               length=0.05, normalize=True, color='r')
            # 短暂暂停以允许绘图更新。这是可能导致崩溃的地方。
            plt.pause(0.0001) 
        except Exception as e:
            # 如果窗口被用户关闭，matplotlib会抛出异常
            print(f"[Visualizer] Matplotlib window may have been closed. Error: {e}")
            return False # 返回信号表示可视化已停止
        return True

    def close(self):
        """关闭matplotlib窗口。"""
        plt.close(self.fig)


class TactileSystem:
    """
    一个管理传感器（立方体）和物体（球体）之间触觉模拟的类。
    """

    def __init__(self, sim: sim_utils.SimulationContext):
        """
        通过创建传感器和物体来初始化触觉系统。
        """
        self.sim = sim
        self.cfg = sim.cfg
        self.device = sim.device

        # 创建立方体和球体
        self._create_objects()

        # 在立方体顶面生成触觉点
        self._generate_tactile_points(num_rows=15, num_cols=15)

        # 为球体初始化SDF
        self._initialize_sdf()

        self.tactile_kn = 800.0  # 法向刚度 (N/m)
        self.enable_visualization = True

        # -- MODIFICATION: Use single-process Matplotlib visualizer --
        if self.enable_visualization:
            self.visualizer = MatplotlibVisualizer()

    def _create_objects(self):
        """创建立方体（传感器）和球体（被感知的物体）。"""
        self.cube_cfg = RigidObjectCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.4, 0.4, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
        self.cube = RigidObject(cfg=self.cube_cfg)

        self.sphere_cfg = RigidObjectCfg(
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
        self.sphere = RigidObject(cfg=self.sphere_cfg)

    def _generate_tactile_points(self, num_rows: int, num_cols: int):
        """在立方体传感器的顶面生成一个触觉点网格。"""
        print("[INFO] Generating tactile points...")
        sensor_size = self.cube_cfg.spawn.size
        top_face_z = sensor_size[2] / 2.0 + 0.02
        x = torch.linspace(-sensor_size[0] / 2.0, sensor_size[0] / 2.0, num_cols, device=self.device)
        y = torch.linspace(-sensor_size[1] / 2.0, sensor_size[1] / 2.0, num_rows, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), top_face_z)], dim=-1)
        self.tactile_points_local = points.unsqueeze(0)
        self.num_tactile_points = self.tactile_points_local.shape[1]
        print(f"[INFO] Generated {self.num_tactile_points} tactile points.")

    def _initialize_sdf(self):
        """使用trimesh和pysdf为球体创建一个SDF对象。"""
        print("[INFO] Initializing SDF for the sphere...")
        radius = self.sphere_cfg.spawn.radius
        mesh = trimesh.primitives.Sphere(radius=radius, subdivisions=4)
        self.sphere_sdf = SDF(mesh.vertices, mesh.faces)
        print("[INFO] SDF initialized.")

    def update(self):
        """在每个仿真步骤中被调用，以计算并施加触觉力。"""
        sphere_pos_w = self.sphere.data.root_pos_w
        sphere_quat_w = self.sphere.data.root_quat_w
        cube_pos_w = self.cube.data.root_pos_w
        cube_quat_w = self.cube.data.root_quat_w

        tactile_points_world = tf_apply(cube_quat_w, cube_pos_w, self.tactile_points_local)
        sphere_pose_inv = tf_inverse(sphere_quat_w, sphere_pos_w)
        tactile_points_sphere_local = tf_apply(sphere_pose_inv[0], sphere_pose_inv[1], tactile_points_world)

        points_np = tactile_points_sphere_local.cpu().numpy().squeeze(0)
        distances_np = self.sphere_sdf(points_np)

        penetration_depth_np = -np.minimum(-distances_np, 0)
        penetration_depth = torch.from_numpy(penetration_depth_np).to(self.device)

        contact_mask = penetration_depth > 0
        
        if self.enable_visualization:
            contact_points_world_viz = tactile_points_world.squeeze(0)[contact_mask]
            force_vectors_viz = torch.zeros_like(contact_points_world_viz)
            if torch.any(contact_mask):
                eps_viz = 1e-6
                points_in_contact_np_viz = points_np[contact_mask.cpu().numpy()]
                grad_x_viz = (self.sphere_sdf(points_in_contact_np_viz + np.array([eps_viz, 0, 0])) - self.sphere_sdf(points_in_contact_np_viz - np.array([eps_viz, 0, 0]))) / (2 * eps_viz)
                grad_y_viz = (self.sphere_sdf(points_in_contact_np_viz + np.array([0, eps_viz, 0])) - self.sphere_sdf(points_in_contact_np_viz - np.array([0, eps_viz, 0]))) / (2 * eps_viz)
                grad_z_viz = (self.sphere_sdf(points_in_contact_np_viz + np.array([0, 0, eps_viz])) - self.sphere_sdf(points_in_contact_np_viz - np.array([0, 0, eps_viz]))) / (2 * eps_viz)
                grad_np_viz = np.stack([grad_x_viz, grad_y_viz, grad_z_viz], axis=-1)
                normals_local_viz = -math_utils.normalize(torch.from_numpy(grad_np_viz).to(self.device))
                forces_world_viz = math_utils.quat_apply(sphere_quat_w.repeat(normals_local_viz.shape[0], 1), normals_local_viz)
                force_vectors_viz = -forces_world_viz
            
            is_running = self.visualizer.update(contact_points_world_viz.cpu().numpy(), force_vectors_viz.cpu().numpy())
            if not is_running:
                self.enable_visualization = False # Stop trying to update if window is closed

        if not torch.any(contact_mask):
            return

        eps = 1e-6
        points_in_contact_np = points_np[contact_mask.cpu().numpy()]
        grad_x = (self.sphere_sdf(points_in_contact_np + np.array([eps, 0, 0])) - self.sphere_sdf(points_in_contact_np - np.array([eps, 0, 0]))) / (2 * eps)
        grad_y = (self.sphere_sdf(points_in_contact_np + np.array([0, eps, 0])) - self.sphere_sdf(points_in_contact_np - np.array([0, eps, 0]))) / (2 * eps)
        grad_z = (self.sphere_sdf(points_in_contact_np + np.array([0, 0, eps])) - self.sphere_sdf(points_in_contact_np - np.array([0, 0, eps]))) / (2 * eps)
        
        grad_np = np.stack([grad_x, grad_y, grad_z], axis=-1)
        grad = torch.from_numpy(grad_np).to(self.device)
        
        normals_local = torch.zeros_like(self.tactile_points_local.squeeze(0))
        normals_local[contact_mask] = -math_utils.normalize(grad)

        force_magnitudes = self.tactile_kn * penetration_depth
        forces_local = normals_local * force_magnitudes.unsqueeze(-1)
        
        forces_world = math_utils.quat_apply(sphere_quat_w.repeat(self.num_tactile_points, 1), forces_local)

        total_force = torch.sum(forces_world, dim=0)
        
        force_on_sphere = total_force.unsqueeze(0)
        sphere_state = self.sphere.data.root_state_w.clone()
        velocity_change_sphere = (force_on_sphere / self.sphere.cfg.spawn.mass_props.mass) * self.sim.get_physics_dt()
        sphere_state[:, 7:10] += velocity_change_sphere
        self.sphere.write_root_velocity_to_sim(sphere_state[:, 7:13])
        
        if count % 10 == 0:
            print(f"Total reaction force on sphere: {force_on_sphere.cpu().numpy()}")

    def cleanup(self):
        """清理可视化窗口。"""
        if self.enable_visualization and hasattr(self, 'visualizer'):
            self.visualizer.close()

    def get_scene_entities(self):
        """返回此系统创建的场景实体。"""
        entities = {
            "cube": self.cube,
            "sphere": self.sphere,
            "tactile_system": self,
        }
        return entities


def design_scene(sim: sim_utils.SimulationContext):
    """设计场景。"""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    tactile_system = TactileSystem(sim)
    return tactile_system.get_scene_entities()


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, object]):
    """运行仿真循环。"""
    cube = entities["cube"]
    sphere = entities["sphere"]
    tactile_system = entities["tactile_system"]
    
    global count
    count = 0

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    
    while simulation_app.is_running():
        if count % 1000 == 0:
            sim_time = 0.0
            count = 0
            root_state_cube = cube.data.default_root_state.clone()
            root_state_cube[:, 2] = 0.2
            cube.write_root_state_to_sim(root_state_cube)
            root_state_sphere = sphere.data.default_root_state.clone()
            root_state_sphere[:, 2] = 0.5
            sphere.write_root_state_to_sim(root_state_sphere)
            cube.reset()
            sphere.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object states...")

        sphere_state = sphere.data.root_state_w.clone()
        if count > 30 and count < 50 :
            sphere_state[:, 8] = -0.5
            sphere.write_root_velocity_to_sim(sphere_state[:, 7:13])
        elif count >= 50:
            sphere_state[:, 8] = 0.5
            sphere.write_root_velocity_to_sim(sphere_state[:, 7:13])

        cube.write_data_to_sim()
        sphere.write_data_to_sim()

        sim.step()
        
        cube.update(sim_dt)
        sphere.update(sim_dt)
        
        tactile_system.update()

        if sim.is_playing():
            sim.render()

        sim_time += sim_dt
        count += 1


def main():
    """主函数。"""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.7, 0.7, 0.7], target=[0.0, 0.0, 0.2])
    scene_entities = design_scene(sim)
    sim.reset()
    print("[INFO]: Setup complete...")
    
    run_simulator(sim, scene_entities)
    return scene_entities.get("tactile_system")


if __name__ == "__main__":
    tactile_system = None
    try:
        tactile_system = main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if tactile_system:
            tactile_system.cleanup()
        # 关闭仿真应用
        simulation_app.close()