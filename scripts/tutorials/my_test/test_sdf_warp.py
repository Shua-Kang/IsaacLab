# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the final, high-performance version of the tactile sensor
simulation, using an analytical SDF inside a NVIDIA Warp kernel.

This approach is the most performant as it avoids building any data structures
and computes the SDF mathematically on the GPU.

.. code-block:: bash

    # Usage
    # Note: This script requires warp-lang and matplotlib packages.
    # You can install them using pip:
    # ./isaaclab.sh -p --pip-install warp-lang matplotlib

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -- MODIFICATION: Import Warp --
import warp as wp
import warp.torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
import isaaclab.utils.math as math_utils
from isaacsim.core.utils.torch.transformations import tf_apply, tf_inverse

# -- MODIFICATION: Initialize Warp --
wp.init()

class MatplotlibVisualizer:
    """
    一个使用Matplotlib进行非阻塞式3D可视化的类。
    """
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Tactile Force Field (Warp - Analytical SDF)')
        self.ax.set_xlim([-0.15, 0.15])
        self.ax.set_ylim([-0.15, 0.15])
        self.ax.set_zlim([0.2, 0.4])
        plt.show()

    def update(self, contact_points_w, force_vectors_w):
        """更新并重新绘制3D力场图。"""
        try:
            self.ax.cla()
            self.ax.set_xlim([-0.15, 0.15])
            self.ax.set_ylim([-0.15, 0.15])
            self.ax.set_zlim([0.2, 0.4])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('Tactile Force Field (Warp - Analytical SDF)')

            if contact_points_w.shape[0] > 0:
                p = contact_points_w
                f = force_vectors_w
                self.ax.quiver(p[:, 0], p[:, 1], p[:, 2],
                               f[:, 0], f[:, 1], f[:, 2],
                               length=0.05, normalize=True, color='r')
            
            plt.pause(0.001) 
        except Exception as e:
            print(f"[Visualizer] Matplotlib window may have been closed. Error: {e}")
            return False
        return True

    def close(self):
        """关闭matplotlib窗口。"""
        plt.close(self.fig)


# -- MODIFICATION: Define the Warp Kernel for force calculation --
@wp.kernel
def compute_forces_kernel(
    tactile_points_local: wp.array(dtype=wp.vec3),
    forces_world_out: wp.array(dtype=wp.vec3),
    contact_mask_out: wp.array(dtype=wp.uint8),
    cube_tf: wp.transform,
    sphere_tf: wp.transform,
    sphere_radius: float,
    stiffness: float,
):
    """
    这个Kernel在GPU上为每个触觉点并行执行。
    """
    # 获取当前线程处理的触觉点的索引
    tid = wp.tid()

    # 步骤 1: 将触觉点从立方体的局部坐标系转换到世界坐标系
    p_local_cube = tactile_points_local[tid]
    p_world = wp.transform_point(cube_tf, p_local_cube)

    # 步骤 2: 将世界坐标系中的点转换到球体的局部坐标系
    sphere_tf_inv = wp.transform_inverse(sphere_tf)
    p_local_sphere = wp.transform_point(sphere_tf_inv, p_world)

    # -- MODIFICATION: Use analytical SDF for sphere --
    # 步骤 3: 使用解析SDF计算距离
    # wp.print(p_local_sphere)
    wp.print(wp.length(p_local_sphere))
    dist = wp.length(p_local_sphere) - sphere_radius
    
    # 步骤 4: 计算惩罚力（仅当穿透时）
    penetration = -wp.min(-dist, 0.0)

    # 表面法线是归一化的位置向量, 指向外
    normal_local_sphere = wp.normalize(p_local_sphere)
    # 计算力的量值
    force_mag = stiffness * penetration
    # -- PHYSICS FIX: Repulsive force on the point should be along the outward normal --
    force_local_sphere = normal_local_sphere * force_mag 
    # 将力从球体的局部坐标系转换回世界坐标系
    force_world = wp.transform_vector(sphere_tf, force_local_sphere)
    # 将结果写入输出数组
    forces_world_out[tid] = force_world
    contact_mask_out[tid] = wp.uint8(1)



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

        self._create_objects()
        self._generate_tactile_points()

        self.tactile_kn = 800.0
        self.enable_visualization = True

        if self.enable_visualization:
            self.visualizer = MatplotlibVisualizer()

    def _create_objects(self):
        """创建立方体（传感器）和球体（被感知的物体）。"""
        # -- FIX: Correctly disable collision using collision_props and collision_group --
        self.cube_cfg = RigidObjectCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
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

    def _generate_tactile_points(self, num_rows=20, num_cols=20):
        """在立方体传感器的顶面生成一个触觉点网格。"""
        print("[INFO] Generating tactile points...")
        sensor_size = self.cube_cfg.spawn.size
        top_face_z = sensor_size[2] / 2.0 + 0.0001
        x = torch.linspace(-sensor_size[0] / 2.0, sensor_size[0] / 2.0, num_cols, device=self.device)
        y = torch.linspace(-sensor_size[1] / 2.0, sensor_size[1] / 2.0, num_rows, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), top_face_z)], dim=-1)
        self.tactile_points_local_th = points
        self.num_tactile_points = self.tactile_points_local_th.shape[0]
        
        def to_wp_vec3(t: torch.Tensor, device: str):
            """
            t: 形状 [N, 3] 的 torch.Tensor
            返回: 形状 [N]、dtype=wp.vec3 的 wp.array
            """
            pts = t.detach().contiguous().view(-1, 3).cpu().numpy().astype(np.float32)
            return wp.array(pts, dtype=wp.vec3, device=("cuda" if "cuda" in device else device))

        self.tactile_points_local_wp = to_wp_vec3(self.tactile_points_local_th, self.device)

        # self.tactile_points_local_wp = warp.torch.from_torch(self.tactile_points_local_th)
        print(f"[INFO] Generated {self.num_tactile_points} tactile points.")

    def update(self):
        """在每个仿真步骤中被调用，以计算并施加触觉力。"""
        # 1. 获取姿态并转换为Warp格式
        sphere_pos_w = self.sphere.data.root_pos_w
        sphere_quat_w = self.sphere.data.root_quat_w
        cube_pos_w = self.cube.data.root_pos_w
        cube_quat_w = self.cube.data.root_quat_w


        def wp_transform_from_torch(pos_th: torch.Tensor, quat_th: torch.Tensor) -> wp.transform:
            # pos_th: [1,3] 或 [3]，quat_th: [1,4] 或 [4]
            pos = pos_th.reshape(-1).tolist()
            quat = quat_th.reshape(-1).tolist()
            return wp.transform(
                wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )
        cube_tf = wp_transform_from_torch(cube_pos_w, cube_quat_w)
        sphere_tf = wp_transform_from_torch(sphere_pos_w, sphere_quat_w)

        # 2. 准备输出数组并启动Kernel
        forces_world_wp = wp.zeros(self.num_tactile_points, dtype=wp.vec3, device=self.device)
        contact_mask_wp = wp.zeros(self.num_tactile_points, dtype=wp.uint8, device=self.device)
        import pdb; pdb.set_trace()
        wp.launch(
            kernel=compute_forces_kernel,
            dim=self.num_tactile_points,
            inputs=[
                self.tactile_points_local_wp,
                forces_world_wp,
                contact_mask_wp,
                cube_tf,
                sphere_tf,
                self.sphere_cfg.spawn.radius,
                self.tactile_kn,
            ],
            device=self.device,
        )

        # -- FIX: Correctly sum forces by converting to torch tensor first --
        # 3. 汇总力并应用到球体
        forces_world_th = warp.torch.to_torch(forces_world_wp)
        total_force_th = torch.sum(forces_world_th, dim=0)

        # -- PHYSICS FIX: Force on sphere is the reaction force (- total_force) --
        force_on_sphere = -total_force_th.unsqueeze(0)
        sphere_state = self.sphere.data.root_state_w.clone()
        velocity_change_sphere = (force_on_sphere / self.sphere.cfg.spawn.mass_props.mass) * self.sim.get_physics_dt()
        sphere_state[:, 7:10] += velocity_change_sphere
        # self.sphere.write_root_velocity_to_sim(sphere_state[:, 7:13])

        if count % 10 == 0:
            print(f"Total reaction force on sphere (Warp): {force_on_sphere.cpu().numpy()}")

        # 4. 可视化
        if self.enable_visualization and count % 5 == 0:
            # -- FIX: Clone tensors converted from Warp before indexing --
            contact_mask_th = warp.torch.to_torch(contact_mask_wp).clone().bool()
            
            if torch.any(contact_mask_th):
                all_tactile_points_world_th = tf_apply(cube_quat_w, cube_pos_w, self.tactile_points_local_th.unsqueeze(0)).squeeze(0)
                all_forces_world_th = forces_world_th.clone()

                # Get indices of contact points
                contact_indices = torch.nonzero(contact_mask_th).squeeze(-1)
                
                # Use indices to select data for visualization
                contact_points_world_viz = torch.index_select(all_tactile_points_world_th, 0, contact_indices)
                force_vectors_to_visualize = torch.index_select(all_forces_world_th, 0, contact_indices)

                is_running = self.visualizer.update(contact_points_world_viz.cpu().numpy(), force_vectors_to_visualize.cpu().numpy())
                if not is_running:
                    self.enable_visualization = False
            else:
                self.visualizer.update(np.array([]), np.array([]))

    def cleanup(self):
        """清理可视化窗口。"""
        if self.enable_visualization and hasattr(self, 'visualizer'):
            self.visualizer.close()

    def get_scene_entities(self):
        """返回此系统创建的场景实体。"""
        return {"cube": self.cube, "sphere": self.sphere, "tactile_system": self}


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
