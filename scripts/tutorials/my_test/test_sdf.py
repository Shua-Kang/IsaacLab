# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the eighth step in creating a penalty-based tactile sensor
simulation in Isaac Lab using pysdf.

Step 8: Using collision filtering to disable default physics and allow for explicit,
        visible overlap controlled only by our custom penalty forces.

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
from isaacsim.core.utils.torch.transformations import tf_apply, tf_inverse


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
        self._generate_tactile_points(num_rows=20, num_cols=20)

        # 为球体初始化SDF
        self._initialize_sdf()

        # -- MODIFICATION: Reduce stiffness to make overlap more visible --
        # 定义惩罚法的物理参数
        self.tactile_kn = 800.0  # 法向刚度 (N/m)
        
        # 控制 trimesh 可视化
        self.enable_visualization = True

    def _create_objects(self):
        """创建立方体（传感器）和球体（被感知的物体）。"""
        # -- MODIFICATION: Disable default collision on the cube entirely --
        # 立方体配置 (我们的 "触觉传感器")
        self.cube_cfg = RigidObjectCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True
                                                                 ,contact_offset = 0.05,
                                                                 rest_offset=0.05
                                                                 ), # This will prevent the cube from having any collision volume
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
            debug_vis = False
        )
        self.cube = RigidObject(cfg=self.cube_cfg)

        # 球体配置 (被 "感觉" 的物体)
        # The sphere keeps its collision properties to interact with the ground plane
        self.sphere_cfg = RigidObjectCfg(
            prim_path="/World/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=0.1,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset = 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
        self.sphere = RigidObject(cfg=self.sphere_cfg)

    def _generate_tactile_points(self, num_rows: int, num_cols: int):
        """在立方体传感器的顶面生成一个触觉点网格。"""
        print("[INFO] Generating tactile points...")
        sensor_size = self.cube_cfg.spawn.size
        # 我们只在顶面 (Z+) 创建点
        top_face_z = sensor_size[2] / 2.0 + 0.02
        
        x = torch.linspace(-sensor_size[0] / 2.0, sensor_size[0] / 2.0, num_cols, device=self.device)
        y = torch.linspace(-sensor_size[1] / 2.0, sensor_size[1] / 2.0, num_rows, device=self.device)
        
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        
        points = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), top_face_z)], dim=-1)
        
        # 将点存储在立方体的局部坐标系中
        self.tactile_points_local = points.unsqueeze(0)  # 增加一个环境维度 (num_envs=1)
        self.num_tactile_points = self.tactile_points_local.shape[1]
        print(f"[INFO] Generated {self.num_tactile_points} tactile points.")

    def _initialize_sdf(self):
        """使用trimesh和pysdf为球体创建一个SDF对象。"""
        print("[INFO] Initializing SDF for the sphere...")
        # 1. 使用 trimesh 创建一个球体网格
        radius = self.sphere_cfg.spawn.radius
        mesh = trimesh.primitives.Sphere(radius=radius, subdivisions=4)
        
        # 2. 使用 pysdf 从网格顶点和面创建 SDF 对象
        self.sphere_sdf = SDF(mesh.vertices, mesh.faces)
        print("[INFO] SDF initialized.")

    def visualize_forces(self, contact_points_w, force_vectors_w, contact_patch_w):
        """使用 Trimesh 可视化力和接触点。"""
        print("[INFO] Visualizing forces with trimesh... Close the new window to continue simulation.")
        force_scale = 0.001  # 缩放因子，使力在视觉上可见

        # 转换为 numpy 以便 trimesh 使用
        points_np = contact_points_w.cpu().numpy()
        vectors_np = force_vectors_w.cpu().numpy() * force_scale
        patch_np = contact_patch_w.cpu().numpy()

        # 为力向量创建线段
        path_vertices = []
        for point, vector in zip(points_np, vectors_np):
            path_vertices.append(point)
            path_vertices.append(point + vector)
        
        path_segments = np.array(path_vertices).reshape(-1, 2, 3)
        force_lines = trimesh.load_path(path_segments, colors=[[255, 0, 0, 255]] * len(path_segments))

        # -- MODIFICATION: Visualize contact patch as a point cloud --
        # 将接触面片显示为绿色的点云
        contact_patch_pc = trimesh.PointCloud(patch_np, colors=[[0, 255, 0, 255]] * len(patch_np))

        # 创建场景
        scene = trimesh.Scene()
        scene.add_geometry(force_lines)
        scene.add_geometry(contact_patch_pc)
        
        # 显示场景 (这将阻塞执行)
        scene.show()

    def update(self):
        """在每个仿真步骤中被调用，以计算并施加触觉力。"""
        # 1. 获取球体和立方体的当前姿态
        sphere_pos_w = self.sphere.data.root_pos_w
        sphere_quat_w = self.sphere.data.root_quat_w
        cube_pos_w = self.cube.data.root_pos_w
        cube_quat_w = self.cube.data.root_quat_w

        # 2. 将局部触觉点转换到世界坐标系
        tactile_points_world = tf_apply(cube_quat_w, cube_pos_w, self.tactile_points_local)

        # 3. 将世界坐标系中的点转换到球体的局部坐标系
        sphere_pose_inv = tf_inverse(sphere_quat_w, sphere_pos_w)
        tactile_points_sphere_local = tf_apply(sphere_pose_inv[0], sphere_pose_inv[1], tactile_points_world)

        # 4. 查询 SDF 以获取穿透深度
        points_np = tactile_points_sphere_local.cpu().numpy().squeeze(0)
        distances_np = self.sphere_sdf(points_np) # pysdf: positive outside, negative inside

        # -- FIX: Correctly calculate penetration depth based on SDF convention --
        # 只有当 distance 为负（即穿透）时，我们才计算一个正的穿透深度。
        # 否则，穿透深度为0。
        penetration_depth_np = -np.minimum(-distances_np, 0)
        penetration_depth = torch.from_numpy(penetration_depth_np).to(self.device)

        # 5. 计算法向力
        contact_mask = penetration_depth > 0
        print(distances_np)
        if not torch.any(contact_mask):
            return
        # -- 使用中心差分法更精确地计算梯度 --
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
        
        # 6. 将力从球体的局部坐标系转换回世界坐标系
        forces_world = math_utils.quat_apply(sphere_quat_w.repeat(self.num_tactile_points, 1), forces_local)

        # 7. 施加力到球体上 (反作用力)
        total_force = torch.sum(forces_world, dim=0)
        
        force_on_sphere = total_force.unsqueeze(0)
        sphere_state = self.sphere.data.root_state_w.clone()
        velocity_change_sphere = (force_on_sphere / self.sphere.cfg.spawn.mass_props.mass) * self.sim.get_physics_dt()
        sphere_state[:, 7:10] += velocity_change_sphere
        # self.sphere.write_root_velocity_to_sim(sphere_state[:, 7:13])
        
        if count % 10 == 0:
            print(f"Total reaction force on sphere: {force_on_sphere.cpu().numpy()}")

        # 8. 可视化力
        if self.enable_visualization: # Periodically visualize
            # -- MODIFICATION: Calculate contact patch for visualization --
            points_in_contact_local = tactile_points_sphere_local.squeeze(0)[contact_mask]
            penetration_depth_in_contact = penetration_depth[contact_mask]
            normals_in_contact = normals_local[contact_mask]
            
            # 从穿透点沿法线方向移回穿透深度，得到表面上的点
            closest_points_local = points_in_contact_local + penetration_depth_in_contact.unsqueeze(-1) * normals_in_contact
            closest_points_world = tf_apply(sphere_quat_w, sphere_pos_w, closest_points_local.unsqueeze(0))

            contact_points_world_viz = tactile_points_world.squeeze(0)[contact_mask]
            force_vectors_to_visualize = -forces_world[contact_mask]
            
            self.visualize_forces(contact_points_world_viz, force_vectors_to_visualize, closest_points_world.squeeze(0))


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
    # 地平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 灯光
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # 触觉系统
    tactile_system = TactileSystem(sim)

    # 返回场景信息
    return tactile_system.get_scene_entities()


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, object]):
    """运行仿真循环。"""
    # 提取场景实体
    cube = entities["cube"]
    sphere = entities["sphere"]
    tactile_system = entities["tactile_system"]
    
    global count
    count = 0

    # 定义仿真步进
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    
    # 仿真物理
    while simulation_app.is_running():
        # 重置
        if count % 1000 == 0: # Increase reset interval
            # 重置计数器
            sim_time = 0.0
            count = 0
            # 重置立方体的根状态
            root_state_cube = cube.data.default_root_state.clone()
            root_state_cube[:, 2] = 0.2
            cube.write_root_state_to_sim(root_state_cube)

            # 重置球体的根状态
            root_state_sphere = sphere.data.default_root_state.clone()
            root_state_sphere[:, 2] = 0.5
            sphere.write_root_state_to_sim(root_state_sphere)

            # 重置缓冲区
            cube.reset()
            sphere.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object states...")

        # -- MODIFICATION: Implement reciprocating rolling motion --
        # 施加动作
        sphere_state = sphere.data.root_state_w.clone()
        if count > 100 and count < 300 :
            # 阶段1: 让球体下落以接触立方体
            sphere_state[:, 8] = -0.5 # vz = -0.5 m/s
            sphere.write_root_velocity_to_sim(sphere_state[:, 7:13])
        elif count >= 300:
            # 阶段2: 施加一个基于正弦波的振荡速度让球体滚动
            # 确保清除下落阶段的残余速度
            sphere_state[:, 8] = 0.0
            # 计算振荡速度
            time_since_roll_started = sim_time - (300 * sim_dt)
            amplitude = 0.1  # m/s
            frequency = 1.5  # rad/s
            sphere_state[:, 7] = amplitude * torch.sin(frequency * time_since_roll_started) # vx
            sphere.write_root_velocity_to_sim(sphere_state[:, 7:13])


        # 将仿真数据写入
        cube.write_data_to_sim()
        sphere.write_data_to_sim()

        # 执行一步
        sim.step()
        
        # 更新缓冲区
        cube.update(sim_dt)
        sphere.update(sim_dt)
        
        # 更新触觉系统（查询SDF并计算力）
        tactile_system.update()

        # 渲染主视口
        if sim.is_playing():
            sim.render()

        # 更新仿真时间
        sim_time += sim_dt
        count += 1


def main():
    """主函数。"""
    # 加载kit助手
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # -- FIX: Corrected typo from Context to SimulationContext --
    sim = SimulationContext(sim_cfg)
    # 设置主摄像头
    sim.set_camera_view(eye=[0.7, 0.7, 0.7], target=[0.0, 0.0, 0.2])
    # 设计场景
    scene_entities = design_scene(sim)
    # 运行仿真器
    sim.reset()
    # 现在我们准备好了!
    print("[INFO]: Setup complete...")
    # 运行仿真
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # 运行主函数
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 关闭仿真应用
        simulation_app.close()
