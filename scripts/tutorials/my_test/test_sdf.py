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
from scipy.spatial.transform import Rotation as R
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on setting up a tactile simulation scene."
)
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
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.utils.shear_tactile_viz_utils import visualize_penetration_depth, visualize_tactile_shear_image
import cv2
import os

import warp as wp
wp.init()
@wp.kernel
def _warp_depth_normal_kernel(
    mesh_id: wp.uint64,
    pts: wp.array(dtype=wp.vec3f),
    out_depth: wp.array(dtype=wp.float32),
    out_mask: wp.array(dtype=wp.int32),
    out_normal: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    p = pts[i]

    q = wp.mesh_query_point_sign_winding_number(
        mesh_id,
        p,
        wp.float32(1.0e6),
    )

    cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    diff = p - cp
    d = wp.length(diff)

    inside = q.sign < wp.float32(0.0)
    out_mask[i]  = wp.where(inside, wp.int32(1), wp.int32(0))
    out_depth[i] = wp.where(inside, d, wp.float32(0.0))
    
    
    eps = wp.float32(1e-9)
    # n_out = diff / (wp.length(diff) + eps)
    n_out = wp.normalize(diff)
    
    out_normal[i] = n_out

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
        self.num_envs = 1
        # 创建立方体和球体
        self._create_objects()

        # 在立方体顶面生成触觉点
        self.num_rows = 20
        self.num_cols = 20
        self._generate_tactile_points(num_rows=self.num_rows, num_cols=self.num_cols)

        # 为球体初始化SDF
        

        # -- MODIFICATION: Reduce stiffness to make overlap more visible --
        # 定义惩罚法的物理参数
        self.tactile_kn = 1.0  # 法向刚度 (N/m)
        self.tactile_mu = 2.
        self.tactile_kt = 0.01
        # 控制 trimesh 可视化
        self.enable_visualization = True
        
        # self.depth_calculation_method = "pysdf" 
        self.depth_calculation_method = "warp" 
        
        self._initialize_sdf()
        
        self.num_tactile_points = self.num_rows * self.num_cols

    def _create_objects(self):
        """创建立方体（传感器）和球体（被感知的物体）。"""
        # -- MODIFICATION: Disable default collision on the cube entirely --
        # 立方体配置 (我们的 "触觉传感器")
        self.cube_cfg = ArticulationCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.UsdFileCfg(
                # usd_path=r"C:\onedrive\OneDrive - University of Virginia\Desktop\isaac\IsaacLab\my_assets_new\ball_rolling_cube\cube.usd",
                usd_path=r"C:\Users\jiuer\OneDrive - University of Virginia\Desktop\isaac\IsaacLab\my_assets_new\ball_rolling_cube\cube.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                # collision_props=sim_utils.CollisionPropertiesCfg(
                #     collision_enabled=True,
                #     contact_offset=0.05,
                #     rest_offset=0.05,
                # ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.06)),
            debug_vis=False,
            actuators = {},
        )
        self.cube = Articulation(cfg=self.cube_cfg)

        # 球体配置 (被 "感觉" 的物体)
        # The sphere keeps its collision properties to interact with the ground plane
        self.sphere_cfg = RigidObjectCfg(
            prim_path="/World/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.1)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.02)),
        )
        self.sphere = RigidObject(cfg=self.sphere_cfg)

    def _generate_tactile_points(self, num_rows: int, num_cols: int):
        """在立方体传感器的顶面生成一个触觉点网格。"""
        print("[INFO] Generating tactile points...")
        sensor_size = (0.05,0.05,0.02)
        # 我们只在顶面 (Z+) 创建点
        top_face_z = -sensor_size[2] - 0.004
        # top_face_z = 0

        x = torch.linspace(
            -sensor_size[0] / 2.0, sensor_size[0] / 2.0, num_cols, device=self.device
        )
        y = torch.linspace(
            -sensor_size[1] / 2.0, sensor_size[1] / 2.0, num_rows, device=self.device
        )

        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        points = torch.stack(
            [
                grid_x.flatten(),
                grid_y.flatten(),
                torch.full_like(grid_x.flatten(), top_face_z),
            ],
            dim=-1,
        )

        # 将点存储在立方体的局部坐标系中
        self.tactile_points_local = points.unsqueeze(0)  # 增加一个环境维度 (num_envs=1)
        self.num_tactile_points = self.tactile_points_local.shape[1]
        print(f"[INFO] Generated {self.num_tactile_points} tactile points.")
        
        rotation = (0, 0, 0) # NOTE [Jie]: assume tactile frame rotation are all the same
        tactile_points_quat = R.from_euler('xyz', rotation).as_quat(scalar_first = True)
        tactile_points_quat_tensor = torch.tensor(tactile_points_quat, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.num_tactile_points, 1)
        self.tactile_quat_local = tactile_points_quat_tensor

    def _initialize_sdf(self):
        """使用trimesh和pysdf为球体创建一个SDF对象。"""
        print("[INFO] Initializing SDF for the sphere...")
        # 1. 使用 trimesh 创建一个球体网格
        radius = self.sphere_cfg.spawn.radius
        mesh = trimesh.primitives.Sphere(radius=radius, subdivisions=4)

        # 2. 使用 pysdf 从网格顶点和面创建 SDF 对象
        self.sphere_sdf = SDF(mesh.vertices, mesh.faces)
        print("[INFO] SDF initialized.")
        
        self.sphere_wp_mesh = wp.Mesh(
            points=wp.array(mesh.vertices, dtype=wp.vec3, device=self.device),
            indices=wp.array(mesh.faces.reshape(-1), dtype=wp.int32, device=self.device),
            support_winding_number=True,
        )

    def visualize_forces(self, contact_points_w, force_vectors_w, contact_patch_w):
        """使用 Trimesh 可视化力和接触点。"""
        print(
            "[INFO] Visualizing forces with trimesh... Close the new window to continue simulation."
        )
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
        force_lines = trimesh.load_path(
            path_segments, colors=[[255, 0, 0, 255]] * len(path_segments)
        )

        # -- MODIFICATION: Visualize contact patch as a point cloud --
        # 将接触面片显示为绿色的点云
        contact_patch_pc = trimesh.PointCloud(
            patch_np, colors=[[0, 255, 0, 255]] * len(patch_np)
        )

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
        sphere_linvel_w = self.sphere.data.root_lin_vel_w
        sphere_angvel_w = self.sphere.data.root_ang_vel_w
        
        cube_pos_w = self.cube.data.root_pos_w
        cube_quat_w = self.cube.data.root_quat_w
        cube_linvel_w = self.cube.data.root_lin_vel_w
        cube_angvel_w = self.cube.data.root_ang_vel_w
        
        
        # 2. 将局部触觉点转换到世界坐标系
        tactile_points_world = tf_apply(
            cube_quat_w, cube_pos_w, self.tactile_points_local
        )

        # 3. 将世界坐标系中的点转换到球体的局部坐标系
        sphere_pose_inv = tf_inverse(sphere_quat_w, sphere_pos_w)
        tactile_points_sphere_local = tf_apply(
            sphere_pose_inv[0], sphere_pose_inv[1], tactile_points_world
        )

        # 4. 查询 SDF 以获取穿透深度
        points_np = tactile_points_sphere_local.cpu().numpy().squeeze(0)
        
        if self.depth_calculation_method == "pysdf":
            distances_np = self.sphere_sdf(
                points_np
            )  # pysdf: positive outside, negative inside
            sdf_penetration_depth_np = -np.minimum(-distances_np, 0)
            sdf_penetration_depth = torch.from_numpy(sdf_penetration_depth_np).to(self.device)

            # 5. 计算法向力
            contact_mask = sdf_penetration_depth > 0
            eps = 1e-6
            # points_in_contact_np = points_np[contact_mask.cpu().numpy()]
            grad_x = (
                self.sphere_sdf(points_np + np.array([eps, 0, 0]))
                - self.sphere_sdf(points_np - np.array([eps, 0, 0]))
            ) / (2 * eps)
            grad_y = (
                self.sphere_sdf(points_np + np.array([0, eps, 0]))
                - self.sphere_sdf(points_np - np.array([0, eps, 0]))
            ) / (2 * eps)
            grad_z = (
                self.sphere_sdf(points_np + np.array([0, 0, eps]))
                - self.sphere_sdf(points_np - np.array([0, 0, eps]))
            ) / (2 * eps)

            grad_np = np.stack([grad_x, grad_y, grad_z], axis=-1)
            grad = torch.from_numpy(grad_np).to(self.device)

            sdf_normals_local = torch.zeros_like(self.tactile_points_local)
            sdf_normals_local[:] = -math_utils.normalize(grad)
            
            penetration_depth = sdf_penetration_depth
            normals_local = sdf_normals_local
        elif self.depth_calculation_method == "warp":
            pts_wp   = wp.array(points_np, dtype=wp.vec3f,  device=self.device)
            out_d_wp = wp.empty(pts_wp.shape[0], dtype=wp.float32, device=self.device)
            out_m_wp = wp.empty(pts_wp.shape[0], dtype=wp.int32,   device=self.device)
            out_n_wp = wp.empty(pts_wp.shape[0], dtype=wp.vec3f,   device=self.device)
            wp.launch(
                kernel=_warp_depth_normal_kernel,
                dim=pts_wp.shape[0],
                inputs=[self.sphere_wp_mesh.id, pts_wp, out_d_wp, out_m_wp, out_n_wp],
                device=self.device,
            )
            wp_distances_np       = torch.from_numpy(out_d_wp.numpy()).to(self.device).reshape(self.num_envs, self.num_tactile_points).clamp(min=0.0)   # (B,N)
            wp_normals_local  = torch.from_numpy(out_n_wp.numpy()).to(self.device).reshape(self.num_envs, self.num_tactile_points, 3)              # (B,N,3)
            contact_mask = wp_distances_np > 0
            
            penetration_depth = wp_distances_np
            normals_local = -wp_normals_local

        

        
        if not torch.any(contact_mask):
            return
        # -- 使用中心差分法更精确地计算梯度 --
        

        # finish the query collision
        
        tactile_points_world_velocity = torch.cross(cube_angvel_w.unsqueeze(1).expand((self.num_envs, self.num_tactile_points, 3)), math_utils.quat_apply(sphere_quat_w, self.tactile_points_local), dim = -1) + cube_linvel_w.expand((self.num_envs, self.num_tactile_points, 3))

        # points_in_contact_local = tactile_points_sphere_local.squeeze(0)[contact_mask]
        
        # penetration_depth_in_contact = penetration_depth[contact_mask]
        # normals_in_contact = normals_local[contact_mask]
        # 从穿透点沿法线方向移回穿透深度，得到表面上的点
        closest_points_local = (tactile_points_sphere_local+ penetration_depth.unsqueeze(-1) *normals_local)
        
        closest_points_world = tf_apply(sphere_quat_w, sphere_pos_w, closest_points_local)



        normal_world = math_utils.quat_apply(sphere_quat_w, normals_local)
        
        
        closest_points_velocity_world = (torch.cross(sphere_angvel_w.unsqueeze(1).expand((self.num_envs, self.num_tactile_points, 3)),math_utils.quat_apply(sphere_pose_inv[0], closest_points_local),dim=-1)+ sphere_linvel_w.expand((self.num_envs, self.num_tactile_points, 3)))
        
        
        relative_velocity_world = tactile_points_world_velocity - closest_points_velocity_world

        vt_world = relative_velocity_world - normal_world * torch.sum(normal_world * relative_velocity_world, dim=-1, keepdim=True)

        

        depth, depth_dot, normal_world, vt_world = penetration_depth, "", normal_world, vt_world

        fc_norm = self.tactile_kn * depth #- self.tactile_damping * depth_dot * depth
        fc_world = fc_norm.unsqueeze(-1) * normal_world
        
        '''compute frictional force'''
        vt_norm = vt_world.norm(dim=-1)
        ft_static_norm = self.tactile_kt * vt_norm
        ft_dynamic_norm = self.tactile_mu * fc_norm
        ft_world = - torch.minimum(ft_static_norm, ft_dynamic_norm).unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9, max=None).unsqueeze(-1)
        # ft_world = -ft_dynamic_norm.unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9, max=None).unsqueeze(-1)
        '''net tactile force'''
        tactile_force_world = fc_world + ft_world
        
        '''tactile force in tactile frame'''
        quat_pad_inv = math_utils.quat_conjugate(cube_quat_w)
        tactile_force_pad = math_utils.quat_apply(quat_pad_inv.unsqueeze(1).expand(self.num_envs, self.num_tactile_points, 4), tactile_force_world)
        
        UnitX = torch.tensor([1., 0., 0.], device=self.device)
        UnitY = torch.tensor([0., 1., 0.], device=self.device)
        UnitZ = torch.tensor([0., 0., -1.], device=self.device)
        tactile_normal_axis = math_utils.quat_apply(self.tactile_quat_local, UnitZ.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        tactile_shear_x_axis = math_utils.quat_apply(self.tactile_quat_local, UnitX.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        tactile_shear_y_axis = math_utils.quat_apply(self.tactile_quat_local, UnitY.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        
        tactile_normal_force = -(tactile_normal_axis * tactile_force_pad).sum(-1)
        tactile_shear_force_x = (tactile_shear_x_axis * tactile_force_pad).sum(-1)
        tactile_shear_force_y = (tactile_shear_y_axis * tactile_force_pad).sum(-1)
        tactile_shear_force = torch.cat((tactile_shear_force_x.unsqueeze(-1), tactile_shear_force_y.unsqueeze(-1)), dim=-1)
        if self.enable_visualization:
            tactile_image = visualize_tactile_shear_image(tactile_normal_force[0].view((self.num_rows, self.num_cols)).cpu().numpy(), tactile_shear_force[0].view((self.num_rows, self.num_cols, 2)).cpu().numpy(), normal_force_threshold=0.003, shear_force_threshold=0.001)
            # import pdb; pdb.set_trace()
            # cv2.imwrite(os.path.join(r"C:\onedrive\OneDrive - University of Virginia\Desktop\isaac", "tactile_shear_image.png"), (visualize_tactile_shear_image(tactile_normal_force[0].view((self.num_rows, self.num_cols)).cpu().numpy(), tactile_shear_force[0].view((self.num_rows, self.num_cols, 2)).cpu().numpy(), normal_force_threshold=0.003, shear_force_threshold=0.002)*255.0).astype(np.uint8))
            cv2.imwrite(os.path.join(r"C:\Users\jiuer\OneDrive - University of Virginia\Desktop\isaac", "tactile_shear_image.png"), (visualize_tactile_shear_image(tactile_normal_force[0].view((self.num_rows, self.num_cols)).cpu().numpy(), tactile_shear_force[0].view((self.num_rows, self.num_cols, 2)).cpu().numpy(), normal_force_threshold=0.003, shear_force_threshold=0.002)*255.0).astype(np.uint8))
            cv2.imshow("tactile_shear_image", tactile_image)
            cv2.waitKey(1)
        # 8. 可视化力
        if self.enable_visualization and False:  # Periodically visualize
            # -- MODIFICATION: Calculate contact patch for visualization --
            points_in_contact_local = tactile_points_sphere_local.squeeze(0)[
                contact_mask
            ]
            penetration_depth_in_contact = penetration_depth[contact_mask]
            normals_in_contact = normals_local[contact_mask]

            # 从穿透点沿法线方向移回穿透深度，得到表面上的点
            closest_points_local = (
                points_in_contact_local
                + penetration_depth_in_contact.unsqueeze(-1) * normals_in_contact
            )
            closest_points_world = tf_apply(
                sphere_quat_w, sphere_pos_w, closest_points_local.unsqueeze(0)
            )

            contact_points_world_viz = tactile_points_world.squeeze(0)[contact_mask]
            force_magnitudes = self.tactile_kn * penetration_depth
            forces_local = normals_local * force_magnitudes.unsqueeze(-1)
            force_vectors_to_visualize = forces_world[contact_mask]
            forces_world = math_utils.quat_apply(sphere_quat_w.repeat(self.num_tactile_points, 1), forces_local)
            self.visualize_forces(
                contact_points_world_viz,
                force_vectors_to_visualize,
                closest_points_world.squeeze(0),
            )

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

    forces  = torch.zeros((1, 1, 3), device="cuda:0")
    
    
    actions = [
        1.0 * torch.tensor([[0, 0.45, 0]]),
        1.0 * torch.tensor([[0, -0.45, 0]]),
        1.0 * torch.tensor([[0, -0.45, 0]]),
        1.0 * torch.tensor([[0, 0.45, 0]]),
        1.0 * torch.tensor([[0.45, 0, 0]]),
        1.0 * torch.tensor([[-0.45, 0, 0]]),
        1.0 * torch.tensor([[-0.45, 0, 0]]),
        1.0 * torch.tensor([[0.45, 0, 0]]),
    ]

    steps = [
        10, 10, 10, 10, 10, 10, 10, 10,
    ]
    # 仿真物理
    while simulation_app.is_running():
        # 重置
        # if count % 1000 == 0 and count != 0:  # Increase reset interval
        #     # 重置计数器
        #     sim_time = 0.0
        #     count = 0
        #     # 重置立方体的根状态
        #     root_state_cube = cube.data.default_root_state.clone()
        #     root_state_cube[:, 2] = 0.2
        #     cube.write_root_state_to_sim(root_state_cube)

        #     # 重置球体的根状态
        #     root_state_sphere = sphere.data.default_root_state.clone()
        #     root_state_sphere[:, 2] = 0.5
        #     sphere.write_root_state_to_sim(root_state_sphere)

        #     # 重置缓冲区
        #     cube.reset()
        #     sphere.reset()
        #     print("----------------------------------------")
        #     print("[INFO]: Resetting object states...")

        # -- MODIFICATION: Implement reciprocating rolling motion --
        # 施加动作
        # sphere_state = sphere.data.root_state_w.clone()
            # 阶段1: 让球体下落以接触立方体
        # sphere_state[:, 8] = -0.5  # vz = -0.5 m/s
        # sphere.write_root_velocity_to_sim(sphere_state[:, 7:13])
        # import pdb; pdb.set_trace()
        # 将仿真数据写入
        
        for i in range(len(steps)):
            for j in range(steps[i]):
                # print("step", i, "step", j)
                forces  = actions[i].to(sim.device)
                torques = torch.zeros_like(forces).to(sim.device)
                cube.set_external_force_and_torque(forces, torques, is_global=True)
                
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
