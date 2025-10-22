# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils
from pxr import Usd, UsdGeom, Gf
import matplotlib.pyplot as plt
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
from scipy.spatial.transform import Rotation as R
from . import factory_control, factory_utils
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv
from isaacsim.core.utils.torch.transformations import tf_apply, tf_inverse
import random
import trimesh
from pysdf import SDF
import os
import rtree
from isaaclab.sensors import TiledCamera, Camera
import torchvision
import numpy as np
import cv2
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.utils.shear_tactile_viz_utils import visualize_penetration_depth, visualize_tactile_shear_image
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

import time
import random
import warp as wp
wp.init()

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

    def __init__(self, env: "FactoryEnv", num_rows_per_finger: int = 20, num_cols_per_finger: int = 20 ):
        """
        通过创建传感器和物体来初始化触觉系统。
        """
        self.env = env
        self.device = env.device
        self.sim = env.sim
        self.cfg = env.cfg
        self.device = env.device
        self.num_envs = 1
        # 创建立方体和球体
        # self._create_objects()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 注意：这里的相对路径'..'的数量可能需要根据您实际的文件结构进行调整
        self.peg_stl_path = os.path.join(current_dir, "..", "..", "..", "..", "..", "my_assets_new", "peg", "peg.stl")
        self.elastomer_stl_path = os.path.join(current_dir, "..", "..", "..", "..", "..", "my_assets_new", "peg", "extruded_elastomer_transformed.stl")
        print(f"[INFO] Peg STL path: {self.peg_stl_path}")
        print(f"[INFO] Elastomer STL path: {self.elastomer_stl_path}")
        # 在立方体顶面生成触觉点
        self.num_rows = num_rows_per_finger
        self.num_cols = num_cols_per_finger
        self.num_tactile_points = self.num_rows * self.num_cols
        self._generate_tactile_points(num_rows=self.num_rows, num_cols=self.num_cols)

        # 为球体初始化SDF
        self._robot = self.env._robot
        self._peg = self.env._held_asset
        self.left_finger_idx = self._robot.body_names.index("elastomer_left")
        self.right_finger_idx = self._robot.body_names.index("elastomer_right")
        # -- MODIFICATION: Reduce stiffness to make overlap more visible --
        # 定义惩罚法的物理参数
        self.tactile_kn = 1.0  # 法向刚度 (N/m)
        self.tactile_mu = 2.
        self.tactile_kt = 0.1
        # 控制 trimesh 可视化
        self.enable_visualization = True
        
        # self.depth_calculation_method = "warp" 
        self.depth_calculation_method = "pysdf" 
        # self.depth_calculation_method = "warp" 
        
        self._initialize_sdf()
        
        

    def _load_mesh_from_file(self, file_path: str) -> trimesh.Trimesh | None:
        """
        一个从本地文件加载trimesh对象的辅助函数。
        """
        try:
            if not os.path.exists(file_path):
                print(f"[VIZ-ERROR] Mesh file not found at: {file_path}")
                return None
            mesh = trimesh.load(file_path, force='mesh')
            print(f"[VIZ-INFO] Successfully loaded mesh from: {file_path}")
            return mesh
        except Exception as e:
            print(f"[VIZ-ERROR] Failed to load mesh from {file_path}. Error: {e}")
            return None

    def _generate_tactile_points(self, num_rows: int = 10, num_cols: int = 10, margin: float = 0.001):
        """
        通过在 elastomer 模型上进行光线投射，来生成触觉点。
        这种方法可以自适应任何模型表面。
        """
        print("[INFO] Generating tactile points via ray casting...")
        
        # 1. 从STL文件加载网格
        mesh = self._load_mesh_from_file(self.elastomer_stl_path)
        if mesh is None:
            raise RuntimeError(f"Cannot generate tactile points because mesh failed to load from {self.elastomer_stl_path}")
        
        # 2. 将网格中心移到原点，与SDF和可视化保持一致
        # mesh.apply_translation(-mesh.centroid)

        # 3. 自动确定网格的“薄”轴，作为光线投射的方向
        elastomer_dims = mesh.bounding_box.extents
        slim_axis = np.argmin(elastomer_dims)
        major_axes = [i for i in range(3) if i != slim_axis]
        
        print(f"[INFO] Detected slim axis: {['X', 'Y', 'Z'][slim_axis]}. Projecting points along this axis.")

        # 4. 在一个平面上创建光线起点的网格，该平面位于模型外部
        bounds = mesh.bounds
        ray_origin_start = bounds[1] + np.array([0.1, 0.1, 0.1]) # 从边界外开始
        ray_dir = np.array([0.0, 0.0, 0.0])
        ray_dir[slim_axis] = -1.0 # 朝向模型 (-1)

        # 在两个主轴上创建网格点
        x_coords = np.linspace(bounds[0][major_axes[0]] + margin, bounds[1][major_axes[0]] - margin, num_cols)
        y_coords = np.linspace(bounds[0][major_axes[1]] + margin, bounds[1][major_axes[1]] - margin, num_rows)
        
        ray_origins = []
        for y in y_coords:
            for x in x_coords:
                point = np.zeros(3)
                point[major_axes[0]] = x
                point[major_axes[1]] = y
                point[slim_axis] = ray_origin_start[slim_axis]
                ray_origins.append(point)
        ray_origins = np.array(ray_origins)
        ray_directions = np.tile(ray_dir, (len(ray_origins), 1))

        # 5. 执行光线投射
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        locations, index_ray, _ = intersector.intersects_location(ray_origins, ray_directions)

        if len(locations) != len(ray_origins):
            print(f"[WARN] Ray casting missed some points. Expected {len(ray_origins)}, got {len(locations)}. Try adjusting margin.")
        visualize = False
        locations = locations[locations[:, 1] < 0]
        locations[:, 2] = locations[:, 2]

        
        if visualize:
            print("[DEBUG] Visualizing generated tactile points... (Close window to continue)")
            # 创建一个点云对象来显示命中的点
            # remove point if y > 0
            
            point_cloud = trimesh.PointCloud(locations, colors=[255, 0, 0]) # 红色点
            # 创建一个场景，包含原始网格和生成的点云
            scene = trimesh.Scene([mesh, point_cloud])
            # 显示场景，这会暂停执行直到窗口被关闭
            scene.show()

        # 6. 将生成的点转换为Tensor并存储
        points = torch.from_numpy(locations).to(device=self.device, dtype=torch.float32)
        
        # points = points[points[:, 1].argsort(stable=True)]  # 按 z 排
        points = points[points[:, 0].argsort(stable=True)]  # 按 y 排
        points = points[points[:, 2].argsort(stable=True)]  # 按 x 排
        # 假设左右手指使用相同的局部点云
        self.tactile_points_left_local = points.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.tactile_points_right_local = points.unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        self.num_points_per_finger = self.tactile_points_left_local.shape[1]

        rotation = (0, 0, 0) # NOTE [Jie]: assume tactile frame rotation are all the same
        tactile_points_quat = R.from_euler('xyz', rotation).as_quat(scalar_first = True)
        tactile_points_quat_tensor = torch.tensor(tactile_points_quat, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.num_tactile_points, 1)
        self.tactile_quat_local = tactile_points_quat_tensor
        print(f"[INFO] Generated {self.num_points_per_finger} tactile points successfully.")


    # def _generate_tactile_points(self, num_rows: int, num_cols: int):
    #     """在立方体传感器的顶面生成一个触觉点网格。"""
    #     print("[INFO] Generating tactile points...")
    #     sensor_size = (0.05,0.05,0.02)
    #     # 我们只在顶面 (Z+) 创建点
    #     top_face_z = -sensor_size[2] - 0.004
    #     # top_face_z = 0

    #     x = torch.linspace(
    #         -sensor_size[0] / 2.0, sensor_size[0] / 2.0, num_cols, device=self.device
    #     )
    #     y = torch.linspace(
    #         -sensor_size[1] / 2.0, sensor_size[1] / 2.0, num_rows, device=self.device
    #     )

    #     grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    #     points = torch.stack(
    #         [
    #             grid_x.flatten(),
    #             grid_y.flatten(),
    #             torch.full_like(grid_x.flatten(), top_face_z),
    #         ],
    #         dim=-1,
    #     )

    #     # 将点存储在立方体的局部坐标系中
    #     self.tactile_points_local = points.unsqueeze(0)  # 增加一个环境维度 (num_envs=1)
    #     self.num_tactile_points = self.tactile_points_local.shape[1]
    #     print(f"[INFO] Generated {self.num_tactile_points} tactile points.")
        
    #     rotation = (0, 0, 0) # NOTE [Jie]: assume tactile frame rotation are all the same
    #     tactile_points_quat = R.from_euler('xyz', rotation).as_quat(scalar_first = True)
    #     tactile_points_quat_tensor = torch.tensor(tactile_points_quat, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, self.num_tactile_points, 1)
    #     self.tactile_quat_local = tactile_points_quat_tensor


    def _extract_mesh_from_prim(self, prim_path: str) -> trimesh.Trimesh | None:
        """
        一个更鲁棒的辅助函数，用于从给定的Prim路径提取Trimesh对象。
        它会递归地组合一个Prim下的所有子网格（包括隐式几何体），并使用兼容的API。
        """
        try:
            root_prim = self.env.scene.stage.GetPrimAtPath(prim_path)
            if not root_prim.IsValid():
                print(f"[VIZ-WARN] Root prim for mesh extraction not valid: {prim_path}")
                return None

            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            combined_mesh = trimesh.Trimesh()

            stack = [root_prim]
            while stack:
                prim = stack.pop()
                stack.extend(prim.GetChildren())
                
                mesh = None
                # 检查是否为显式网格
                if prim.IsA(UsdGeom.Mesh):
                    geom_mesh = UsdGeom.Mesh(prim)
                    vertices = np.array(geom_mesh.GetPointsAttr().Get())
                    if vertices.size > 0:
                        faces = np.array(geom_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                # 检查是否为隐式几何体
                elif prim.IsA(UsdGeom.Cube):
                    geom = UsdGeom.Cube(prim)
                    size = geom.GetSizeAttr().Get()
                    mesh = trimesh.primitives.Box(extents=[size, size, size])
                elif prim.IsA(UsdGeom.Sphere):
                    geom = UsdGeom.Sphere(prim)
                    radius = geom.GetRadiusAttr().Get()
                    mesh = trimesh.primitives.Sphere(radius=radius)
                elif prim.IsA(UsdGeom.Cylinder):
                    geom = UsdGeom.Cylinder(prim)
                    radius = geom.GetRadiusAttr().Get()
                    height = geom.GetHeightAttr().Get()
                    mesh = trimesh.primitives.Cylinder(radius=radius, height=height)
                elif prim.IsA(UsdGeom.Capsule):
                    geom = UsdGeom.Capsule(prim)
                    radius = geom.GetRadiusAttr().Get()
                    height = geom.GetHeightAttr().Get()
                    mesh = trimesh.primitives.Capsule(radius=radius, height=height)

                # 如果找到了任何类型的几何体，将其变换并添加到主网格中
                if mesh is not None:
                    mesh_to_world_transform = xform_cache.GetLocalToWorldTransform(prim)
                    root_to_world_transform = xform_cache.GetLocalToWorldTransform(root_prim)
                    world_to_root_transform = root_to_world_transform.GetInverse()
                    relative_transform = mesh_to_world_transform * world_to_root_transform
                    
                    # 将Gf.Matrix4d转换为numpy数组以供trimesh使用
                    transform_np = np.array(relative_transform).T
                    mesh.apply_transform(transform_np)
                    
                    combined_mesh += mesh

            if len(combined_mesh.vertices) == 0:
                print(f"[VIZ-WARN] No mesh geometry found under: {prim_path}")
                return None
            
            print(f"[VIZ-INFO] Extracted mesh from {prim_path} with {len(combined_mesh.vertices)} vertices.")
            return combined_mesh

        except Exception as e:
            print(f"[VIZ-ERROR] Failed to extract mesh from {prim_path}. Error: {e}")
            return None
    

    def _initialize_sdf(self):
        """使用trimesh和pysdf为球体创建一个SDF对象。"""
        print("[INFO] Initializing SDF for the sphere...")
        # 1. 使用 trimesh 创建一个球体网格
        self.peg_sdf = None
        peg_mesh = self._extract_mesh_from_prim("/World/envs/env_0/HeldAsset")

        # 2. 使用 pysdf 从网格顶点和面创建 SDF 对象
        self.sphere_sdf = SDF(peg_mesh.vertices, peg_mesh.faces)
        print("[INFO] SDF initialized.")
        
        self.sphere_wp_mesh = wp.Mesh(
            points=wp.array(peg_mesh.vertices, dtype=wp.vec3, device=self.device),
            indices=wp.array(peg_mesh.faces.reshape(-1), dtype=wp.int32, device=self.device),
            support_winding_number=True,
        )

    def calculate_normal_shear_force(self) -> tuple[torch.Tensor, torch.Tensor]:
        """在每个仿真步骤中被调用，以计算并施加触觉力。"""
        # 1. 获取球体和立方体的当前姿态
        
        sphere_pos_w = self._peg.data.root_pos_w
        sphere_quat_w = self._peg.data.root_quat_w
        sphere_linvel_w = self._peg.data.root_lin_vel_w
        sphere_angvel_w = self._peg.data.root_ang_vel_w
        # import pdb; pdb.set_trace()
        cube_pos_w = self._robot.data.body_pos_w[:, self.left_finger_idx]
        cube_quat_w = self._robot.data.body_quat_w[:, self.left_finger_idx]
        cube_linvel_w = self._robot.data.body_lin_vel_w[:, self.left_finger_idx]
        cube_angvel_w = self._robot.data.body_ang_vel_w[:, self.left_finger_idx]
        
        
        # 2. 将局部触觉点转换到世界坐标系
        tactile_points_world = tf_apply(
            cube_quat_w, cube_pos_w, self.tactile_points_left_local
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

            # write a visaulization using trimesh: peg and points_np
            # scene = trimesh.Scene()
            # scene.add_geometry(self.sphere_sdf)
            # scene.add_geometry(trimesh.PointCloud(points_np, colors=[0, 0, 255]))
            # scene.show()

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

            # grad_np = np.stack([grad_x, grad_y, grad_z], axis=-1)
            # grad_np = np.stack([grad_y, grad_x, grad_z], axis=-1)
            grad_np = np.stack([grad_x, grad_y, grad_z], axis=-1)
            # grad_np = np.stack([grad_z, grad_y, grad_z], axis=-1)
            # grad_np = np.stack([grad_z, grad_x, grad_y], axis=-1)
            grad = torch.from_numpy(grad_np).to(self.device)

            sdf_normals_local = torch.zeros_like(self.tactile_points_left_local)
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

        
        save_depth_image = True
        if save_depth_image:
            depth_image = penetration_depth.view((self.num_rows, self.num_cols)).cpu().numpy()
            # depth_image = penetration_depth.view((self.num_cols, self.num_rows)).cpu().numpy()
            # import pdb; pdb.set_trace()
            cv2.imwrite(os.path.join(r"C:\Users\jiuer\OneDrive - University of Virginia\Desktop\isaac", "depth_image.png"), (depth_image * 25500.0 * 5).astype(np.uint8))
        
        if not torch.any(contact_mask):
            return
        # -- 使用中心差分法更精确地计算梯度 --

        # finish the query collision
        
        tactile_points_world_velocity = torch.cross(cube_angvel_w.unsqueeze(1).expand((self.num_envs, self.num_tactile_points, 3)), math_utils.quat_apply(sphere_quat_w, self.tactile_points_left_local), dim = -1) + cube_linvel_w.expand((self.num_envs, self.num_tactile_points, 3))

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
            # cv2.imshow("tactile_shear_image", tactile_image)
            # cv2.waitKey(1)


    def update(self):
        """在每个仿真步骤中被调用，以计算并施加触觉力。"""

        self.calculate_normal_shear_force()
        return

        # 1. 获取球体和立方体的当前姿态
        # sphere_pos_w = self.env._held_asset.data.root_pos_w
        # sphere_quat_w = self.env._held_asset.data.root_quat_w
        # sphere_linvel_w = self.env._held_asset.data.root_lin_vel_w
        # sphere_angvel_w = self.env._held_asset.data.root_ang_vel_w

        # left_finger_pos_w, left_finger_quat_w = self._robot.data.body_pos_w[:, self.left_finger_idx], self._robot.data.body_quat_w[:, self.left_finger_idx]
        # left_finger_lin_vel_w, left_finger_ang_vel_w = self._robot.data.body_lin_vel_w[:, self.left_finger_idx] , self._robot.data.body_ang_vel_w[:, self.left_finger_idx]

        # right_finger_pos_w, right_finger_quat_w = self._robot.data.body_pos_w[:, self.right_finger_idx], self._robot.data.body_quat_w[:, self.right_finger_idx]
        # right_finger_lin_vel_w, right_finger_ang_vel_w = self._robot.data.body_lin_vel_w[:, self.right_finger_idx] , self._robot.data.body_ang_vel_w[:, self.right_finger_idx]

        # # cube_pos_w = left_finger_pos_w
        # # cube_quat_w = left_finger_quat_w
        # # cube_linvel_w = left_finger_lin_vel_w
        # # cube_angvel_w = left_finger_ang_vel_w
        
        # cube_pos_w = right_finger_pos_w
        # cube_quat_w = right_finger_quat_w
        # cube_linvel_w = right_finger_lin_vel_w
        # cube_angvel_w = right_finger_ang_vel_w
        
        # self.tactile_points_local = self.tactile_points_left_local

        # # 2. 将局部触觉点转换到世界坐标系
        # tactile_points_world = tf_apply(
        #     cube_quat_w, cube_pos_w, self.tactile_points_local
        # )

        # # 3. 将世界坐标系中的点转换到球体的局部坐标系
        # sphere_pose_inv = tf_inverse(sphere_quat_w, sphere_pos_w)
        # tactile_points_sphere_local = tf_apply(
        #     sphere_pose_inv[0], sphere_pose_inv[1], tactile_points_world
        # )

        # # 4. 查询 SDF 以获取穿透深度
        # points_np = tactile_points_sphere_local.cpu().numpy().squeeze(0)
        
        # if self.depth_calculation_method == "pysdf":
        #     distances_np = self.sphere_sdf(
        #         points_np
        #     )  # pysdf: positive outside, negative inside
        #     sdf_penetration_depth_np = -np.minimum(-distances_np, 0)
        #     sdf_penetration_depth = torch.from_numpy(sdf_penetration_depth_np).to(self.device)
        #     # import pdb; pdb.set_trace()
        #     # 5. 计算法向力
        #     contact_mask = sdf_penetration_depth > 0
        #     eps = 1e-6
        #     # points_in_contact_np = points_np[contact_mask.cpu().numpy()]
        #     grad_x = (
        #         self.sphere_sdf(points_np + np.array([eps, 0, 0]))
        #         - self.sphere_sdf(points_np - np.array([eps, 0, 0]))
        #     ) / (2 * eps)
        #     grad_y = (
        #         self.sphere_sdf(points_np + np.array([0, eps, 0]))
        #         - self.sphere_sdf(points_np - np.array([0, eps, 0]))
        #     ) / (2 * eps)
        #     grad_z = (
        #         self.sphere_sdf(points_np + np.array([0, 0, eps]))
        #         - self.sphere_sdf(points_np - np.array([0, 0, eps]))
        #     ) / (2 * eps)

        #     grad_np = np.stack([grad_x, grad_y, grad_z], axis=-1)
        #     grad = torch.from_numpy(grad_np).to(self.device)

        #     sdf_normals_local = torch.zeros_like(self.tactile_points_local)
        #     sdf_normals_local[:] = -math_utils.normalize(grad)
            
        #     penetration_depth = sdf_penetration_depth
        #     normals_local = sdf_normals_local
        # elif self.depth_calculation_method == "warp":
        #     pts_wp   = wp.array(points_np, dtype=wp.vec3f,  device=self.device)
        #     out_d_wp = wp.empty(pts_wp.shape[0], dtype=wp.float32, device=self.device)
        #     out_m_wp = wp.empty(pts_wp.shape[0], dtype=wp.int32,   device=self.device)
        #     out_n_wp = wp.empty(pts_wp.shape[0], dtype=wp.vec3f,   device=self.device)
        #     wp.launch(
        #         kernel=_warp_depth_normal_kernel,
        #         dim=pts_wp.shape[0],
        #         inputs=[self.sphere_wp_mesh.id, pts_wp, out_d_wp, out_m_wp, out_n_wp],
        #         device=self.device,
        #     )
        #     wp_distances_np       = torch.from_numpy(out_d_wp.numpy()).to(self.device).reshape(self.num_envs, self.num_tactile_points).clamp(min=0.0)   # (B,N)
        #     wp_normals_local  = torch.from_numpy(out_n_wp.numpy()).to(self.device).reshape(self.num_envs, self.num_tactile_points, 3)              # (B,N,3)
        #     contact_mask = wp_distances_np > 0
            
        #     penetration_depth = wp_distances_np
        #     normals_local = -wp_normals_local

        

        
        # if not torch.any(contact_mask):
        #     return
        # # -- 使用中心差分法更精确地计算梯度 --
        

        # # finish the query collision
        
        # tactile_points_world_velocity = torch.cross(cube_angvel_w.unsqueeze(1).expand((self.num_envs, self.num_tactile_points, 3)), math_utils.quat_apply(sphere_quat_w, self.tactile_points_local), dim = -1) + cube_linvel_w.expand((self.num_envs, self.num_tactile_points, 3))

        # # points_in_contact_local = tactile_points_sphere_local.squeeze(0)[contact_mask]
        
        # # penetration_depth_in_contact = penetration_depth[contact_mask]
        # # normals_in_contact = normals_local[contact_mask]
        # # 从穿透点沿法线方向移回穿透深度，得到表面上的点
        # closest_points_local = (tactile_points_sphere_local+ penetration_depth.unsqueeze(-1) *normals_local)
        
        # closest_points_world = tf_apply(sphere_quat_w, sphere_pos_w, closest_points_local)



        # normal_world = math_utils.quat_apply(sphere_quat_w, normals_local)
        
        
        # closest_points_velocity_world = (torch.cross(sphere_angvel_w.unsqueeze(1).expand((self.num_envs, self.num_tactile_points, 3)),math_utils.quat_apply(sphere_pose_inv[0], closest_points_local),dim=-1)+ sphere_linvel_w.expand((self.num_envs, self.num_tactile_points, 3)))
        
        
        # relative_velocity_world = tactile_points_world_velocity - closest_points_velocity_world

        # vt_world = relative_velocity_world - normal_world * torch.sum(normal_world * relative_velocity_world, dim=-1, keepdim=True)

        

        # depth, depth_dot, normal_world, vt_world = penetration_depth, "", normal_world, vt_world

        # fc_norm = self.tactile_kn * depth #- self.tactile_damping * depth_dot * depth
        # fc_world = fc_norm.unsqueeze(-1) * normal_world
        
        # '''compute frictional force'''
        # vt_norm = vt_world.norm(dim=-1)
        # ft_static_norm = self.tactile_kt * vt_norm
        # ft_dynamic_norm = self.tactile_mu * fc_norm
        # ft_world = - torch.minimum(ft_static_norm, ft_dynamic_norm).unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9, max=None).unsqueeze(-1)
        # # ft_world = -ft_dynamic_norm.unsqueeze(-1) * vt_world / vt_norm.clamp(min=1e-9, max=None).unsqueeze(-1)
        # '''net tactile force'''
        # tactile_force_world = fc_world + ft_world
        
        # '''tactile force in tactile frame'''
        # quat_pad_inv = math_utils.quat_conjugate(cube_quat_w)
        # tactile_force_pad = math_utils.quat_apply(quat_pad_inv.unsqueeze(1).expand(self.num_envs, self.num_tactile_points, 4), tactile_force_world)
        
        # UnitX = torch.tensor([1., 0., 0.], device=self.device)
        # UnitY = torch.tensor([0., 1., 0.], device=self.device)
        # UnitZ = torch.tensor([0., 0., -1.], device=self.device)
        # tactile_normal_axis = math_utils.quat_apply(self.tactile_quat_local, UnitZ.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        # tactile_shear_x_axis = math_utils.quat_apply(self.tactile_quat_local, UnitX.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        # tactile_shear_y_axis = math_utils.quat_apply(self.tactile_quat_local, UnitY.unsqueeze(0).unsqueeze(0).expand(self.num_envs, self.num_tactile_points, 3))
        
        # tactile_normal_force = -(tactile_normal_axis * tactile_force_pad).sum(-1)
        # tactile_shear_force_x = (tactile_shear_x_axis * tactile_force_pad).sum(-1)
        # tactile_shear_force_y = (tactile_shear_y_axis * tactile_force_pad).sum(-1)
        # tactile_shear_force = torch.cat((tactile_shear_force_x.unsqueeze(-1), tactile_shear_force_y.unsqueeze(-1)), dim=-1)
        # if self.enable_visualization:
        #     tactile_image = visualize_tactile_shear_image(tactile_normal_force[0].view((self.num_rows, self.num_cols)).cpu().numpy(), tactile_shear_force[0].view((self.num_rows, self.num_cols, 2)).cpu().numpy(), normal_force_threshold=0.003, shear_force_threshold=0.001)
        #     # import pdb; pdb.set_trace()
        #     # cv2.imwrite(os.path.join(r"C:\onedrive\OneDrive - University of Virginia\Desktop\isaac", "tactile_shear_image.png"), (visualize_tactile_shear_image(tactile_normal_force[0].view((self.num_rows, self.num_cols)).cpu().numpy(), tactile_shear_force[0].view((self.num_rows, self.num_cols, 2)).cpu().numpy(), normal_force_threshold=0.003, shear_force_threshold=0.002)*255.0).astype(np.uint8))
        #     cv2.imwrite(os.path.join(r"C:\Users\jiuer\OneDrive - University of Virginia\Desktop\isaac", "tactile_shear_image.png"), (visualize_tactile_shear_image(tactile_normal_force[0].view((self.num_rows, self.num_cols)).cpu().numpy(), tactile_shear_force[0].view((self.num_rows, self.num_cols, 2)).cpu().numpy(), normal_force_threshold=0.003, shear_force_threshold=0.002)*255.0).astype(np.uint8))
        #     # cv2.imshow("tactile_shear_image", tactile_image)
        #     # cv2.waitKey(1)
        # # 8. 可视化力
        # if self.enable_visualization and False:  # Periodically visualize
        #     # -- MODIFICATION: Calculate contact patch for visualization --
        #     points_in_contact_local = tactile_points_sphere_local.squeeze(0)[
        #         contact_mask
        #     ]
        #     penetration_depth_in_contact = penetration_depth[contact_mask]
        #     normals_in_contact = normals_local[contact_mask]

        #     # 从穿透点沿法线方向移回穿透深度，得到表面上的点
        #     closest_points_local = (
        #         points_in_contact_local
        #         + penetration_depth_in_contact.unsqueeze(-1) * normals_in_contact
        #     )
        #     closest_points_world = tf_apply(
        #         sphere_quat_w, sphere_pos_w, closest_points_local.unsqueeze(0)
        #     )

        #     contact_points_world_viz = tactile_points_world.squeeze(0)[contact_mask]
        #     force_magnitudes = self.tactile_kn * penetration_depth
        #     forces_local = normals_local * force_magnitudes.unsqueeze(-1)
        #     force_vectors_to_visualize = forces_world[contact_mask]
        #     forces_world = math_utils.quat_apply(sphere_quat_w.repeat(self.num_tactile_points, 1), forces_local)
        #     self.visualize_forces(
        #         contact_points_world_viz,
        #         force_vectors_to_visualize,
        #         closest_points_world.squeeze(0),
        #     )

    def get_scene_entities(self):
        """返回此系统创建的场景实体。"""
        entities = {
            "cube": self.cube,
            "sphere": self.sphere,
            "tactile_system": self,
        }
        return entities

class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task
        self.initial_tactile_image = None
        super().__init__(cfg, render_mode, **kwargs)
        self.tactile_image_scale = 35
        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_img_save_path = os.path.join(current_dir, "..", "..", "..", "..", "..", "..")
        # self.tactile_system = TactileSystem(self)

    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        # Fixed asset.
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        import isaaclab.sim.schemas as schemas_utils

        from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg, spawn_rigid_body_material

        soft_material_cfg = RigidBodyMaterialCfg(
            compliant_contact_stiffness=6000.0,
            compliant_contact_damping=0.0
        )
        
        # 在场景中创建一个新的物理材质 Prim。路径可以自定义，"/World/Looks/" 是常用约定
        soft_material_path = "/World/Looks/SoftElastomerMaterial"
        spawn_rigid_body_material(prim_path=soft_material_path, cfg=soft_material_cfg)
        print(f"已创建自定义物理材质于: {soft_material_path}")

        # --- 步骤 2: 将新材质应用到每个环境的机器人手指上 ---
        
        self.sim.step() # 确保材质和机器人 Prim 都已加载

        # self._gripper_camera = Camera(self.cfg.gripper_camera)
        # self.scene.sensors["gripper_camera"] = self._gripper_camera

        # self._tactile_camera = Camera(self.cfg.tactile_camera)
        # self.scene.sensors["tactile_camera"] = self._tactile_camera

        

        # return
        # self.nominal_depth = self.scene.sensors["tactile_camera"].data.output["distance_to_image_plane"].clone()
        # import pdb; pdb.set_trace()
        # return
        for i in range(self.scene.num_envs):
            # 定义左右两个手指的 碰撞体 Prim 的路径/World/envs/env_0/Robot/franka_bak/elastomer_right
            paths_to_modify = [
                f"/World/envs/env_{i}/Robot/franka_bak/elastomer_right/collisions",
                f"/World/envs/env_{i}/Robot/franka_bak/elastomer_left/collisions"
            ]
            from pxr import UsdShade, UsdPhysics
            for path in paths_to_modify:
                prim = self.scene.stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    print(f"警告: 未找到 Prim: {path}")
                    continue

                # (a) 确保碰撞是启用的 (如果之前禁用了)

                # (b) 将我们创建的物理材质“绑定”到这个碰撞体上
                # 这是实现您需求的核心步骤
                material_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                material_binding_api.Bind(
                    UsdShade.Material.Get(self.scene.stage, soft_material_path),
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )
                
                # print(f"成功将材质 '{soft_material_path}' 应用到: {path}")


        # collider_cfg = schemas_utils.CollisionPropertiesCfg(
        #     collision_enabled=True,   # 启用碰撞
        #     contact_offset=0.05,      # 设置 contact offset (单位:米)
        #     rest_offset=0.01        # 设置 rest offset (单位:米)
        # )
        # # 提醒：你需要调整 contact_offset 和 rest_offset 的值来获得期望的“柔度”
        
        # # 2. 遍历所有环境，修改每个机器人手指的碰撞属性
        # for i in range(self.scene.num_envs):
        #     # 定义左右两个手指的 碰撞体 Prim 的路径
        #     paths_to_modify = [
        #         f"/World/envs/env_{i}/Robot/elastomer_right/collisions",
        #         f"/World/envs/env_{i}/Robot/elastomer_left/collisions"
        #     ]

        #     for path in paths_to_modify:
        #         # 3. 调用 Isaac Lab 的官方工具函数来修改属性
        #         was_modified = schemas_utils.modify_collision_properties(
        #             prim_path=path,
        #             cfg=collider_cfg,
        #             stage=self.scene.stage
        #         )
                
        #         # 添加打印信息用于调试
        #         if was_modified:
        #             print(f"成功使用官方API修改属性: {path}")
        #         else:
        #             # 根据文档，如果 prim 不存在或没有应用 schema，此函数会返回 False
        #             print(f"警告: modify_collision_properties 未成功作用于 {path}")

        # ==================================================================
    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()
        
        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions
        action =  action[:,0:6]
        # action[:,0] = 1.0
        # print("action", action)
        # self.actions = (
        #     self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        # )
        self.actions = action
        # lift_action = torch.tensor([[0,0,1.0,0,0,0]], device=self.device)
        # down_action = torch.tensor([[0,0,-1.0,0,0,0]], device=self.device)
        # self.actions = lift_action
        

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )


    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)
        # tactile_data = self.tactile_system.update()
        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    # def _apply_action(self):
    #     """Apply actions for policy as delta targets from current position."""
    #     # Note: We use finite-differenced velocities for control and observations.
    #     # Check if we need to re-compute velocities within the decimation loop.
    #     if self.last_update_timestamp < self._robot._data._sim_timestamp:
    #         self._compute_intermediate_values(dt=self.physics_dt)

    #     # Interpret actions as target pos displacements and set pos target
    #     pos_actions = self.actions[:, 0:3] * self.pos_threshold

    #     # Interpret actions as target rot (axis-angle) displacements
    #     rot_actions = self.actions[:, 3:6]
    #     if self.cfg_task.unidirectional_rot:
    #         rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
    #     rot_actions = rot_actions * self.rot_threshold

    #     ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
    #     # To speed up learning, never allow the policy to move more than 5cm away from the base.
    #     fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
    #     delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
    #     pos_error_clipped = torch.clip(
    #         delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
    #     )
    #     ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

    #     # Convert to quat and set rot target
    #     angle = torch.norm(rot_actions, p=2, dim=-1)
    #     axis = rot_actions / angle.unsqueeze(-1)

    #     rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
    #     rot_actions_quat = torch.where(
    #         angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
    #         rot_actions_quat,
    #         torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
    #     )
    #     ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

    #     target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
    #     target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
    #     target_euler_xyz[:, 1] = 0.0

    #     ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
    #         roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
    #     )
    #     # print("ctrl_target_fingertip_midpoint_pos, self.fingertip_midpoint_pos", ctrl_target_fingertip_midpoint_pos, self.fingertip_midpoint_pos)
    #     self.generate_ctrl_signals(
    #         ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
    #         ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
    #         ctrl_target_gripper_dof_pos=0.0,
    #     )

    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh" or self.cfg_task.name == "lighter":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Keep track of episode statistics and log rewards."""
        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        offsets = factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # Action penalties.
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

        rew_dict = {
            "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    def initialize_tactile_image(self):
        if(self.initial_tactile_image is None):
            self.initial_tactile_image = self.scene.sensors["tactile_camera"].data.output["distance_to_image_plane"].clone().transpose(1, 3).transpose(2, 3)
            torchvision.utils.save_image((self.initial_tactile_image - self.initial_tactile_image.min()) / (self.initial_tactile_image.max() - self.initial_tactile_image.min()), os.path.join(self.log_img_save_path, "initial_tactile_image.png" ) )
            self.initial_rgb_image = self.scene.sensors["tactile_camera"].data.output["rgb"].clone()
            torchvision.utils.save_image(self.initial_rgb_image.transpose(1, 3).transpose(2, 3) / 255.0, os.path.join(self.log_img_save_path, "initial_rgb_image.png" ) )

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        # self.initialize_tactile_image()
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
    ):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = factory_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "lighter":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = factory_utils.get_held_base_pos_local(
                self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        if self.cfg_task.name == "nut_thread":
            # Rotate along z-axis of frame for default position.
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)
        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets only.

        This method should only be called during resets when all environments
        reset at the same time.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (c) iterative IK Method
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1
            break

        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 2] -= 0.01
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
