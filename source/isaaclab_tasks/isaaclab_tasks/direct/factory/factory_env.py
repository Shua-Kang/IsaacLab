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

from . import factory_control as fc
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv
from isaacsim.core.utils.torch.transformations import tf_apply, tf_inverse

import trimesh
from pysdf import SDF
import os
import rtree
from isaaclab.sensors import TiledCamera, Camera

import numpy as np
import cv2
def visualize_tactile_shear_image(tactile_normal_force, tactile_shear_force,
                                  normal_force_threshold=0.00008, shear_force_threshold=0.0005,
                                  resolution=30):
    """
    Visualize the tactile shear field.

    Args:
        tactile_normal_force (np.ndarray): Array of tactile normal forces.
        tactile_shear_force (np.ndarray): Array of tactile shear forces.
        normal_force_threshold (float): Threshold for normal force visualization.
        shear_force_threshold (float): Threshold for shear force visualization.
        resolution (int): Resolution for the visualization.

    Returns:
        np.ndarray: Image visualizing the tactile shear forces.
    """
    nrows = tactile_normal_force.shape[0]
    ncols = tactile_normal_force.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)

    # print('(min, max) tactile normal force: ', np.min(tactile_normal_force), np.max(tactile_normal_force))
    try:
        for row in range(nrows):
            for col in range(ncols):
                loc0_x = row * resolution + resolution // 2
                loc0_y = col * resolution + resolution // 2
                loc1_x = loc0_x + tactile_shear_force[row, col][0] / shear_force_threshold * resolution
                loc1_y = loc0_y + tactile_shear_force[row, col][1] / shear_force_threshold * resolution
                color = (0.,
                        max(0., 1. - tactile_normal_force[row][col] / normal_force_threshold),
                        min(1., tactile_normal_force[row][col] / normal_force_threshold)
                        )

                cv2.arrowedLine(imgs_tactile,
                                (int(loc0_y), int(loc0_x)),
                                (int(loc1_y), int(loc1_x)),
                                color, 6, tipLength=0.4)
    except Exception as e:
        print(f"[VIZ-ERROR] Failed to visualize tactile shear image. Error: {e}")
        import pdb; pdb.set_trace()
        return None
    return imgs_tactile



class TactileSensingSystem:
    """
    一个管理机器人手指（传感器）和Peg（物体）之间触觉模拟的类。
    使用SDF来计算穿透深度并生成触觉数据。
    """

    def __init__(self, env: "FactoryEnv", num_rows_per_finger: int = 50, num_cols_per_finger: int = 50):
        """
        通过引用环境中的机器人和物体来初始化触觉系统。

        Args:
            env (FactoryEnv): 对主环境的引用。
            num_rows_per_finger (int): 每个手指上传感器点的行数。
            num_cols_per_finger (int): 每个手指上传感器点的列数。
        """
        print("[INFO] Initializing Tactile Sensing System...")
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        # 获取对机器人和被抓取物体的引用
        self._robot = self.env._robot
        self._peg = self.env._held_asset

        # 获取传感器（手指）的body索引
        self.left_finger_idx = self._robot.body_names.index("elastomer_left")
        self.right_finger_idx = self._robot.body_names.index("elastomer_right")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 注意：这里的相对路径'..'的数量可能需要根据您实际的文件结构进行调整
        self.peg_stl_path = os.path.join(current_dir, "..", "..", "..", "..", "..", "my_assets_new", "peg", "peg.stl")
        self.elastomer_stl_path = os.path.join(current_dir, "..", "..", "..", "..", "..", "my_assets_new", "peg", "extruded_elastomer_transformed.stl")
        print(f"[INFO] Peg STL path: {self.peg_stl_path}")
        print(f"[INFO] Elastomer STL path: {self.elastomer_stl_path}")

        self.num_rows_per_finger = num_rows_per_finger
        self.num_cols_per_finger = num_cols_per_finger
        # 在每个手指表面生成触觉点
        self._generate_tactile_points(num_rows=num_rows_per_finger, num_cols=num_cols_per_finger)

        # 为Peg物体初始化SDF
        self._initialize_peg_sdf()
        print("[INFO] Tactile Sensing System Initialized.")
        
        # --- 新增：可视化相关的控制参数 ---
        self.enable_tactile_visualization = False  # 总开关
        self.enable_debug_visualization = False
        self.visualization_counter = 0      # 帧计数器
        self.visualization_interval = 10   # 每隔多少帧显示一次
        self.colormap = plt.get_cmap("jet") # 用于生成热力图的颜色映射
        print("[INFO] Tactile Sensing System Initialized.")

        self.tactile_kn = 1 # 法向刚度 (N/m), 用于计算压力
        self.tactile_kt = 1   # 剪切刚度 (N*s/m), 用于计算剪切力
        
        # self.depth_camera = TiledCamera(self.env.cfg.TACTILE_CAMERA_CFG)
        # self.env.scene.sensors["tactile_camera"] = self.depth_camera
        
        
        # self.env.scene.add_camera("tactile_camera", self.env.cfg.TACTILE_CAMERA_CFG)
        

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
        
        

        # 假设左右手指使用相同的局部点云
        self.tactile_points_left_local = points.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.tactile_points_right_local = points.unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        self.num_points_per_finger = self.tactile_points_left_local.shape[1]

        
        print(f"[INFO] Generated {self.num_points_per_finger} tactile points successfully.")

    def _initialize_peg_sdf(self):
        """
        修改为从仿真环境中提取网格来初始化SDF。
        """
        print("[INFO] Initializing SDF for the Peg by extracting mesh from stage...")
        self.peg_sdf = None
        try:
            # 使用新的、更鲁棒的函数从环境中提取网格
            peg_mesh = self._extract_mesh_from_prim("/World/envs/env_0/HeldAsset")
            if peg_mesh is None:
                raise RuntimeError("Failed to extract mesh from HeldAsset for SDF.")
            
            self.peg_sdf = SDF(peg_mesh.vertices, peg_mesh.faces)
            print("[INFO] Peg SDF initialized from stage.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize SDF. Error: {e}")

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
    
    def _debug_visualize_transforms(self, local_l, local_r, world_l, world_r, peg_local, transforms, all_tactile_points_w):
        """
        一个专门用于调试坐标变换的3D可视化函数。
        """
        print("[DEBUG] Visualizing coordinate transforms... (Close window to continue)")
        
        scene = trimesh.Scene()
        
        # --- 1. 从环境中提取并添加上下文几何体 ---
        peg_mesh = self._extract_mesh_from_prim("/World/envs/env_0/HeldAsset")
        if peg_mesh:
            peg_mesh.visual.face_colors = [255, 255, 0, 150] # 黄色, 半透明
            scene.add_geometry(peg_mesh, transform=transforms["peg_w"])
        else:
            print("[VIZ-WARN] Could not visualize Peg mesh.")

        finger_mesh_template = self._load_mesh_from_file(self.elastomer_stl_path)
        if finger_mesh_template:
            finger_mesh_template.visual.face_colors = [128, 128, 128, 150] # 灰色, 半透明

            # 右手指 (直接使用)
            right_finger_mesh = finger_mesh_template.copy()
            scene.add_geometry(right_finger_mesh, transform=transforms["finger_r_w"])

            # 左手指 (进行Y轴镜像)
            left_finger_mesh = finger_mesh_template.copy()
            # mirror_transform = np.diag([1, -1, 1, 1])
            # left_finger_mesh.apply_transform(mirror_transform)
            scene.add_geometry(left_finger_mesh, transform=transforms["finger_l_w"])
        else:
            print("[VIZ-WARN] Could not visualize Finger meshes. Check file path in __init__.")

        # --- 2. 可视化点云 ---
        pc_local_l = trimesh.PointCloud(local_l, colors=[0, 255, 0])
        # pc_local_l.apply_transform(transforms["finger_l_w"])
        scene.add_geometry(pc_local_l)
        
        pc_local_r = trimesh.PointCloud(local_r, colors=[0, 255, 0])
        # pc_local_r.apply_transform(transforms["finger_r_w"])
        scene.add_geometry(pc_local_r)



         # --- 2. 可视化点云 ---
        pc_local_l = trimesh.PointCloud(world_l, colors=[0, 255, 255])
        # pc_local_l.apply_transform(transforms["finger_l_w"])
        scene.add_geometry(pc_local_l)
        
        pc_local_r = trimesh.PointCloud(world_r, colors=[0, 255, 255])
        # pc_local_r.apply_transform(transforms["finger_r_w"])
        scene.add_geometry(pc_local_r)

        # scene.add_geometry(trimesh.PointCloud(np.vstack([world_l, world_r]), colors=[0, 0, 255]))

        pc_peg_local = trimesh.PointCloud(peg_local, colors=[255, 0, 0])
        # pc_peg_local.apply_transform(transforms["peg_w"])
        scene.add_geometry(pc_peg_local)
        
        # --- 3. 添加世界坐标系轴，方便定位 ---
        world_axes = trimesh.creation.axis(origin_size=0.005, axis_radius=0.001, axis_length=0.05)
        scene.add_geometry(world_axes)

        scene.show()

    def _visualize_tactile_contact(self, tactile_points_w, tactile_image, all_tactile_points_peg_local, transforms):
        """
        一个更直观的可视化函数，显示接触点、表面投影和穿透连线。
        """
        print("[DEBUG] Visualizing tactile contact... (Close window to continue)")
        
        # --- 1. 准备数据 (仅 env 0) ---
        points_w = tactile_points_w[0].cpu().numpy()
        depths = tactile_image[0].cpu().numpy()
        points_peg_local = all_tactile_points_peg_local[0].cpu().numpy()

        contact_mask = depths > 1e-6 # 过滤有接触的点
        if not np.any(contact_mask):
            print("[DEBUG] No contact to visualize.")
            return

        contact_points_w = points_w[contact_mask]
        contact_points_peg_local = points_peg_local[contact_mask]
        contact_depths = depths[contact_mask]

        # --- 2. 计算表面法线和投影点 ---
        # SDF的梯度是法线方向
        eps = 1e-6
        grad_x = self.peg_sdf(contact_points_peg_local + np.array([eps, 0, 0])) - self.peg_sdf(contact_points_peg_local - np.array([eps, 0, 0]))
        grad_y = self.peg_sdf(contact_points_peg_local + np.array([0, eps, 0])) - self.peg_sdf(contact_points_peg_local - np.array([0, eps, 0]))
        grad_z = self.peg_sdf(contact_points_peg_local + np.array([0, 0, eps])) - self.peg_sdf(contact_points_peg_local - np.array([0, 0, eps]))
        
        grad = np.stack([grad_x, grad_y, grad_z], axis=-1) / (2 * eps)
        contact_normals_peg_local = -grad
        # 归一化法线
        norms = np.linalg.norm(contact_normals_peg_local, axis=1, keepdims=True)
        contact_normals_peg_local /= np.where(norms == 0, 1e-6, norms)

        # 表面点 = 接触点 + 穿透深度 * 法线 (都在Peg局部坐标系中)
        surface_points_peg_local = contact_points_peg_local + contact_depths[:, np.newaxis] * contact_normals_peg_local

        # 将表面点转换回世界坐标系
        peg_transform = transforms["peg_w"]
        surface_points_w = trimesh.transform_points(surface_points_peg_local, peg_transform)

        # --- 3. 创建可视化场景 ---
        scene = trimesh.Scene()
        
        # (a) 添加上下文模型
        peg_mesh = self._extract_mesh_from_prim("/World/envs/env_0/HeldAsset")
        if peg_mesh:
            peg_mesh.visual.face_colors = [255, 255, 0, 150] # 黄色, 半透明
            scene.add_geometry(peg_mesh, transform=transforms["peg_w"])
        else:
            print("[VIZ-WARN] Could not visualize Peg mesh.")

        finger_mesh_template = self._load_mesh_from_file(self.elastomer_stl_path)
        if finger_mesh_template:
            # finger_mesh_template.apply_translation(-finger_mesh_template.centroid)
            finger_mesh_template.visual.face_colors = [128, 128, 128, 100] # 灰色, 更透明
            right_finger_mesh = finger_mesh_template.copy()
            scene.add_geometry(right_finger_mesh, transform=transforms["finger_r_w"])
            left_finger_mesh = finger_mesh_template.copy()
            # left_finger_mesh.apply_transform(np.diag([1, -1, 1, 1]))
            scene.add_geometry(left_finger_mesh, transform=transforms["finger_l_w"])

        # (b) 添加点云和连线
        scene.add_geometry(trimesh.PointCloud(contact_points_w, colors=[0, 0, 255])) # 蓝色: 接触点
        scene.add_geometry(trimesh.PointCloud(surface_points_w, colors=[255, 0, 0])) # 红色: 表面点

        # 创建穿透连线
        lines = np.hstack([contact_points_w, surface_points_w]).reshape(-1, 2, 3)
        # 根据深度着色
        max_depth = 0.005 # 预期的最大穿透深度，用于颜色映射
        normalized_depths = np.clip(contact_depths / max_depth, 0, 1)

        #use opencv to visualize normalized_depths as depth image
        

        line_colors = (self.colormap(normalized_depths) * 255).astype(np.uint8)
        
        path_visual = trimesh.load_path(lines, colors=line_colors)
        scene.add_geometry(path_visual)
        
        # (c) 添加世界坐标系轴
        scene.add_geometry(trimesh.creation.axis(origin_size=0.005, axis_radius=0.001, axis_length=0.05))

        scene.show()

    # def update(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     self.visualization_counter += 1
    #     if self.peg_sdf is None:
    #         num_total_points = 2 * self.num_points_per_finger
    #         return torch.zeros(self.num_envs, num_total_points, device=self.device), \
    #                torch.zeros(self.num_envs, num_total_points, 2, device=self.device)

    #     # --- 步骤 1: 获取所有位姿和速度 ---
    #     peg_pos_w, peg_quat_w = self._peg.data.root_pos_w, self._peg.data.root_quat_w
    #     peg_lin_vel_w, peg_ang_vel_w = self._peg.data.root_lin_vel_w, self._peg.data.root_ang_vel_w

    #     left_finger_pos_w, left_finger_quat_w = self._robot.data.body_pos_w[:, self.left_finger_idx], self._robot.data.body_quat_w[:, self.left_finger_idx]
    #     left_finger_lin_vel_w, left_finger_ang_vel_w = self._robot.data.body_lin_vel_w[:, self.left_finger_idx], self._robot.data.body_ang_vel_w[:, self.left_finger_idx]
        
    #     right_finger_pos_w, right_finger_quat_w = self._robot.data.body_pos_w[:, self.right_finger_idx], self._robot.data.body_quat_w[:, self.right_finger_idx]
    #     right_finger_lin_vel_w, right_finger_ang_vel_w = self._robot.data.body_lin_vel_w[:, self.right_finger_idx], self._robot.data.body_ang_vel_w[:, self.right_finger_idx]
        
    #     # --- 步骤 2: 计算触觉点的位置和速度 ---
    #     tactile_points_left_w = tf_apply(left_finger_quat_w, left_finger_pos_w, self.tactile_points_left_local)
    #     tactile_points_right_w = tf_apply(right_finger_quat_w, right_finger_pos_w, self.tactile_points_right_local)
    #     all_tactile_points_w = torch.cat([tactile_points_left_w, tactile_points_right_w], dim=1)

    #     r_left = tactile_points_left_w - left_finger_pos_w.unsqueeze(1)
    #     tactile_vel_left_w = left_finger_lin_vel_w.unsqueeze(1) + torch.cross(left_finger_ang_vel_w.unsqueeze(1), r_left, dim=-1)
    #     r_right = tactile_points_right_w - right_finger_pos_w.unsqueeze(1)
    #     tactile_vel_right_w = right_finger_lin_vel_w.unsqueeze(1) + torch.cross(right_finger_ang_vel_w.unsqueeze(1), r_right, dim=-1)
    #     all_tactile_vel_w = torch.cat([tactile_vel_left_w, tactile_vel_right_w], dim=1)

    #     # --- 步骤 3: 计算穿透深度 (法向力) ---
    #     peg_pose_inv_quat, peg_pose_inv_pos = tf_inverse(peg_quat_w, peg_pos_w)
    #     all_tactile_points_peg_local = tf_apply(peg_pose_inv_quat, peg_pose_inv_pos, all_tactile_points_w)
        
    #     batch_size, num_points, _ = all_tactile_points_peg_local.shape
    #     points_np = all_tactile_points_peg_local.view(-1, 3).cpu().numpy()
    #     distances_np = self.peg_sdf(points_np)

    #     penetration_depth = torch.from_numpy(-np.minimum(-distances_np, 0)).to(self.device).view(batch_size, num_points)
        
    #     normal_forces = self.tactile_kn * penetration_depth

    #     # --- 步骤 4: 计算剪切力 ---
    #     shear_forces = torch.zeros(batch_size, num_points, 2, device=self.device)
    #     contact_mask = penetration_depth > 1e-6

    #     # 使用循环处理每个环境，以简化张量操作
    #     for i in range(batch_size):
    #         env_mask = contact_mask[i]
    #         if not torch.any(env_mask):
    #             continue
            
    #         contact_points_peg_local = all_tactile_points_peg_local[i, env_mask]
            
    #         # (a) 计算法线
    #         eps = 1e-6
    #         grad_x = self.peg_sdf(contact_points_peg_local.cpu().numpy() + np.array([eps, 0, 0])) - self.peg_sdf(contact_points_peg_local.cpu().numpy() - np.array([eps, 0, 0]))
    #         grad_y = self.peg_sdf(contact_points_peg_local.cpu().numpy() + np.array([0, eps, 0])) - self.peg_sdf(contact_points_peg_local.cpu().numpy() - np.array([0, eps, 0]))
    #         grad_z = self.peg_sdf(contact_points_peg_local.cpu().numpy() + np.array([0, 0, eps])) - self.peg_sdf(contact_points_peg_local.cpu().numpy() - np.array([0, 0, eps]))
    #         grad = torch.from_numpy(np.stack([grad_x, grad_y, grad_z], axis=-1)).to(self.device) / (2 * eps)
    #         contact_normals_local = -torch.nn.functional.normalize(grad, p=2, dim=-1)

    #         # (b) 将法线旋转到世界坐标系
    #         num_contact_points = contact_normals_local.shape[0]
    #         peg_quat_repeated = peg_quat_w[i].unsqueeze(0).expand(num_contact_points, -1)
    #         contact_normals_w = quat_apply(peg_quat_repeated, contact_normals_local)
            
    #         # (c) 计算Peg表面点的速度
    #         surface_points_w = all_tactile_points_w[i, env_mask] - penetration_depth[i, env_mask].unsqueeze(-1) * contact_normals_w
    #         r_peg = surface_points_w - peg_pos_w[i]
    #         # surface_vel_w = peg_lin_vel_w[i] + torch.cross(peg_ang_vel_w[i], r_peg, dim=-1)
    #         surface_vel_w = peg_lin_vel_w[i].unsqueeze(0) + torch.cross(peg_ang_vel_w[i].unsqueeze(0), r_peg, dim=-1)

    #         # (d) 计算切向相对速度
    #         relative_vel_w = all_tactile_vel_w[i, env_mask] - surface_vel_w
    #         normal_vel_w = torch.sum(relative_vel_w * contact_normals_w, dim=-1, keepdim=True) * contact_normals_w
    #         tangential_vel_w = relative_vel_w - normal_vel_w
            
    #         # (e) 计算3D剪切力
    #         shear_force_3d = -self.tactile_kt * tangential_vel_w
            
    #         # (f) 将3D剪切力投影到2D传感器平面
    #         world_y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand_as(contact_normals_w)
    #         sensor_x_axis = torch.nn.functional.normalize(torch.cross(world_y_axis, contact_normals_w, dim=-1), p=2, dim=-1)
    #         sensor_y_axis = torch.cross(contact_normals_w, sensor_x_axis, dim=-1)
            
    #         shear_force_2d_x = torch.sum(shear_force_3d * sensor_x_axis, dim=-1)
    #         shear_force_2d_y = torch.sum(shear_force_3d * sensor_y_axis, dim=-1)
            
    #         shear_forces[i, env_mask] = torch.stack([shear_force_2d_x, shear_force_2d_y], dim=-1)

    #         left_normal_forces = normal_forces[i, :self.num_points_per_finger].view(self.num_rows_per_finger, self.num_cols_per_finger)
    #         right_normal_forces = normal_forces[i, self.num_points_per_finger:].view(self.num_rows_per_finger, self.num_cols_per_finger)

    #         left_shear_forces = shear_forces[i, :self.num_points_per_finger,:].view(self.num_rows_per_finger, self.num_cols_per_finger,2)
    #         right_shear_forces = shear_forces[i, self.num_points_per_finger:,:].view(self.num_rows_per_finger, self.num_cols_per_finger,2)

    #         if(isinstance(left_normal_forces, torch.Tensor)):
    #             left_normal_forces = left_normal_forces.cpu().numpy()
    #             left_shear_forces = left_shear_forces.cpu().numpy()
    #             left_shear_forces = left_shear_forces * 0.0
                
    #         else:
    #             print("[VIZ-WARN] Failed to visualize tactile shear image. left_normal_forces is not a tensor.")
    #         import pdb; pdb.set_trace()
    #         img = visualize_tactile_shear_image(left_normal_forces, left_shear_forces, normal_force_threshold=0.008, shear_force_threshold=0.1, resolution=30)
    #         cv2.imshow("left_tactile_shear_image", img)
    #         cv2.waitKey(1)
    #         # import pdb; pdb.set_trace()
    #     # visualize_tactile_shear_image(right_normal_forces, right_shear_forces, normal_force_threshold=0.00008, shear_force_threshold=0.0005, resolution=30)
    #     return normal_forces, shear_forces

    def update(self) -> torch.Tensor:
        self.visualization_counter += 1

        if self.peg_sdf is None:
            return torch.zeros(self.num_envs, 2 * self.num_points_per_finger, device=self.device)

        # --- 1. 获取所有位姿 ---
        peg_pos_w, peg_quat_w = self._peg.data.root_pos_w, self._peg.data.root_quat_w
        left_finger_pos_w, left_finger_quat_w = self._robot.data.body_pos_w[:, self.left_finger_idx], self._robot.data.body_quat_w[:, self.left_finger_idx]
        right_finger_pos_w, right_finger_quat_w = self._robot.data.body_pos_w[:, self.right_finger_idx], self._robot.data.body_quat_w[:, self.right_finger_idx]
        
        # --- 2. 执行坐标变换 ---
        tactile_points_left_w = tf_apply(left_finger_quat_w, left_finger_pos_w, self.tactile_points_left_local)
        tactile_points_right_w = tf_apply(right_finger_quat_w, right_finger_pos_w, self.tactile_points_right_local)

        all_tactile_points_w = torch.cat([tactile_points_left_w, tactile_points_right_w], dim=1)
        peg_pose_inv_quat, peg_pose_inv_pos = tf_inverse(peg_quat_w, peg_pos_w)
        all_tactile_points_peg_local = tf_apply(peg_pose_inv_quat, peg_pose_inv_pos, all_tactile_points_w)

        # --- 3. 检查是否需要进行调试可视化 ---
        if self.enable_debug_visualization and self.visualization_counter % self.visualization_interval == 0:
            # -- FIX: Manually construct transformation matrices from pos and quat --
            # Helper function to create a 4x4 matrix
            def create_transform_matrix(pos_np, quat_np_wxyz):
                # trimesh expects quaternion as [w, x, y, z]
                quat_np_wxyz = np.array([quat_np_wxyz[0], quat_np_wxyz[1], quat_np_wxyz[2], quat_np_wxyz[3]])
                matrix = trimesh.transformations.quaternion_matrix(quat_np_wxyz)
                matrix[:3, 3] = pos_np
                return matrix

            # Prepare data for env 0
            peg_pos_np = self._peg.data.root_pos_w[0].cpu().numpy()
            peg_quat_np = self._peg.data.root_quat_w[0].cpu().numpy()
            
            finger_l_pos_np = self._robot.data.body_pos_w[0, self.left_finger_idx].cpu().numpy()
            finger_l_quat_np = self._robot.data.body_quat_w[0, self.left_finger_idx].cpu().numpy()

            finger_r_pos_np = self._robot.data.body_pos_w[0, self.right_finger_idx].cpu().numpy()
            finger_r_quat_np = self._robot.data.body_quat_w[0, self.right_finger_idx].cpu().numpy()

            # Create the dictionary of matrices
            transforms = {
                "peg_w": create_transform_matrix(peg_pos_np, peg_quat_np),
                "finger_l_w": create_transform_matrix(finger_l_pos_np, finger_l_quat_np),
                "finger_r_w": create_transform_matrix(finger_r_pos_np, finger_r_quat_np)
            }
            
            self._debug_visualize_transforms(
                local_l=self.tactile_points_left_local[0].cpu().numpy(),
                local_r=self.tactile_points_right_local[0].cpu().numpy(),
                world_l=tactile_points_left_w[0].cpu().numpy(),
                world_r=tactile_points_right_w[0].cpu().numpy(),
                peg_local=all_tactile_points_peg_local[0].cpu().numpy(),
                transforms=transforms,
                all_tactile_points_w = all_tactile_points_w
            )
        
        # --- 4. 计算SDF并生成触觉图像 (这部分逻辑不变) ---
        batch_size, num_points, _ = all_tactile_points_peg_local.shape
        points_np = all_tactile_points_peg_local.view(-1, 3).cpu().numpy()
        distances_np = self.peg_sdf(points_np)
        
        penetration_depth_np = -np.minimum(-distances_np, 0)
        tactile_image = torch.from_numpy(penetration_depth_np).to(self.device).view(batch_size, num_points)

        # import cv2
        # print(tactile_image)
        depth_image = (tactile_image.reshape(50, 100).cpu().numpy() * 25500).astype(np.uint8)
        depth_image = cv2.resize(depth_image, (300, 600))
        # cv2.imshow("depth_image", depth_image)
        # cv2.waitKey(1)
        # --- 5. 检查是否需要进行最终的触觉热力图可视化 ---
        if self.enable_tactile_visualization and self.visualization_counter % self.visualization_interval == 0:
            def create_transform_matrix(pos_np, quat_np_wxyz):
                # Isaac Lab [w, x, y, z] -> trimesh [w, x, y, z]
                matrix = trimesh.transformations.quaternion_matrix(quat_np_wxyz)
                matrix[:3, 3] = pos_np
                return matrix

            transforms = {
                "peg_w": create_transform_matrix(peg_pos_w[0].cpu().numpy(), peg_quat_w[0].cpu().numpy()),
                "finger_l_w": create_transform_matrix(left_finger_pos_w[0].cpu().numpy(), left_finger_quat_w[0].cpu().numpy()),
                "finger_r_w": create_transform_matrix(right_finger_pos_w[0].cpu().numpy(), right_finger_quat_w[0].cpu().numpy())
            }
            
            self._visualize_tactile_contact(
                all_tactile_points_w,
                tactile_image,
                all_tactile_points_peg_local,
                transforms
            )

        depth_image = self.env.scene.sensors["tactile_depth_camera"].data.output["distance_to_image_plane"]
        
        debug = True
        if debug:
            
            depth_image_np = (depth_image.reshape(320, 240).cpu().numpy()).astype(np.uint8)
            min_max_norm = (depth_image_np - np.min(depth_image_np)) / (np.max(depth_image_np) - np.min(depth_image_np) + 1e-6)
            cv2.imshow("depth_image", min_max_norm * 255)
            cv2.waitKey(1)

            pause = False
            if pause:
                import pdb; pdb.set_trace()

        return tactile_image



class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)
        self.tactile_system = TactileSensingSystem(self)

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
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        if self.cfg_task.name == "peg_insert":
            held_base_z_offset = 0.0
        elif self.cfg_task.name == "gear_mesh":
            gear_base_offset = self._get_target_gear_base_offset()
            held_base_x_offset = gear_base_offset[0]
            held_base_z_offset = gear_base_offset[2]
        elif self.cfg_task.name == "nut_thread":
            held_base_z_offset = self.cfg_task.fixed_asset_cfg.base_height
        else:
            raise NotImplementedError("Task not implemented")

        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        if self.cfg_task.name == "peg_insert":
            self.fixed_success_pos_local[:, 2] = 0.0
        elif self.cfg_task.name == "gear_mesh":
            gear_base_offset = self._get_target_gear_base_offset()
            self.fixed_success_pos_local[:, 0] = gear_base_offset[0]
            self.fixed_success_pos_local[:, 2] = gear_base_offset[2]
        elif self.cfg_task.name == "nut_thread":
            head_height = self.cfg_task.fixed_asset_cfg.base_height
            shank_length = self.cfg_task.fixed_asset_cfg.height
            thread_pitch = self.cfg_task.fixed_asset_cfg.thread_pitch
            self.fixed_success_pos_local[:, 2] = head_height + shank_length - thread_pitch * 1.5
        else:
            raise NotImplementedError("Task not implemented")

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

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
            compliant_contact_stiffness=100.0,
            compliant_contact_damping=0.0
            # 其他属性如 friction, restitution 会使用默认值
        )
        
        # 在场景中创建一个新的物理材质 Prim。路径可以自定义，"/World/Looks/" 是常用约定
        soft_material_path = "/World/Looks/SoftElastomerMaterial"
        spawn_rigid_body_material(prim_path=soft_material_path, cfg=soft_material_cfg)
        print(f"已创建自定义物理材质于: {soft_material_path}")

        # --- 步骤 2: 将新材质应用到每个环境的机器人手指上 ---
        
        self.sim.step() # 确保材质和机器人 Prim 都已加载

        self._tiled_camera = Camera(self.cfg.tactile_depth_camera)
        self.scene.sensors["tactile_depth_camera"] = self._tiled_camera

        return
        self.nominal_depth = self.scene.sensors["tactile_camera"].data.output["distance_to_image_plane"].clone()
        import pdb; pdb.set_trace()
        for i in range(self.scene.num_envs):
            # 定义左右两个手指的 碰撞体 Prim 的路径
            paths_to_modify = [
                f"/World/envs/env_{i}/Robot/elastomer_right/collisions",
                f"/World/envs/env_{i}/Robot/elastomer_left/collisions"
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
                
                print(f"成功将材质 '{soft_material_path}' 应用到: {path}")


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

        # Keypoint tensors.
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat,
                self.target_held_base_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
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
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ["prev_actions"]]
        obs_tensors = torch.cat(obs_tensors, dim=-1)
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ["prev_actions"]]
        state_tensors = torch.cat(state_tensors, dim=-1)
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)
        action =  action[:,0:6]
        # action[:,0] = 1.0
        # print("action", action)
        # self.actions = (
        #     self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        # )
        self.actions = action
        tactile_data = self.tactile_system.update()

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

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
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        xy_dist = torch.linalg.vector_norm(self.target_held_base_pos[:, 0:2] - self.held_base_pos[:, 0:2], dim=1)
        z_disp = self.held_base_pos[:, 2] - self.target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
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
            is_rotated = self.curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_buf = self._update_rew_buf(curr_successes)

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

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""
        rew_dict = {}

        # Keypoint rewards.
        def squashing_fn(x, a, b):
            return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        rew_dict["kp_baseline"] = squashing_fn(self.keypoint_dist, a0, b0)
        # a1, b1 = 25, 2
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        rew_dict["kp_coarse"] = squashing_fn(self.keypoint_dist, a1, b1)
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # a2, b2 = 300, 0
        rew_dict["kp_fine"] = squashing_fn(self.keypoint_dist, a2, b2)

        # Action penalties.
        rew_dict["action_penalty"] = torch.norm(self.actions, p=2)
        rew_dict["action_grad_penalty"] = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        rew_dict["curr_engaged"] = (
            self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False).clone().float()
        )
        rew_dict["curr_successes"] = curr_successes.clone().float()

        rew_buf = (
            rew_dict["kp_coarse"]
            + rew_dict["kp_baseline"]
            + rew_dict["kp_fine"]
            - rew_dict["action_penalty"] * self.cfg_task.action_penalty_scale
            - rew_dict["action_grad_penalty"] * self.cfg_task.action_grad_penalty_scale
            + rew_dict["curr_engaged"]
            + rew_dict["curr_successes"]
        )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _get_target_gear_base_offset(self):
        """Get offset of target gear from the gear base asset."""
        target_gear = self.cfg_task.target_gear
        if target_gear == "gear_large":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.large_gear_base_offset
        elif target_gear == "gear_medium":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
        elif target_gear == "gear_small":
            gear_base_offset = self.cfg_task.fixed_asset_cfg.small_gear_base_offset
        else:
            raise ValueError(f"{target_gear} not valid in this context!")
        return gear_base_offset

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

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
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
        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros_like(self.held_base_pos_local)
            gear_base_offset = self._get_target_gear_base_offset()
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = self.held_base_pos_local
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = self.identity_quat
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
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
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
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self._get_target_gear_base_offset()[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.hand_down_euler = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
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
            self.hand_down_euler[bad_envs, ...] = hand_down_euler
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (c) iterative IK Method
            self.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = above_fixed_pos[bad_envs, ...]
            self.ctrl_target_fingertip_midpoint_quat[bad_envs, ...] = hand_down_quat[bad_envs, :]

            pos_error, aa_error = self.set_pos_inverse_kinematics(env_ids=bad_envs)
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
            t2=torch.zeros_like(self.fingertip_midpoint_pos),
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
        self.held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            self.held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        self.held_asset_pos_noise = self.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=self.identity_quat,
            t2=self.held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
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
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        # Back out what actions should be for initial state.
        # Relative position to bolt tip.
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        pos_actions = self.fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
