import torch
import numpy as np
import sys
import os
import cv2
import json
from scipy.ndimage import zoom

class Dataset10KParse:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.scene_json_file = os.path.join(self.root_dir, "_scene_meta_backup.json")
        
    def parse(self, select_ids, target_shape=None):
        if not os.path.exists(self.scene_json_file):
            print(f"Scene JSON file {self.scene_json_file} does not exist.")
            return None
        scene_info = json.load(open(self.scene_json_file, 'r'))
        scene_frames_info = scene_info["frames"]
        if select_ids is not None:
            scene_frames_info = [scene_frames_info[i] for i in select_ids]

        fx = scene_info["fl_x"]
        fy = scene_info["fl_y"]
        cx = scene_info["cx"]
        cy = scene_info["cy"]

        self.all_frames = []
        self.all_gt_c2w = []
        self.all_gt_w2c = []
        self.all_intrinsics = []
        self.all_depth = []
        for frame_info in scene_frames_info:
            # img
            img_file = frame_info["image"]
            img_path = os.path.join(self.root_dir, img_file)
            if not os.path.exists(img_path):
                print(f"Image file {img_path} does not exist.")
                continue
            frame = cv2.imread(img_path)        # 图像
            # c2w
            transform_matrix = frame_info["transform_matrix"]
            gt_c2w = np.array(transform_matrix, dtype=np.float32)
            gt_w2c = np.linalg.inv(gt_c2w)
 
            # intrinsic
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])      # 内参

            # depth
            depth_file = frame_info["mvsanywhere_depth"]
            depth_path = os.path.join(self.root_dir, depth_file)
            if not os.path.exists(depth_path):
                print(f"Depth file {depth_path} does not exist.")
                continue
            depth_map = self.read_exr_depth(depth_path, frame.shape[1], frame.shape[0])    # 深度图

            self.all_frames.append(frame)                                       # 1、图像
            self.all_gt_c2w.append(gt_c2w)  # (4, 4)                            # 2、camera2world坐标系的pose
            self.all_gt_w2c.append(gt_w2c)  # (4, 4)                            # 3、world2camera坐标系的pose
            self.all_intrinsics.append(K)                                       # 4、内参
            self.all_depth.append(depth_map)                                    # 5、深度图
        
        self.all_frames = np.array(self.all_frames)                             # (N, H, W, C)
        self.all_intrinsics = np.array(self.all_intrinsics).astype(np.float32)  # (N, 3, 3)
        self.all_gt_c2w = np.array(self.all_gt_c2w)          # (N, 4, 4)
        self.all_gt_w2c = np.array(self.all_gt_w2c)          # (N, 4, 4)
        self.all_depth = np.array(self.all_depth)            # (N, H, W)

        # 数据集的depth分辨率和frame分辨率不一样，frame的分辨率和depth对齐
        if target_shape != None:
            target_H, target_W = target_shape
            N, H, W, _ = self.all_frames.shape
            _, depth_H, depth_W = self.all_depth.shape
            zoom_factor_1 = (
                1,
                target_H / H,
                target_W / W,
                1
            )
            zoom_factor_2 = (
                1,
                target_H / depth_H,
                target_W / depth_W,
            )
            # 调整内参 + frames
            for K in self.all_intrinsics:
                K[0, 2] *= target_W / W
                K[1, 2] *= target_H / H
                K[0, 0] *= target_W / W
                K[1, 1] *= target_H / H
            self.all_frames = zoom(self.all_frames, zoom_factor_1, order=0)
            self.all_depth = zoom(self.all_depth, zoom_factor_2, order=0)

        dataset_infos = {
            "frames": self.all_frames,                    # (N, 704, 704, 3)
            "intrinsics": self.all_intrinsics,            # (N, 3, 3)
            "c2w": self.all_gt_c2w,                       # (N, 4, 4)
            "w2c": self.all_gt_w2c,                       # (N, 4, 4)
            "depth": self.all_depth                       # (N, 704, 704)
        }

        # 验证一下
        gt_pointcloud_in_cam = []
        for frame_idx in range(N):
            depth_map = self.all_depth[frame_idx]
            intrinsic = self.all_intrinsics[frame_idx]
            cam_coords = self.depth_to_cam_coords_points(depth_map, intrinsic)
            gt_pointcloud_in_cam.append(cam_coords)
        gt_pointcloud_in_cam = np.stack(gt_pointcloud_in_cam, axis=0)

        self.save_all_points_in_world(self.all_gt_c2w, 
                                     gt_pointcloud_in_cam, 
                                     self.all_frames / 255,
                                     "/root/gt.ply")

        return dataset_infos 
    
    def save_all_points_in_world(self, c2w_poses, points_in_cam_coord, points_color, file_path):
        N, H, W, _ = points_in_cam_coord.shape
        points_in_cam_coord_homo = np.concatenate([points_in_cam_coord, np.ones((N, H, W, 1))], axis=-1)

        points_in_world = np.einsum("nij, n...j->n...i", c2w_poses, points_in_cam_coord_homo)
        points_in_world = points_in_world[..., :3]

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_in_world.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(points_color.reshape(-1, 3))
        pcd.voxel_down_sample(0.3)
        o3d.io.write_point_cloud(file_path, pcd)

    def depth_to_cam_coords_points(self, depth_map, intrinsic):
        # 根据深度图 + 内参，恢复相机坐标系的点云
        H, W = depth_map.shape
        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x_cam = (u - cu) * depth_map / fu
        y_cam = (v - cv) * depth_map / fv
        z_cam = depth_map
        cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
        return cam_coords

    # 读取exr格式的深度图
    def read_exr_depth(self, file_path, width, height):
        import OpenEXR
        import Imath
        try:
            exr_file = OpenEXR.InputFile(file_path)
            header = exr_file.header()
            dw = header['dataWindow']
            w, h = (dw.max.x - dw.min.x + 1), (dw.max.y - dw.min.y + 1)
        
            for channel_name in ['Z', 'Y', 'R', 'G', 'B']:
                if channel_name in header['channels']:
                    depth_channel = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
                    depth_np = np.frombuffer(depth_channel, dtype=np.float32).reshape(h, w)
                    if depth_np.shape != (height, width):
                        if abs(depth_np.shape[1]/width - depth_np.shape[0]/height) < 0.01:
                            from skimage.transform import resize
                            depth_np = resize(depth_np, (height, width), order=0, preserve_range=True)
                    return depth_np
        except Exception as e:
            print(f"error: {e}")