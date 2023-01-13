import os
import cv2
import yaml

import pyrealsense2 as rs
import numpy as np
import open3d as o3d

from cv2 import aruco

class Camera:
    def __init__(self, cfg):

        self.cfg = cfg

        self.context = rs.context()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.spatial_filter = rs.spatial_filter()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        print(self.device_product_line + " is ready")
        self.device_name = device.get_info(rs.camera_info.name).replace(" ", "_")
        self.device_name = self.device_name + "_" + device.get_info(rs.camera_info.serial_number)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.color_intrinsic = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsic = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fx = self.color_intrinsic.fx
        self.fy = self.color_intrinsic.fy
        self.ppx = self.color_intrinsic.ppx
        self.ppy = self.color_intrinsic.ppy

        self.camera_mat = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]], dtype=np.float)
        self.dist_coeffs = np.zeros(4)
        self.colorizer = rs.colorizer(color_scheme = 2)

        self.saw_yaml = False
        self.saw_charuco = False
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_dict_ch = aruco.Dictionary_get(aruco.DICT_4X4_250)

    def stop(self):
        self.pipeline.stop(self.config)
        
    def stream(self, colored_depth = False):
        
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        ## filter depth frame
        # depth_frame = self.self.spatial_filter.process(depth_frame)

        colored_depth_frame = self.colorizer.colorize(depth_frame)

        self.color_image = np.asanyarray(color_frame.get_data())

        if colored_depth == True:
            self.depth_image = np.asanyarray(colored_depth_frame.get_data())
        else: 
            self.depth_image = np.asanyarray(depth_frame.get_data())

        return self.color_image, self.depth_image

    def generate(self, depth):
        self.pcd = o3d.geometry.PointCloud()
        ## return raw point cloud self.xyz from intensity-aligned depth map
        ## using color intrinsics and vectorized computation programing technique
        w                = np.shape(depth)[1]
        h                = np.shape(depth)[0]
        z                = depth * self.depth_scale  # raw distance
        u                = np.arange(0, w)
        v                = np.arange(0, h)
        mesh_u, mesh_v   = np.meshgrid(u, v)
        mesh_x           = (mesh_u - self.ppx) * z / self.fx
        mesh_y           = (mesh_v - self.ppy) * z / self.fy
        ## remove zeros and NaN values from x, y, z  this is valid regardless of if depth image is filtered or not
        if np.any(z == 0) or np.isnan(z).any():
            z = z[np.nonzero(z)]
            z = z[~ np.isnan(z)]
            mesh_x = mesh_x[np.nonzero(mesh_x)]
            mesh_x = mesh_x[~ np.isnan(mesh_x)]
            mesh_y = mesh_y[np.nonzero(mesh_y)]
            mesh_y = mesh_y[~ np.isnan(mesh_y)]
        ## raw point cloud in numpy format
        self.xyz         = np.zeros((np.size(mesh_x), 3))
        self.xyz[:, 0]   = np.reshape(mesh_x, -1)
        self.xyz[:, 1]   = np.reshape(mesh_y, -1)
        self.xyz[:, 2]   = np.reshape(z,      -1)
        ## raw point cloud in o3d format
        self.pcd.points  = o3d.utility.Vector3dVector(self.xyz)
        # self.pcd = self.pcd.voxel_down_sample(voxel_size = 0.008)
        self.xyz         = np.asarray(self.pcd.points)
        return self.xyz

    def drawWorkSpace(self):
        corner = np.reshape([-self.offset_from_corner, self.offset_from_corner, 0, 1], (4, 1))
        corner101_from_cam = np.dot(self.cam2marker, corner)[:3]
        pixel = (np.dot(self.camera_mat, corner101_from_cam)/corner101_from_cam[-1])[:2]
        pixel = pixel.astype(np.int64)
        pixel = np.reshape(pixel, (2,))
        pixel = tuple(pixel)
        cv2.circle(img=self.color_image, center=pixel, radius=4, color=(0, 0, 255), thickness=-1)

        width_end_point = np.reshape([-self.offset_from_corner, self.offset_from_corner-self.W, 0, 1], (4, 1))
        width_end_point_from_cam = np.dot(self.cam2marker, width_end_point)[:3]
        pixel_width = (np.dot(self.camera_mat, width_end_point_from_cam)/width_end_point_from_cam[-1])[:2]
        pixel_width = pixel_width.astype(np.int64)
        pixel_width = np.reshape(pixel_width, (2,))
        pixel_width = tuple(pixel_width)
        cv2.circle(img=self.color_image, center=pixel_width, radius=4, color=(0, 0, 255), thickness=-1)

        height_end_point = np.reshape([-self.offset_from_corner+self.L, self.offset_from_corner, 0, 1], (4, 1))
        height_end_point_from_cam = np.dot(self.cam2marker, height_end_point)[:3]
        pixel_length = (np.dot(self.camera_mat, height_end_point_from_cam)/height_end_point_from_cam[-1])[:2]
        pixel_length = pixel_length.astype(np.int64)
        pixel_length = np.reshape(pixel_length, (2,))
        pixel_length = tuple(pixel_length)
        cv2.circle(img=self.color_image, center=pixel_length, radius=4, color=(0, 0, 255), thickness=-1)

        vector2corner4th = np.reshape([-self.offset_from_corner+self.L, self.offset_from_corner-self.W, 0, 1], (4, 1))
        vector2corner4th_from_cam = np.dot(self.cam2marker, vector2corner4th)[:3]
        pixel_4th = (np.dot(self.camera_mat, vector2corner4th_from_cam)/vector2corner4th_from_cam[-1])[:2]
        pixel_4th = pixel_4th.astype(np.int64)
        pixel_4th = np.reshape(pixel_4th, (2,))
        pixel_4th = tuple(pixel_4th)
        cv2.circle(img=self.color_image, center=pixel_4th, radius=4, color=(0, 0, 255), thickness=-1)

        cv2.line(img=self.color_image, pt1=pixel, pt2=pixel_width, color=(0, 0, 255), thickness=2)
        cv2.line(img=self.color_image, pt1=pixel, pt2=pixel_length, color=(0, 0, 255), thickness=2)
        cv2.line(img=self.color_image, pt1=pixel_4th, pt2=pixel_width, color=(0, 0, 255), thickness=2)
        cv2.line(img=self.color_image, pt1=pixel_4th, pt2=pixel_length, color=(0, 0, 255), thickness=2)

    def detectCharuco(self):
        if self.saw_charuco != True:
            self.board = aruco.CharucoBoard_create(7, 5, 0.035, 0.025, self.aruco_dict_ch)
            self.params = aruco.DetectorParameters_create()
            self.saw_charuco = True

        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_points  =  aruco.detectMarkers(gray_img, self.aruco_dict_ch, parameters=self.params)
        if np.shape(corners)[0] <= 0:
            print("INFO: No Marker Detected")
        
        else:
            print("INFO: Marker Detected:",len(corners))
            aruco.refineDetectedMarkers(gray_img, self.board, corners, ids, rejected_points)
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray_img, self.board, self.camera_mat, self.dist_coeffs)
            if len(corners) > 10:
                _, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.camera_mat, self.dist_coeffs)
                R, _ = cv2.Rodrigues(rvec)
                tvec = np.reshape(tvec, (3, 1))
                cam2marker = np.concatenate((R, tvec), axis = 1)
                cam2marker = np.concatenate((cam2marker, np.array([[0, 0, 0, 1]])), axis = 0)
                aruco.drawAxis(self.color_image, self.camera_mat, self.dist_coeffs, rvec, tvec, 0.07)
                aruco.drawDetectedCornersCharuco(self.color_image, charuco_corners, charuco_ids, (255, 0, 0))

    def detectCropRegion(self):
        if self.saw_yaml != True:
            self.contents = 0
            ref_path = os.getcwd()
            with open(ref_path+"/bin.yml") as f:
                Bin = yaml.load(f, Loader=yaml.FullLoader)
                self.W = Bin["N432"][self.device_name]["Width"]
                self.L = Bin["N432"][self.device_name]["Length"]
                self.marker_size = Bin["N432"][self.device_name]["Marker_size"]
                self.offset_from_corner = self.marker_size/2
                if self.device_product_line == 'L500':
                    self.xp_off = Bin['L515']['xp_off']
                    self.xn_off = Bin['L515']['xn_off']
                    self.yp_off = Bin['L515']['yp_off']
                    self.yn_off = Bin['L515']['yn_off']
                elif self.device_product_line == 'D400':
                    self.xp_off = Bin['D400']['xp_off']
                    self.xn_off = Bin['D400']['xn_off']
                    self.yp_off = Bin['D400']['yp_off']
                    self.yn_off = Bin['D400']['yn_off']
                self.saw_yaml = True

       
        gray_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray_img, self.aruco_dict, parameters=parameters)
        self.color_image = aruco.drawDetectedMarkers(self.color_image, corners, ids)
        if np.shape(corners)[0] > 0:
            for i in range(np.shape(corners)[0]):
                if ids[i] == 101:
           
                    self.rvecs, self.tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], self.marker_size, cameraMatrix=self.camera_mat, distCoeffs=self.dist_coeffs)
                    self.contents += 1
                    print(self.contents, end="\r")
                  
                    self.color_image = cv2.drawFrameAxes(self.color_image, cameraMatrix=self.camera_mat, distCoeffs=self.dist_coeffs, rvec=self.rvecs, tvec=self.tvecs, length=0.050, thickness=2)

                    R, _ = cv2.Rodrigues(self.rvecs)
                    self.tvecs = np.reshape(self.tvecs, (3, 1))
                    self.cam2marker = np.concatenate((R, self.tvecs), axis = 1)
                    self.cam2marker = np.concatenate((self.cam2marker, np.array([[0, 0, 0, 1]])), axis = 0)

                    self.drawWorkSpace()

 
                    
    def cropPoints(self):

        self.detectCropRegion()

        R = self.cam2marker[:3, :3]
        self.tvecs = self.cam2marker[:3, 3]
        R_inv = np.transpose(R)
        t_inv = -1 * np.dot(R_inv, self.tvecs)
        t_inv = np.reshape(t_inv, (3, 1))
        H_inv = np.concatenate((R_inv, t_inv), axis = 1)
        H_inv = np.concatenate((H_inv, np.array([[0, 0, 0, 1]])), axis = 0)
        self.pcd.transform(H_inv)
        self.xyz = np.asarray(self.pcd.points)
        
        if self.device_product_line == 'L500':
            valid_idx = np.where(((self.xyz[:, 0] > -self.xn_off) & (self.xyz[:, 0] < -(-self.L + self.xp_off))) & ((self.xyz[:, 1] < self.yp_off) & (self.xyz[:, 1] > (-self.W + self.yn_off))) & ((self.xyz[:, 2] > 0.004) & (self.xyz[:,2] < 0.5)))[0]
        elif self.device_product_line == 'D400':
            valid_idx = np.where(((self.xyz[:, 0] > -self.xn_off) & (self.xyz[:, 0] < -(-self.L + self.xp_off))) & ((self.xyz[:, 1] < self.yp_off) & (self.xyz[:, 1] > (-self.W + self.yn_off))) &  ((self.xyz[:,2] < 0.5) & (self.xyz[:, 2] > 0.0037)))[0]
        self.pcd = self.select_by_index(self.pcd, valid_idx)
        # self.pcd = self.pcd.select_by_index(valid_idx)

        ## transform point cloud to original frame (camera frame)
        self.pcd.transform(self.cam2marker)
        self.xyz = np.asarray(self.pcd.points)
        return self.xyz

    @staticmethod
    def select_by_index(pcd, index=[]): 
        
        xyz = o3d.geometry.PointCloud()

        if pcd.has_points():
            points = np.asarray(pcd.points)
            xyz.points = o3d.utility.Vector3dVector(points[index])

        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            xyz.normals = o3d.utility.Vector3dVector(normals[index])

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            xyz.colors = o3d.utility.Vector3dVector(colors[index])

        return xyz

# if __name__ == '__main__':
#     ref_path = os.getcwd()

#     # with open(ref_path+"/config/suction_config.yml") as f:
#     #     cfg = yaml.load(f, Loader=yaml.FullLoader)
#     #     cfg['TF']['end2cam'] = np.reshape(cfg['TF']['end2cam'], (4, 4))

#     cam = Camera([])

#     pcd = o3d.geometry.PointCloud()
#     vis = o3d.visualization.Visualizer()
#     vis.create_window("Point Clouds", width=848, height=480)
#     added = True
#     rgb_img, depth_img = cam.stream(colored_depth=False)

#     xyz = cam.generate(depth_img)
#     # cam.detectCharuco()
#     xyz = cam.cropPoints()
#     pcd.points = o3d.utility.Vector3dVector(xyz)

#     o3d.io.write_point_cloud("/home/juyong/Desktop/d435_practice/offline_unknown_shape_recognition/test_set/original_density/test_3.pcd", pcd)

#     while 1:
#         rgb_img, depth_img = cam.stream(colored_depth=False)

#         xyz = cam.generate(depth_img)
#         # cam.detectCharuco()
#         xyz = cam.cropPoints()
#         pcd.points = o3d.utility.Vector3dVector(xyz)

#         ## visualize rgb and depth image
#         cv2.imshow("rgb", rgb_img)
#         cv2.imshow("depth", depth_img)
#         cv2.waitKey(1)

#         ## visualize point cloud caculated from the depth image
#         if added == True:
#             vis.add_geometry(pcd)
#             added = False
#         vis.update_geometry(pcd)
#         vis.poll_events()
#         vis.update_renderer()