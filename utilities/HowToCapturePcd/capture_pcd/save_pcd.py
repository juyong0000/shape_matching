
from camera import Camera
import open3d as o3d
import cv2
import os

if __name__ == "__main__":

    cam = Camera([])
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Clouds", width=848, height=480)
    added = True
    rgb_img, depth_img = cam.stream(colored_depth=False)

    xyz = cam.generate(depth_img)

    xyz = cam.cropPoints()
    pcd.points = o3d.utility.Vector3dVector(xyz)


    ref_path = os.getcwd()
    o3d.io.write_point_cloud(ref_path+"/pcd_data/test.pcd", pcd)

    while 1:
        rgb_img, depth_img = cam.stream(colored_depth=False)

        xyz = cam.generate(depth_img)
        # cam.detectCharuco()
        xyz = cam.cropPoints()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        ## visualize rgb and depth image
        cv2.imshow("rgb", rgb_img)
        cv2.imshow("depth", depth_img)
        cv2.waitKey(1)

        ## visualize point cloud caculated from the depth image
        if added == True:
            vis.add_geometry(pcd)
            added = False
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()