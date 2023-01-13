import open3d as o3d
print("Environment Ready")
# ----------------------->
if __name__ == "__main__":
    # load point cloud data
    # simple geometries residing within test_set used to verify performance of region growing algorithm
    pcd = o3d.io.read_point_cloud("../test_set/down_sampled_density/cone.pcd")
    # more complex geometries residing within test_set/original_density used to verify performance of shape recognition algorithm
    pcd = o3d.io.read_point_cloud("../test_set/original_density/unknown_geometry_0.pcd")
    # computation of normal vectors
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    # visualization
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
