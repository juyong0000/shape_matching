import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import subprocess, signal
import copy
import sys
sys.path.append("../utilities")
from SO3 import SO3
from SE3 import SE3
sys.path.append("../camera")
from camera import IntelCamera
import sympy as sp
from sympy.utilities.autowrap import autowrap
import cv2
import time
import logging
from colorlog import ColoredFormatter
import os
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
print("Environment Ready")
# ------------------------------>
T_W_O = np.array([0.27770281, 0.10200414, 0.91577012, 5.46966638e-05,  7.09911726e-01, -7.04009753e-01, -1.98898238e-02])


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def draw_bounding_box(T, box_dim, color):
    R          = T[:3, :3]
    ref_pt     = T[:3, 3].reshape(1,3)
    array_pts  = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]) * box_dim
    lines      = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    array_pts               = (R.dot(array_pts.transpose())).transpose()
    bounding_box            = {}
    bounding_box["type"]    = "line_set"
    bounding_box["refined"] = {}
    points                  = array_pts.tolist()
    points                  = points + ref_pt
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color(color)
    bounding_box["refined"]["model"] = line_set
    return bounding_box

def draw_geometries(xyz, geometry):
    models       = []
    pcd          = o3d.geometry.PointCloud()
    pcd.points   = o3d.utility.Vector3dVector(xyz)

    for i in range(len(geometry)):
        if geometry[i]["type"] == "plane":
            length = geometry[i]["refined"]["length"]
            width  = geometry[i]["refined"]["width"]
            height = geometry[i]["refined"]["height"]
            T      = geometry[i]["refined"]["T"]
            mesh_model  = o3d.geometry.TriangleMesh.create_box(width= length, height= width, depth= height)
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07, origin=[0, 0, 0])
            mesh_model.compute_vertex_normals()
            mesh_model.paint_uniform_color([0.9, 0.1, 0.1])

            mesh_model.translate([-length/2, -width/2, -height/2])
            mesh_model.transform(T)
            world_frame.transform(T)
            #mesh_model = mesh_model + world_frame
            models.append(mesh_model)

        elif geometry[i]["type"] == "sphere":
            r = geometry[i]["refined"]["model"][3]
            T = geometry[i]["refined"]["T"]
            mesh_model = o3d.geometry.TriangleMesh.create_sphere(radius = abs(r))
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07, origin=[0, 0, 0])
            mesh_model.compute_vertex_normals()
            mesh_model.paint_uniform_color([0, 0, 1])

            mesh_model.transform(T)
            world_frame.transform(T)
            #mesh_model = mesh_model + world_frame
            models.append(mesh_model)
        elif geometry[i]["type"] == "cylinder":
            r      = geometry[i]["refined"]["model"][4]
            height = geometry[i]["refined"]["height"]
            T      = geometry[i]["refined"]["T"]

            mesh_model  = o3d.geometry.TriangleMesh.create_cylinder(radius = abs(r), height = height, resolution = 100)
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07, origin=[0, 0, 0])
            mesh_model.compute_vertex_normals()
            mesh_model.paint_uniform_color([0, 0.51, 0.14])
            # rotation matrix that align Z-axis to X-axis
            R = SO3(ypr = np.array([0, np.pi/2, 0])).R
            mesh_model.rotate(R, center=(0, 0, 0))
            world_frame.rotate(R, center=(0, 0, 0))
            # CAUTION: # the cylinder and the world frame would arbitrarily oriented in both direction
            # However, this is not perceptually important since the cylinder is symmetric geometry
            mesh_model.transform(T)
            world_frame.transform(T)
            #mesh_model = mesh_model + world_frame
            models.append(mesh_model)
        elif geometry[i]["type"] == "cone":
            radius_bottom = geometry[i]["refined"]["cone_radius_bottom"]
            height_bottom = geometry[i]["refined"]["cone_height_bottom"]
            radius_top    = geometry[i]["refined"]["cone_radius_top"]
            height_top    = geometry[i]["refined"]["cone_height_top"]
            height        = height_bottom - height_top
            axis          = geometry[i]["refined"]["A"]
            center        = geometry[i]["refined"]["cone_center"]
            T             = geometry[i]["refined"]["T"]
            phi           = geometry[i]["refined"]["model"][2][0]
            theta         = geometry[i]["refined"]["model"][1][0]
            alpha         = geometry[i]["refined"]["model"][3][0]

            # ----------------------------> draw cone axis
            axis_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius = abs(0.001), height = height * 2, resolution = 100)
            axis_mesh.compute_vertex_normals()
            axis_mesh.paint_uniform_color([0, 0, 0])
            T_axis = np.eye(4)
            T_axis[:3, 3]  = center
            T_axis[:3, :3] = T[:3, :3]
            tip = o3d.geometry.TriangleMesh.create_cone(radius = 0.005, height = 0.03, resolution=30, split=1)
            tip.compute_vertex_normals()
            tip.paint_uniform_color([0, 0, 0])
            T_tip            = np.eye(4)
            T_tip[:3, 3]     = center + 2 * height * axis / 2
            T_tip[:3, :3]    = T[:3, :3]
            # ----------------------------> draw cone top
            top_circle = o3d.geometry.TriangleMesh.create_cylinder(radius = abs(radius_top), height = 0.001, resolution = 100)
            top_circle.compute_vertex_normals()
            top_circle.paint_uniform_color([1, 0.549, 0])
            T_top = np.eye(4)
            T_top[:3, :3] = T[:3, :3]
            center_top = center - height * axis / 2
            T_top[:3, 3]  = center_top
            # ----------------------------> draw cone bottom
            bottom_circle = o3d.geometry.TriangleMesh.create_cylinder(radius = abs(radius_bottom), height = 0.001, resolution = 100)
            bottom_circle.compute_vertex_normals()
            bottom_circle.paint_uniform_color([1, 0.549, 0])
            T_bottom = np.eye(4)
            T_bottom[:3, :3] = T[:3, :3]
            center_bottom = center + height * axis / 2
            T_bottom[:3, 3]  = center_bottom

            theta  =  np.linspace(0, np.pi*2, 100)
            center_top = center_top.reshape(3,1)
            center_bottom = center_bottom.reshape(3,1)
            points = []
            lines  = []
            for i in range(len(theta)):
                circular_pt_top    = center_top + T[:3,:3].dot(np.array([[0], [radius_top*np.cos(theta[i])], [radius_top*np.sin(theta[i])]]))
                circular_pt_bottom = center_bottom + T[:3,:3].dot(np.array([[0], [radius_bottom*np.cos(theta[i])], [radius_bottom*np.sin(theta[i])]]))
                points.append([circular_pt_top[0,0], circular_pt_top[1,0], circular_pt_top[2,0]])
                points.append([circular_pt_bottom[0,0], circular_pt_bottom[1,0], circular_pt_bottom[2,0]])
                lines.append([i*2, i*2 + 1])

            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),lines=o3d.utility.Vector2iVector(lines),)
            line_set.paint_uniform_color([1, 0.549, 0])

            # ---------> Transformation
            # rotation matrix that align Z-axis to X-axis
            R = SO3(ypr = np.array([0, np.pi/2, 0])).R
            top_circle.rotate(R, center=(0, 0, 0))
            bottom_circle.rotate(R, center=(0, 0, 0))
            axis_mesh.rotate(R, center=(0, 0, 0))
            tip.rotate(R, center=(0, 0, 0))

            top_circle.transform(T_top)
            bottom_circle.transform(T_bottom)
            axis_mesh.transform(T_axis)
            tip.transform(T_tip)

            mesh_model = top_circle + bottom_circle + axis_mesh + tip
            mesh_model = top_circle + bottom_circle
            models.append(mesh_model)
            models.append(line_set)
        elif geometry[i]["type"] == "line_set":
            models.append(geometry[i]["refined"]["model"])

    for i in range(len(models)):
        models[i] = models[i].transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    models.append(pcd)
    # o3d.visualization.draw_geometries(models, point_show_normal=True)

# ************** distance-based related jacobian sphere ************** #
def jacobian_sphere():
    # Notation:
    # M_i = (m_ix, m_iy, m_iz) : range point M_i
    # C   = (c_x, c_y, c_z)    : sphere centre
    # r   : sphere radius
    m_ix, m_iy, m_iz = sp.symbols('m_ix m_iy m_iz')
    c_x, c_y, c_z    = sp.symbols('c_x c_y c_z')
    r                = sp.symbols('r')
    # distance of element M_i from estimated surface
    e_sph     = sp.sqrt((m_ix - c_x)**2 + (m_iy - c_y)**2 + (m_iz - c_z)**2) - sp.sign(r)*r
    # jacobian of distance function (i.e. lost function)
    J_cx      = sp.diff(e_sph, c_x)
    J_cy      = sp.diff(e_sph, c_y)
    J_cz      = sp.diff(e_sph, c_z)
    J_r       = sp.diff(e_sph, r)
    jacobian_sphere  = sp.Matrix([J_cx, J_cy, J_cz, J_r])
    jacobian_sphere  = jacobian_sphere.subs(sp.diff(sp.sign(r), r), 0)
    jacobian_sphere  = autowrap(jacobian_sphere, backend="f2py", args = [m_ix, m_iy, m_iz, c_x, c_y, c_z, r])
    return jacobian_sphere

# ********************** cylinder-related jacobian ******************* #
def jacobian_cylinder():
    # Notation:
    # M_i = (m_ix, m_iy, m_iz) : range point M_i
    # d                        : distance from camera origin O to closest point P on cylinder's axis
    # theta                    : polar coordinate of OP
    # phi                      : polar coordinate of OP
    # alpha                    : angle between cylinder's axis A and N_theta
    # r                        : cylinder radius r
    # range point M_i
    m_ix, m_iy, m_iz         = sp.symbols('m_ix, m_iy, m_iz')
    M_i                      = sp.Matrix([m_ix, m_iy, m_iz])
    # cylinder parameters
    d, theta, phi, alpha, r  = sp.symbols('d, theta, phi, alpha, r')
    # closest point P on cylinder axis to camera origin
    N          = sp.Matrix([sp.cos(phi)*sp.sin(theta), sp.sin(phi)*sp.sin(theta), sp.cos(theta)]) # column vector
    P          = d*N                                                                              # column vector
    # cylinder axis
    N_theta    = sp.Matrix([sp.cos(phi)*sp.cos(theta), sp.sin(phi)*sp.cos(theta), -sp.sin(theta)])
    N_phi      = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
    A          = N_theta*sp.cos(alpha) + N_phi*sp.sin(alpha)
    # distance from range element M_i to estimated cylinder surface
    D_i        = P + A*(M_i - P).dot(A) - M_i
    e_cyl      = sp.sqrt(D_i[0]**2 + D_i[1]**2 + D_i[2]**2) - sp.sign(r)*r
    # jacobian of loss function
    J_d        = sp.diff(e_cyl, d)
    J_theta    = sp.diff(e_cyl, theta)
    J_phi      = sp.diff(e_cyl, phi)
    J_alpha    = sp.diff(e_cyl, alpha)
    J_r        = sp.diff(e_cyl, r)
    J_cylinder = sp.Matrix([J_d, J_theta, J_phi, J_alpha, J_r])
    J_cylinder = J_cylinder.subs(sp.diff(sp.sign(r), r), 0)
    J_cylinder = autowrap(J_cylinder, backend="f2py", args = [m_ix, m_iy, m_iz, d, theta, phi, alpha, r])
    return J_cylinder

# **********************  cone related jacobian ********************** #
def jacobian_cone():
    # Notation
    # M_i = (m_ix, m_iy, m_iz) : range point M_i
    # d                        : distance from camera origin O to closest point P on cone axis
    # theta                    : polar coordinate of OP
    # phi                      : polar coordinate of OP
    # alpha                    : angle between cone axis A and N_theta
    # r                        : cone radius r at point P (might negative or positive)
    # range point M_i
    m_ix, m_iy, m_iz         = sp.symbols('m_ix, m_iy, m_iz')
    M_i                      = sp.Matrix([m_ix, m_iy, m_iz])
    # cone parameters
    d, theta, phi, alpha, r, delta  = sp.symbols('d, theta, phi, alpha, r, delta')
    # closest point P on cone axis to camera origin
    N          = sp.Matrix([sp.cos(phi)*sp.sin(theta), sp.sin(phi)*sp.sin(theta), sp.cos(theta)]) # column vector
    P          = d*N
    # cone axis
    N_theta    = sp.Matrix([sp.cos(phi)*sp.cos(theta), sp.sin(phi)*sp.cos(theta), -sp.sin(theta)])
    N_phi      = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
    A          = N_theta*sp.cos(alpha) + N_phi*sp.sin(alpha)
    # cone apex
    C          = P + r/sp.tan(delta)*A
    # distance from range element M_i to estimated cone surface
    D_i        = C + A*(M_i - C).dot(A) - M_i
    r_i        = (M_i - C).dot(A)*sp.tan(delta)
    e_cone     = sp.sqrt(D_i[0]**2 + D_i[1]**2 + D_i[2]**2) - r_i

    # jacobian of loss function
    J_d        = sp.diff(e_cone, d)
    J_theta    = sp.diff(e_cone, theta)
    J_phi      = sp.diff(e_cone, phi)
    J_alpha    = sp.diff(e_cone, alpha)
    J_r        = sp.diff(e_cone, r)
    J_delta    = sp.diff(e_cone, delta)
    J_cone     = sp.Matrix([J_d, J_theta, J_phi, J_alpha, J_r, J_delta])
    J_cone     = autowrap(J_cone, backend = "f2py", args = [m_ix, m_iy, m_iz, d, theta, phi, alpha, r, delta])
    return J_cone

#  *******************  cone-axis related jacobian ******************* #
def jacobian_cone_axis():
    # estimation of cone axis
    # notation
    # A   : cone's axis, expressed in polar coordinates
    # phi, theta               : polar coordinates of cone axis
    # N_i : normal vector N_i

    # normal vector N_i
    N_ix, N_iy, N_iz         = sp.symbols('N_ix, N_iy, N_iz')
    N_i                      = sp.Matrix([N_ix, N_iy, N_iz])
    # cone axis A
    phi, theta               = sp.symbols('phi, theta')
    A                        = sp.Matrix([sp.cos(phi)*sp.sin(theta), sp.sin(phi)*sp.sin(theta), sp.cos(theta)]) # column vector
    # deviation of arccos(A, N_i) and xi
    e_cone_axis              = sp.acos(N_i.dot(A))
    # jacobian of loss function associated with cone's axis estimation process
    J_cone_axis_phi          = sp.diff(e_cone_axis, phi)
    J_cone_axis_theta        = sp.diff(e_cone_axis, theta)
    J_cone_axis              = sp.Matrix([J_cone_axis_phi, J_cone_axis_theta])
    J_cone_axis              = autowrap(J_cone_axis, backend = "f2py", args = [N_ix, N_iy, N_iz, phi, theta])
    return J_cone_axis

def compute_jacobian_sphere(lambdified_jacobian_sphere, input_vector):
    result  = lambdified_jacobian_sphere(*input_vector).reshape((4,)).astype('float32')
    return result

def compute_jacobian_cylinder(lambdified_jacobian_cylinder, input_vector):
    result  = lambdified_jacobian_cylinder(*input_vector).reshape((5,)).astype('float32')
    return result

def compute_jacobian_cone(lambdified_jacobian_cone, input_vector):
    result  = lambdified_jacobian_cone(*input_vector).reshape((6,)).astype('float32')
    return result

def compute_jacobian_cone_axis(lambdified_jacobian_cone_axis, input_vector):
    result = lambdified_jacobian_cone_axis(*input_vector).reshape((2,)).astype('float32')
    return result

def fitting_plane(cluster):
    cluster_xyz       = cluster[0]
    no_pts            = cluster_xyz.shape[0]
    cluster_centroid  = np.mean(cluster_xyz, axis=0)
    # -----------------> principal component analysis
    cov      = (1/no_pts)*((cluster_xyz - cluster_centroid).T).dot(cluster_xyz - cluster_centroid)
    # eigenvalues eigenvector analysis
    w, v                 = np.linalg.eig(cov)
    idx                  = w.argsort()[::-1]
    w                    = w[idx]
    v                    = v[:,idx]
    # conform eigenvector with right-hand side coordinate system
    v[:,0] = np.cross(v[:, 1], v[:, 2])
    normals              = v[:,2]
    # define the cluster's central frame (DONE)
    if normals.dot(cluster_centroid) > 0: # orient the normals vector towards the camera origin
        normals = -normals
        v[:, 2] = -v[:, 2]
        v[:, 1] = -v[:, 1]
    # cluster's coordinates with respect to new central frame
    centered_cluster = cluster_xyz - cluster_centroid
    R_inv  = SO3(R = v).inverse().R
    reoriented_cluster = R_inv.dot(centered_cluster.transpose())
    reoriented_cluster = reoriented_cluster.transpose()
    length             = np.max(reoriented_cluster[:, 0]) - np.min(reoriented_cluster[:, 0])
    width              = np.max(reoriented_cluster[:, 1]) - np.min(reoriented_cluster[:, 1])
    height             = np.max(reoriented_cluster[:, 2]) - np.min(reoriented_cluster[:, 2])
    # ----------------> estimation of plane model that best fits to given cluster
    a = normals[0]
    b = normals[1]
    c = normals[2]
    d = -np.dot(normals, cluster_centroid)
    model = [a, b, c, d]
    # ----------------> the mean square regression error is the smallest eigenvalue
    residual = w[2]
    # posture between cluster's central frame and camera frame
    T      = SE3(position = cluster_centroid, orientation = SO3(R = v)).T
    # ----------------> summarize
    result = [model, residual, T, length, width, height]
    return result

def fitting_sphere(cluster_info, init_geometry_meta_info, optim_config_params):
    # parsing
    internal_states        = []
    cluster_xyz            = cluster_info[0]
    cluster_normals        = cluster_info[1]
    no_pts                 = cluster_xyz.shape[0]
    # parsing
    damping_factor         = optim_config_params["damping_factor"]
    c                      = optim_config_params["c"]
    threshold              = optim_config_params["threshold"]
    # parsing
    initial_model          = init_geometry_meta_info["model"]
    initial_error          = init_geometry_meta_info["error"]
    initial_quadratic_lost = init_geometry_meta_info["quadratic_lost"]
    # ------------------------> initial estimation of jacobian
    input_vector = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(initial_model.transpose()), axis = 1)
    jacobian     = []
    for m in range(input_vector.shape[0]):
        jacobian.append(compute_jacobian_sphere(lambdified_jacobian_sphere, input_vector[m, :]))
    jacobian = np.array(jacobian)
    # -------------------> update initial values for refinement process
    quadratic_lost = initial_quadratic_lost
    error          = initial_error
    solution       = initial_model
    initial_state  = [solution[0,0],solution[1,0], solution[2,0], solution[3,0], quadratic_lost]
    internal_states.append(initial_state)

    for x in range(100):

        # -------> update incremental change in estimation of sphere's centre and radius
        H        = np.dot(jacobian.transpose(), jacobian) # Hesian
        t1       = H + damping_factor * np.eye(H.shape[0]) # adding damping factor
        try:
            t2       = np.linalg.inv(t1)
        except:
            break
        t3       = np.dot(t2, jacobian.transpose())
        delta    = -t3.dot(error)
        if np.linalg.norm(delta) <= threshold:
            break

        # -------> update sphere centre and radius
        solution_temp       = solution + delta
        # -------> update error vector and quadratic lost and jacobian

        error_temp, quadratic_lost_temp, rmse_temp, _, _, _ \
                              = compute_distances_to_sphere(solution_temp, cluster_xyz)

        internal_state      = [solution_temp[0,0],solution_temp[1,0], solution_temp[2,0], solution_temp[3,0], quadratic_lost_temp]
        internal_states.append(internal_state)
        # Since we do not know if the current incremental change in solution resulting in reduced quadratic loss
        if quadratic_lost_temp < quadratic_lost:
            # error decreased, reducing damping factor
            damping_factor = damping_factor / c
            # update the solution
            solution       = solution_temp
            # update error vector
            error          = error_temp
            # update the quadratic lost
            quadratic_lost = quadratic_lost_temp
            # update jacobian
            input_vector   = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(solution.transpose()), axis = 1)
            jacobian       = []
            for m in range(input_vector.shape[0]):
                jacobian.append(compute_jacobian_sphere(lambdified_jacobian_sphere, input_vector[m, :]))
            jacobian = np.array(jacobian)
        else:
            # error increased: discard and raising damping factor
            damping_factor = c * damping_factor

    # ---------> transformation that aligns sphere's frame and camera's frame
    T = SE3(position = solution[:3, 0]).T
    # ---------> summarize
    result   = [[solution, quadratic_lost, T], internal_states]
    return result

def fitting_cylinder(cluster_info, init_geometry_meta_info, optim_config_params):
    # parsing
    internal_states = []
    cluster_xyz     = cluster_info[0]
    cluster_normals = cluster_info[1]
    no_pts          = cluster_xyz.shape[0]
    # computation of covariance feature

    # parsing
    damping_factor  = optim_config_params["damping_factor"]
    c               = optim_config_params["c"]
    threshold       = optim_config_params["threshold"]
    # parsing
    initial_model          = init_geometry_meta_info["model"]
    initial_error          = init_geometry_meta_info["error"]
    initial_quadratic_lost = init_geometry_meta_info["quadratic_lost"]
    # ------------------------> initial estimation of jacobian
    input_vector = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(initial_model.transpose()), axis = 1)
    jacobian     = []
    for m in range(input_vector.shape[0]):
        jacobian.append(compute_jacobian_cylinder(lambdified_jacobian_cylinder, input_vector[m,:]))
    jacobian = np.array(jacobian)

    # ---------------------> update the initial value for refinement process
    error          = initial_error
    quadratic_lost = initial_quadratic_lost
    solution       = initial_model
    initial_state  = [solution[0,0],solution[1,0], solution[2,0], solution[3,0], solution[4,0], quadratic_lost]
    internal_states.append(initial_state)

    for x in range(100):
        # -----------> update the incremental change in all cylinder's parameters
        H = np.dot(jacobian.transpose(), jacobian)     # Hessian
        t1 = H + damping_factor * np.eye(H.shape[0])   # adding damping factor
        try:
            t2 = np.linalg.inv(t1)
        except:
            break
        t3 = np.dot(t2, jacobian.transpose())
        delta = -t3.dot(error)
        if np.linalg.norm(delta) <= threshold:
            break
        # -----------> update cylinder's parameters, error vector, quadratic lost and jacobian
        solution_temp        = solution + delta

        error_temp, quadratic_lost_temp, rmse, _, _, _ \
                             = compute_distances_to_cylinder(solution_temp, cluster_xyz)

        internal_state      = [solution_temp[0,0],solution_temp[1,0], solution_temp[2,0], solution_temp[3,0], solution_temp[4,0], quadratic_lost_temp]
        internal_states.append(internal_state)

        if quadratic_lost_temp < quadratic_lost:
            # error decreased, reducing damping factor
            damping_factor   = damping_factor / c
            # update the solution
            solution         = solution_temp
            # update error vector
            error            = error_temp
            # update the quadratic lost
            quadratic_lost   = quadratic_lost_temp
            # update the jacobian
            input_vector     = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(solution.transpose()), axis = 1)
            jacobian     = []
            for m in range(input_vector.shape[0]):
                jacobian.append(compute_jacobian_cylinder(lambdified_jacobian_cylinder, input_vector[m,:]))
            jacobian = np.array(jacobian)
        else:
            # error increased: discard and raising damping factor
            damping_factor   = c * damping_factor

    d              = solution[0, 0]
    theta          = solution[1, 0]
    phi            = solution[2, 0]
    alpha          = solution[3, 0]
    r              = solution[4, 0]
    N              = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    N_theta        = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)])
    N_phi          = np.array([-np.sin(phi), np.cos(phi), 0])
    P              = d*N
    A              = N_theta * np.cos(alpha) + N_phi * np.sin(alpha)
    # Projecting points onto the cylinder's axis
    temp                 = np.sum((cluster_xyz - P)*A, axis = 1).reshape((no_pts, 1))
    projection           = P + temp.dot(A.reshape((1, 3)))
    min_index            = np.argmin(projection[:, 0])
    max_index            = np.argmax(projection[:, 0])
    height               = np.linalg.norm(projection[min_index, :] - projection[max_index, :])
    directional_vector   = (projection[max_index, :] - projection[min_index, :])/np.linalg.norm(projection[max_index, :] - projection[min_index, :])
    center               = projection[min_index, :]  + directional_vector * height/2
    # --------> estimation of relative pose between cylinder's frame and camera's frame
    T = SE3(custom = {"t": center, "composed_rotational_components" : {"orders": ["z", "y", "z"], "angles" : [phi, theta, alpha]}}).T
    # --------> summarize
    result   = [[solution, quadratic_lost, T, height, A], internal_states]
    return result

def fitting_cone(cluster_info, init_geometry_meta_info, optim_config_params):
    # parsing
    internal_states        = []
    cluster_xyz            = cluster_info[0]
    cluster_normals        = cluster_info[1]
    no_pts                 = cluster_xyz.shape[0]
    # parsing
    damping_factor         = optim_config_params["damping_factor"]
    c                      = optim_config_params["c"]
    threshold              = optim_config_params["threshold"]
    # parsing
    initial_model           = init_geometry_meta_info["model"]
    initial_error           = init_geometry_meta_info["error"]
    initial_quadratic_lost  = init_geometry_meta_info["quadratic_lost"]
    # ------------------------> initial estimation of jacobian
    input_vector            = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(initial_model.transpose()), axis = 1)
    jacobian                = []
    for m in range(input_vector.shape[0]):
        jacobian.append(compute_jacobian_cone(lambdified_jacobian_cone, input_vector[m, :]))
    jacobian = np.array(jacobian)
    # ------------------------> update the initial value for refinement process
    quadratic_lost         = initial_quadratic_lost
    error                  = initial_error
    solution               = initial_model
    initial_state          = [solution[0,0], solution[1,0], solution[2,0], solution[3,0], solution[4,0], solution[5,0], quadratic_lost]
    internal_states.append(initial_state)
    for x in range(100):
        # --------> update the incremental change in all cone's parameters
        H     = np.dot(jacobian.transpose(), jacobian)     # Hessian
        t1    = H + damping_factor * np.eye(H.shape[0])    # adding damping factor
        try:
            t2    = np.linalg.inv(t1)
        except:
            break
        t3        = np.dot(t2, jacobian.transpose())
        delta_inc = -t3.dot(error)
        if np.linalg.norm(delta_inc) <= threshold:
            break
        # --------> update cone's parameters
        solution_temp = solution + delta_inc
        error_temp, quadratic_lost_temp, rmse, _, _, _  \
                               = compute_distances_to_cone(solution_temp, cluster_xyz)

        internal_state         = [solution_temp[0,0], solution_temp[1,0], solution_temp[2,0], solution_temp[3,0], solution_temp[4,0], solution_temp[5,0], quadratic_lost_temp]
        internal_states.append(internal_state)
        if quadratic_lost_temp < quadratic_lost:
            # error decreased reducing damping fself.logger.warning(f" found from {i}th cluster {ratios[0]*100}% valley")actor
            damping_factor = damping_factor / c
            # update the solution
            solution       = solution_temp
            # update error vector
            error          = error_temp
            # update the quadratic lost
            quadratic_lost = quadratic_lost_temp
            # update the jacobian
            input_vector   = np.append(cluster_xyz, np.ones((no_pts, 1)).dot(solution.transpose()), axis = 1)
            jacobian                = []
            for m in range(input_vector.shape[0]):
                jacobian.append(compute_jacobian_cone(lambdified_jacobian_cone, input_vector[m, :]))
            jacobian = np.array(jacobian)
        else:
            # error increased: discard and raising damping factor
            damping_factor = c * damping_factor

    d              = solution[0, 0]
    theta          = solution[1, 0]
    phi            = solution[2, 0]
    alpha          = solution[3, 0]
    r              = solution[4, 0]
    delta          = solution[5, 0]
    N              = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    N_theta        = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)])
    N_phi          = np.array([-np.sin(phi), np.cos(phi), 0])
    P              = d*N
    A              = N_theta * np.cos(alpha) + N_phi * np.sin(alpha)
    C              = P + (r/np.tan(delta))*A
    # --------------> Compute cone height and cone raidius for visualization
    # Projecting points onto the cone's axis
    temp1                = np.sum((cluster_xyz - C)*A, axis = 1).reshape((no_pts, 1))
    projection           = C + temp1.dot(A.reshape((1, 3)))
    dists                = np.linalg.norm(projection - C, axis = 1)
    cone_height_bottom   = dists[np.argmax(dists)]
    cone_height_top      = dists[np.argmin(dists)]
    cone_height          = cone_height_bottom - cone_height_top
    cone_center          = projection[np.argmin(dists)] + cone_height * A/2
    cone_radius_bottom   = np.tan(delta) * cone_height_bottom
    cone_radius_top      = np.tan(delta) * cone_height_top
    # --------> estimation of relative pose between cone's frame and camera's frame
    T              = SE3(custom = {"t": C, "composed_rotational_components" : {"orders": ["z", "y", "z"], "angles" : [phi, theta, alpha]}}).T
    # --------> summarize
    result         = [[solution, quadratic_lost, T, A, cone_height_bottom, cone_radius_bottom, cone_height_top, cone_radius_top, cone_center], internal_states]
    return result

def compute_distances_to_plane(model, cluster_xyz, clipping_distance = 0.003):
    # Find projection of known point to a given plane
    # https://mathinsight.org/distance_point_plane#:~:text=The%20shortest%20distance%20from%20a,as%20a%20gray%20line%20segment.
    no_pts             = cluster_xyz.shape[0]
    a                  = model[0]
    b                  = model[1]
    c                  = model[2]
    d                  = model[3]
    signed_distances   = ((a * cluster_xyz[:,0] + b*cluster_xyz[:,1] + c*cluster_xyz[:,2] + d) / np.sqrt(a*a + b*b + c*c)).reshape((no_pts, 1))
    lost               = np.sum(signed_distances * signed_distances) / no_pts
    rmse               = np.sqrt(lost)
    distances          = np.absolute(signed_distances)
    inliers            = np.argwhere(distances <= clipping_distance)[:, 0]
    adherence          = inliers.shape[0] / no_pts
    return signed_distances, lost, rmse, adherence, inliers

def compute_distances_to_sphere(model, cluster_xyz, clipping_distance = 0.003):
    no_pts           = cluster_xyz.shape[0]
    signed_distances = (np.linalg.norm(cluster_xyz - model[:3, 0], axis=1) - abs(model[3, 0])).reshape((no_pts, 1))
    proj_normals     = cluster_xyz - model[:3, 0]
    proj_normals     = proj_normals / np.linalg.norm(proj_normals, axis=-1)[:, np.newaxis]
    lost             = np.sum(signed_distances * signed_distances) / no_pts
    rmse             = np.sqrt(lost)
    distances        = np.absolute(signed_distances)
    inliers          = np.argwhere(distances <= clipping_distance)[:, 0]
    adherence        = inliers.shape[0] / no_pts
    return signed_distances, lost, rmse, adherence, inliers, proj_normals

def compute_distances_to_cylinder(model, cluster_xyz, clipping_distance = 0.003):
    no_pts               = cluster_xyz.shape[0]
    d                    = model[0, 0]
    theta                = model[1, 0]
    phi                  = model[2, 0]
    alpha                = model[3, 0]
    r                    = model[4, 0]
    N                    = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    N_theta              = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)])
    N_phi                = np.array([-np.sin(phi), np.cos(phi), 0])
    P                    = d*N
    A                    = N_theta * np.cos(alpha) + N_phi * np.sin(alpha)
    # signed distances from xyz_s to model's surface
    temp                 = np.sum((cluster_xyz - P)*A, axis = 1).reshape((no_pts, 1))
    D                    = P + temp.dot(A.reshape((1, 3))) - cluster_xyz
    proj_normals         = cluster_xyz - (P + temp.dot(A.reshape((1, 3))))
    proj_normals         = proj_normals / np.linalg.norm(proj_normals, axis=-1)[:, np.newaxis]
    signed_distances     = (np.linalg.norm(D, axis = 1) - abs(r)).reshape((no_pts, 1))
    lost                 = np.sum(signed_distances*signed_distances) / no_pts
    rmse                 = np.sqrt(lost)
    distances            = np.absolute(signed_distances)
    inliers              = np.argwhere(distances <= clipping_distance)[:, 0]
    adherence            = inliers.shape[0] / no_pts
    return signed_distances, lost, rmse, adherence, inliers, proj_normals

def compute_distances_to_cone(model, cluster_xyz, clipping_distance = 0.003):
    no_pts        = cluster_xyz.shape[0]
    d             = model[0, 0]
    theta         = model[1, 0]
    phi           = model[2, 0]
    alpha         = model[3, 0]
    r             = model[4, 0]
    delta         = model[5, 0]
    N             = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    N_theta       = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)])
    N_phi         = np.array([-np.sin(phi), np.cos(phi), 0])
    P             = d*N
    A             = N_theta * np.cos(alpha) + N_phi * np.sin(alpha)
    C             = P + (r/np.tan(delta))*A
    # signed distances from xyz_s to model's surface
    temp                   = np.sum((cluster_xyz - C) * A, axis = 1).reshape((no_pts, 1))
    D                      = C + temp.dot(A.reshape((1, 3))) - cluster_xyz
    r                      = np.sum((cluster_xyz - C) * A, axis = 1)*np.tan(delta)
    # projection of xyz on cone's axis and corresponding directional vectors
    proj_on_axis      = C + temp.dot(A.reshape((1, 3)))
    direction_vectors = cluster_xyz - proj_on_axis
    direction_vectors = direction_vectors / np.linalg.norm(direction_vectors, axis=-1)[:, np.newaxis]
    # projection of xyz on cone's surface
    proj_on_surface    = proj_on_axis + (direction_vectors.transpose() * r).transpose()
    # corresponding side vectors
    cone_side_vectors  = C - proj_on_surface
    cone_side_vectors  = cone_side_vectors / np.linalg.norm(cone_side_vectors, axis=-1)[:, np.newaxis]
    rotational_vectors = np.cross(cone_side_vectors, direction_vectors)
    proj_normals       = np.cross(rotational_vectors, cone_side_vectors)


    signed_distances       = (np.linalg.norm(D, axis = 1) - r).reshape((no_pts, 1))
    lost                   = np.sum(signed_distances * signed_distances) / no_pts
    rmse                   = np.sqrt(lost)
    distances              = np.absolute(signed_distances)
    inliers                = np.argwhere(distances <= clipping_distance)[:, 0]
    adherence              = inliers.shape[0] / no_pts
    return signed_distances, lost, rmse, adherence, inliers, proj_normals

def compute_adherence(geom_meta_info):
    type        = geom_meta_info[0]
    model       = geom_meta_info[1]
    cluster_xyz = geom_meta_info[2]
    _, _, _, adherence, _, _ = compute_distances_to_cylinder(model, cluster_xyz, clipping_distance = 0.003)
    return adherence

def compute_geometry(meta_info):
    start_time                    = time.time()
    # parsing cluster info
    cluster_xyz                   = meta_info[0]
    cluster_normals               = meta_info[1]
    cluster_info                  = [cluster_xyz, cluster_normals]
    # parsing optimizer configuration parameters
    optim_config_params           = meta_info[2]
    no_pts                        = cluster_xyz.shape[0]

    # parsing optimizer config params
    # --------------------------------------------------------------->
    damping_factor_cyl            = optim_config_params["cylinder"]["damping_factor"]
    c_cyl                         = optim_config_params["cylinder"]["c"]
    threshold_cyl                 = optim_config_params["cylinder"]["threshold"]

    # --------------------------------------------------------------->
    rmses                         = {}
    geometry_meta_info            = {}
    geometry_meta_info["init"]    = {}
    geometry_meta_info["refined"] = {}
    # ----------> Coarse geometry info
    cluster_centroid  =  np.mean(cluster_xyz, axis=0)
    cov               = (1/no_pts)*((cluster_xyz - cluster_centroid).T).dot(cluster_xyz - cluster_centroid)
    w, v              =  np.linalg.eig(cov)
    idx               =  w.argsort()[::-1] # sort in descending order
    w                 =  w[idx]            # sorted eigenvalues
    v                 =  v[:,idx]          # sorted eigenvectors
    # compute covariance features
    P = (w[1] - w[2]) / w[0] # planarity
    S = w[2] / w[0]          # sphericity

    if P / S  > 20:
        # -----------------------------> fitting plane
        
        # conform eigenvector with right-hand side coordinate system
        v[:,2] = np.cross(v[:, 0],v[:, 1])
        normals              = v[:,2] #X axis   

            
        # cluster's coordinates with respect to new central frame
        centered_cluster = cluster_xyz - cluster_centroid
        R_inv  = SO3(R = v).inverse().R
        reoriented_cluster = R_inv.dot(centered_cluster.transpose())
        reoriented_cluster = reoriented_cluster.transpose()

        a            = normals[0]
        b            = normals[1]
        c            = normals[2]
        d            = -np.dot(normals, cluster_centroid)

        residual     = w[2]

        length       = np.max(reoriented_cluster[:, 0]) - np.min(reoriented_cluster[:, 0]) #long side
        width        = np.max(reoriented_cluster[:, 1]) - np.min(reoriented_cluster[:, 1]) #short side
        # height       = np.max(reoriented_cluster[:, 2]) - np.min(reoriented_cluster[:, 2]) #normal
        height = T_W_O[2] - cluster_centroid[2]
        
        new_v = copy.deepcopy(v)

        xyz_list =  [length, width, height]

        W0 = copy.deepcopy(new_v[:,0])
        W1 = copy.deepcopy(new_v[:,1])
        W2 = copy.deepcopy(new_v[:,2])

        if height == min(xyz_list):
            new_v[:,0] =  W2
            new_v[:,1] =  W1
            new_v[:,2] =  -W0

        elif height == max(xyz_list):
            new_v[:,0] =  W1
            new_v[:,1] =  W0
            new_v[:,2] =  -W2    

        else:
            new_v[:,0] =  W1
            new_v[:,1] =  W2
            new_v[:,2] =  W0                   



        model        = [a, b, c, d]
        # posture between cluster's central frame and camera frame
        T            = SE3(position = cluster_centroid, orientation = SO3(R = new_v)).T
        geometry_meta_info["type"]                      = "plane"
        geometry_meta_info["refined"]["model"]          =  model
        geometry_meta_info["refined"]["length"]         =  length
        geometry_meta_info["refined"]["width"]          =  width
        geometry_meta_info["refined"]["height"]         =  height
        geometry_meta_info["refined"]["T"]              =  T
        geometry_meta_info["refined"]["quadratic_lost"] =  residual

    else:
        # -----------------------------> fitting sphere
        t1                    = np.sum(cluster_xyz * cluster_normals)
        t2                    = np.sum(np.sum(cluster_xyz, axis=0) * np.sum(cluster_normals, axis=0))
        t3                    = np.sum(cluster_normals * cluster_normals)
        t4                    = np.sum(np.sum(cluster_normals, axis=0) * np.sum(cluster_normals, axis=0))
        r                     = -(no_pts*t1 - t2)/(no_pts*t3 - t4)
        C                     = np.sum(cluster_xyz + r*cluster_normals, axis = 0) / no_pts
        sphere_model          = np.array([C[0], C[1], C[2], r]).reshape(4, 1)

        sphere_error, sphere_quadratic_lost, rmse, _, _, _ \
                                = compute_distances_to_sphere(sphere_model, cluster_xyz)

        rmses["sphere"]       = rmse

        # -----------------------------> fitting cylinder
        # central normals of cluster
        N_i                            = np.mean(cluster_normals, axis = 0)
        N_i                            = N_i / np.linalg.norm(N_i)
        Z                              = np.array([0, 0, -1])
        # transformation that aligns N_i with Z-axis
        theta                          = np.arccos(np.dot(N_i, Z))
        A                              = np.cross(N_i, Z) # rotational axis
        theta_r                        = np.array([theta, A[0], A[1], A[2]])
        R                              = SO3(axis_angle = theta_r).R
        R_inv                          = SO3(axis_angle = theta_r).inverse().R
        reoriented_cluster_normals     = np.dot(R, cluster_normals.transpose())
        reoriented_cluster_normals     = reoriented_cluster_normals.transpose()
        projection_XY                  = reoriented_cluster_normals[:, :2]
        # ---------> initial estimation of cylinder axis using PCA analysis on projection_XY
        k                              = projection_XY.shape[0]
        cov                            = (1/k)*(projection_XY.T).dot(projection_XY)
        w, v                           = np.linalg.eig(cov) # eigenvalues eigenvector analysis
        idx                            = w.argsort()[::-1]
        w                              = w[idx]
        v                              = v[:,idx]
        A0_projection_XY               = np.array([v[0,1], v[1,1], 0])
        A0                             = np.dot(R_inv, A0_projection_XY)
        if A0[2] > 0:
            Z = np.array([0, 0, 1])
        else:
            Z = np.array([0, 0, -1])
        # --------------------> initial estimation of cylinder radius and center
        # transformation that aligns A0 with the Z-axis
        theta                           = np.arccos(np.dot(A0, Z))
        A                               = np.cross(A0, Z) #rotational axis
        theta_r                         = np.array([theta, A[0], A[1], A[2]])
        R                               = SO3(axis_angle = theta_r).R
        R_inv                           = SO3(axis_angle = theta_r).inverse().R
        reoriented_cluster_normals      = np.dot(R, cluster_normals.transpose())
        reoriented_cluster_normals      = reoriented_cluster_normals.transpose()
        reoriented_cluster_normals[:,2] = 0
        reoriented_cluster_xyz          = np.dot(R, cluster_xyz.transpose())
        reoriented_cluster_xyz          = reoriented_cluster_xyz.transpose()
        reoriented_cluster_xyz[:,2]     = 0
        # initial estimation of radius r
        no_pts                          = cluster_xyz.shape[0]
        t1 = np.sum(reoriented_cluster_xyz * reoriented_cluster_normals)
        t2 = np.sum(np.sum(reoriented_cluster_xyz, axis=0) * np.sum(reoriented_cluster_normals, axis=0))
        t3 = np.sum(reoriented_cluster_normals * reoriented_cluster_normals)
        t4 = np.sum(np.sum(reoriented_cluster_normals, axis=0) * np.sum(reoriented_cluster_normals, axis=0))
        r0 = -(no_pts*t1 - t2)/(no_pts*t3 - t4)
        # initial estimation of centre of the cylinder
        C0_projection = np.sum(reoriented_cluster_xyz + r0*reoriented_cluster_normals, axis = 0) / no_pts
        C0            = np.dot(R_inv, C0_projection)
        # ---------------------> initial estimation of closest point P on A0 nearest to the origin
        # Ref: https://math.stackexchange.com/questions/3158880/get-rectangular-coordinates-of-a-3d-point-with-the-polar-coordinates
        r0            = r0                                                    # initial estimation
        P0            = C0 - A0*(np.sum(C0 * A0))                             # initial estimation
        d0            = np.linalg.norm(P0)                                    # initial estimation
        N0            = P0 / d0
        # Ref: https://stackoverflow.com/questions/35749246/python-atan-or-atan2-what-should-i-use
        # use of arctan2 instead of arctan
        theta_0        = np.arctan2(np.sqrt(N0[0]*N0[0] + N0[1]*N0[1]), N0[2]) # initial estimation
        phi_0          = np.arctan2(N0[1], N0[0])                              # initial estimation
        N0_theta       = np.array([np.cos(phi_0)*np.cos(theta_0), np.sin(phi_0)*np.cos(theta_0), -np.sin(theta_0)])
        N0_phi         = np.array([-np.sin(phi_0), np.cos(phi_0), 0])
        cos_alpha_0    = A0[2]/N0_theta[2]
        sin_alpha_0    = (A0[0] - N0_theta[0]*A0[2]/N0_theta[2])/N0_phi[0]
        alpha_0        = np.arctan2(sin_alpha_0, cos_alpha_0)                  # initial estimation
        cyl_init_model = np.array([d0, theta_0, phi_0, alpha_0, r0]).reshape(5, 1)

        # initial estimation of residual
        cyl_error, cyl_quadratic_lost, rmse, _, _, _ \
                                = compute_distances_to_cylinder(cyl_init_model, cluster_xyz)
        rmses["cylinder"]     = rmse

        # -----------------------------> Summarize # ------------------------------------>
        rmses_unsorted       = np.array([rmses["sphere"], rmses["cylinder"], rmses["cone"]])
        # reference: Least Squares Orthogonal Distance Fitting of Curves and Surfaces in Space
        # Preliminary selection of proper model feature
        # first condition  : a better model feature has smaller performance index rmse
        # second condition : a simple model feature is preferred to a complex one
        # 0: sphere, 1: cylinder, 2: cone
        rmse_idx = rmses_unsorted.argsort()        # sort in ascending order

        if rmse_idx[0] == 0:
            geometry_meta_info["type"]                   = "sphere"
            geometry_meta_info["init"]["model"]          = sphere_model
            geometry_meta_info["init"]["error"]          = sphere_error
            geometry_meta_info["init"]["quadratic_lost"] = sphere_quadratic_lost
            result = fitting_sphere(cluster_info, geometry_meta_info["init"], optim_config_params["sphere"])
            geometry_meta_info["refined"]["model"]          = result[0][0]
            geometry_meta_info["refined"]["quadratic_lost"] = result[0][1]
            geometry_meta_info["refined"]["T"]              = result[0][2]



        elif (rmse_idx[0] == 2 and rmse_idx[1] == 1 and (rmses_unsorted[1] / rmses_unsorted[2] <= 1.07)) or (rmse_idx[0] == 1):
            # compacity check is required
            geometry_meta_info["type"]                       = "cylinder"
            geometry_meta_info["init"]["model"]              = cyl_init_model
            geometry_meta_info["init"]["error"]              = cyl_error
            geometry_meta_info["init"]["quadratic_lost"]     = cyl_quadratic_lost
            result = fitting_cylinder(cluster_info, geometry_meta_info["init"], optim_config_params["cylinder"])
            geometry_meta_info["refined"]["model"]           = result[0][0]
            geometry_meta_info["refined"]["quadratic_lost"]  = result[0][1]
            geometry_meta_info["refined"]["T"]               = result[0][2]
            geometry_meta_info["refined"]["height"]          = result[0][3]
            geometry_meta_info["refined"]["A"]               = result[0][4]

        else: 
            print("No Primtiive shapes drawing bounding box")



    # # -----------------------------> fitting cylinder
    # # central normals of cluster
    # N_i                            = np.mean(cluster_normals, axis = 0)
    # N_i                            = N_i / np.linalg.norm(N_i)
    # Z                              = np.array([0, 0, -1])
    # # transformation that aligns N_i with Z-axis
    # theta                          = np.arccos(np.dot(N_i, Z))
    # A                              = np.cross(N_i, Z) # rotational axis
    # theta_r                        = np.array([theta, A[0], A[1], A[2]])
    # R                              = SO3(axis_angle = theta_r).R
    # R_inv                          = SO3(axis_angle = theta_r).inverse().R
    # reoriented_cluster_normals     = np.dot(R, cluster_normals.transpose())
    # reoriented_cluster_normals     = reoriented_cluster_normals.transpose()
    # projection_XY                  = reoriented_cluster_normals[:, :2]
    # # ---------> initial estimation of cylinder axis using PCA analysis on projection_XY
    # k                              = projection_XY.shape[0]
    # cov                            = (1/k)*(projection_XY.T).dot(projection_XY)
    # w, v                           = np.linalg.eig(cov) # eigenvalues eigenvector analysis
    # idx                            = w.argsort()[::-1]
    # w                              = w[idx]
    # v                              = v[:,idx]
    # A0_projection_XY               = np.array([v[0,1], v[1,1], 0])
    # A0                             = np.dot(R_inv, A0_projection_XY)
    # if A0[2] > 0:
    #     Z = np.array([0, 0, 1])
    # else:
    #     Z = np.array([0, 0, -1])
    # # --------------------> initial estimation of cylinder radius and center
    # # transformation that aligns A0 with the Z-axis
    # theta                           = np.arccos(np.dot(A0, Z))
    # A                               = np.cross(A0, Z) #rotational axis
    # theta_r                         = np.array([theta, A[0], A[1], A[2]])
    # R                               = SO3(axis_angle = theta_r).R
    # R_inv                           = SO3(axis_angle = theta_r).inverse().R
    # reoriented_cluster_normals      = np.dot(R, cluster_normals.transpose())
    # reoriented_cluster_normals      = reoriented_cluster_normals.transpose()
    # reoriented_cluster_normals[:,2] = 0
    # reoriented_cluster_xyz          = np.dot(R, cluster_xyz.transpose())
    # reoriented_cluster_xyz          = reoriented_cluster_xyz.transpose()
    # reoriented_cluster_xyz[:,2]     = 0
    # # initial estimation of radius r
    # no_pts                          = cluster_xyz.shape[0]
    # t1 = np.sum(reoriented_cluster_xyz * reoriented_cluster_normals)
    # t2 = np.sum(np.sum(reoriented_cluster_xyz, axis=0) * np.sum(reoriented_cluster_normals, axis=0))
    # t3 = np.sum(reoriented_cluster_normals * reoriented_cluster_normals)
    # t4 = np.sum(np.sum(reoriented_cluster_normals, axis=0) * np.sum(reoriented_cluster_normals, axis=0))
    # r0 = -(no_pts*t1 - t2)/(no_pts*t3 - t4)
    # # initial estimation of centre of the cylinder
    # C0_projection = np.sum(reoriented_cluster_xyz + r0*reoriented_cluster_normals, axis = 0) / no_pts
    # C0            = np.dot(R_inv, C0_projection)
    # # ---------------------> initial estimation of closest point P on A0 nearest to the origin
    # # Ref: https://math.stackexchange.com/questions/3158880/get-rectangular-coordinates-of-a-3d-point-with-the-polar-coordinates
    # r0            = r0                                                    # initial estimation
    # P0            = C0 - A0*(np.sum(C0 * A0))                             # initial estimation
    # d0            = np.linalg.norm(P0)                                    # initial estimation
    # N0            = P0 / d0
    # # Ref: https://stackoverflow.com/questions/35749246/python-atan-or-atan2-what-should-i-use
    # # use of arctan2 instead of arctan
    # theta_0        = np.arctan2(np.sqrt(N0[0]*N0[0] + N0[1]*N0[1]), N0[2]) # initial estimation
    # phi_0          = np.arctan2(N0[1], N0[0])                              # initial estimation
    # N0_theta       = np.array([np.cos(phi_0)*np.cos(theta_0), np.sin(phi_0)*np.cos(theta_0), -np.sin(theta_0)])
    # N0_phi         = np.array([-np.sin(phi_0), np.cos(phi_0), 0])
    # cos_alpha_0    = A0[2]/N0_theta[2]
    # sin_alpha_0    = (A0[0] - N0_theta[0]*A0[2]/N0_theta[2])/N0_phi[0]
    # alpha_0        = np.arctan2(sin_alpha_0, cos_alpha_0)                  # initial estimation
    # cyl_init_model = np.array([d0, theta_0, phi_0, alpha_0, r0]).reshape(5, 1)

    # # initial estimation of residual
    # cyl_error, cyl_quadratic_lost, rmse, _, _, _ \
    #                       = compute_distances_to_cylinder(cyl_init_model, cluster_xyz)
    # rmses["cylinder"]     = rmse

    # # compacity check is required
    # geometry_meta_info["type"]                       = "cylinder"
    # geometry_meta_info["init"]["model"]              = cyl_init_model
    # geometry_meta_info["init"]["error"]              = cyl_error
    # geometry_meta_info["init"]["quadratic_lost"]     = cyl_quadratic_lost
    # result = fitting_cylinder(cluster_info, geometry_meta_info["init"], optim_config_params["cylinder"])
    # geometry_meta_info["refined"]["model"]           = result[0][0]
    # geometry_meta_info["refined"]["quadratic_lost"]  = result[0][1]
    # geometry_meta_info["refined"]["T"]               = result[0][2]
    # geometry_meta_info["refined"]["height"]          = result[0][3]
    # geometry_meta_info["refined"]["A"]               = result[0][4]

    # Computing adherence of spatial measurement data to estimated geometry
    type      = geometry_meta_info["type"]
    model     = geometry_meta_info["refined"]["model"]
    geometry_info = (type, model, cluster_xyz)
    adherence = compute_adherence(geometry_info)
    geometry_meta_info["refined"]["adherence"] = adherence
    # Computing time given analytical-based geometry Initializer
    computing_time = time.time() - start_time
    geometry_meta_info["computing_time"] = computing_time
    return geometry_meta_info

def computeLargestConnectedGrid(bitmap):
    nb_components, output, stats, centroid = cv2.connectedComponentsWithStats(bitmap, connectivity=4)
    sizes     = stats[:, -1]
    max_label = 1
    max_size  = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size  = sizes[i]
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 1
    return img2

def cylinder_domain_volume_growing(voxel_size, T, model, cluster_xyz, cluster_normals, safety_margin, outlier_threshold_factor, angle_threshold):
    # ---------------------> widening the domain volume
    s_distances, _, _, _, inliers, proj_normals  = compute_distances_to_cylinder(model, cluster_xyz, clipping_distance = safety_margin)
    # first condition: within clipping margin
    # first condition is a loose condition, size of inliers is always different from zero
    inline_normals       = cluster_normals[inliers]
    inline_proj_normals  = proj_normals[inliers]
    deviate_angle        = np.arccos(abs(np.sum(inline_normals * inline_proj_normals, axis = 1)))
    inliers              = inliers[np.argwhere(deviate_angle <= angle_threshold)[:, 0]]
    if len(inliers) != 0:
        inline_xyz       = cluster_xyz[inliers]
        inline_xyz       = np.hstack((inline_xyz, np.ones((inline_xyz.shape[0], 1))))
        # transforming inline_xyz from camera's coordinate to the object's coordinate
        inline_xyz       = np.linalg.inv(T).dot(inline_xyz.transpose())
        inline_xyz       = inline_xyz.transpose()[:, :3]
        no_vox           = np.ceil((np.max(inline_xyz[:, :2], axis=0) - np.min(inline_xyz[:, :2], axis = 0))/voxel_size)
        bitmap           = np.zeros((int(no_vox[1]), int(no_vox[0]))).astype("uint8")
        voxel_key        = ((inline_xyz[:, :2] - np.min(inline_xyz[:, :2], axis = 0)) // voxel_size)
        non_empty_voxel_keys, inverse, no_pts_per_voxel \
                         = np.unique(voxel_key.astype(int), axis = 0, return_inverse = True, return_counts = True)
        idx_pts_vox_sorted  =  np.argsort(inverse)
        for i in range(non_empty_voxel_keys.shape[0]):
            bitmap[non_empty_voxel_keys[i, 1], non_empty_voxel_keys[i, 0]] = 1
        # computeLargestConnectedGrid
        img2 = computeLargestConnectedGrid(bitmap)
        last_seen = 0
        idxes     = []
        for i in range(non_empty_voxel_keys.shape[0]):
            if img2[non_empty_voxel_keys[i, 1], non_empty_voxel_keys[i, 0]] != 0:
                idx = idx_pts_vox_sorted[last_seen:last_seen+no_pts_per_voxel[i]].tolist()
                idxes.append(idx)
            last_seen += no_pts_per_voxel[i]
        idxes = list(np.concatenate(idxes).flat)
        idxes = np.array(idxes)
        inliers = inliers[idxes]
        # since the third condition is a loose condition, the returned number of inliers
        # are always larger than zero, proceeding to the final condition
        inline_xyz           = cluster_xyz[inliers]
        inline_distances     = np.absolute(s_distances[inliers])
        # ---------------------> contracting domain volume
        _, _, rmse, _, _, _  = compute_distances_to_cylinder(model, inline_xyz)
        idx                  = np.argwhere(inline_distances <= outlier_threshold_factor * rmse)[:, 0]
        # inliers
        inliers              = inliers[idx]
    else:
        # if the second condition is not passed, the return number of inliers is none
        inliers = np.zeros((0,0))
    return inliers

def iterative_region_growing(meta_info):
    cluster_xyz                 = meta_info[0]
    cluster_normals             = meta_info[1]
    geometry                    = meta_info[2]
    optim_config_params         = meta_info[3]
    voxel_size                  = optim_config_params["voxel_size"]
    domain_growing_factor       = 1.25
    outlier_threshold_factor    = 2
    angle_threshold             = 20 * np.pi / 180
    refined_geometry            = {}
    refined_geometry["refined"] = {}

    model                            =  geometry["refined"]["model"]
    T                                =  geometry["refined"]["T"]
    r                                =  abs(model[4, 0])
    optim_config_params_cylinder     =  optim_config_params["cylinder"]
    init_geometry_meta_info          =  {}

    safety_margin                    =  r / 2
    rmse                             =  0.003 # mm initial set
    temp                             =  0     # store of number of inliers
    while rmse >= 0.001:
        inliers = cylinder_domain_volume_growing(voxel_size, T, model, cluster_xyz, cluster_normals, safety_margin, outlier_threshold_factor, angle_threshold)
        # if the number of inlier does not change in compared to the previous step, terminate the loop
        if inliers.shape[0] == temp or inliers.shape[0] <= 300:
            break
        inline_xyz     = cluster_xyz[inliers]
        inline_normals = cluster_normals[inliers]
        error, lost, _, _, _, _  = compute_distances_to_cylinder(model, inline_xyz)

        init_geometry_meta_info["error"]          = error
        init_geometry_meta_info["quadratic_lost"] = lost
        init_geometry_meta_info["model"]          = model
        cluster_info                              = [inline_xyz, inline_normals]

        meta_info     = fitting_cylinder(cluster_info, init_geometry_meta_info, optim_config_params_cylinder)
        model         = meta_info[0][0]
        T             = meta_info[0][2]
        rmse          = np.sqrt(meta_info[0][1])
        safety_margin = domain_growing_factor * outlier_threshold_factor * rmse
        # store number of inlier of the current iteration
        temp          = inliers.shape[0]
    # in case iterative region growing algorithm successfully refines geometry with associted set of inliers
    # and with rmse <= 0.0007, return the refined_geometry, otherwise, algorithm fails to refine geometry
    print(rmse, "rmse cylinder")
    if rmse <= 0.001:
        refined_geometry["type"]                      = "cylinder"
        refined_geometry["refined"]["model"]          = meta_info[0][0]
        refined_geometry["refined"]["quadratic_lost"] = meta_info[0][1]
        refined_geometry["refined"]["T"]              = meta_info[0][2]
        refined_geometry["refined"]["height"]         = meta_info[0][3]
        refined_geometry["refined"]["A"]              = meta_info[0][4]
        refined_geometry["refined"]["inliers"]        = inliers
    else:
        refined_geometry["type"] = "undefined"
        refined_geometry["refined"]["adherence"] = 0
        print("can not refine initial cylinder")

    print("finish iterative region growing")
    # if initial geometry is refined, computing adherence
    if refined_geometry["type"] != "undefined":
        type          = refined_geometry["type"]
        model         = refined_geometry["refined"]["model"]
        geometry_info = (type, model, cluster_xyz)
        adherence     = compute_adherence(geometry_info)
        refined_geometry["refined"]["adherence"] = adherence
    return refined_geometry

def compute_unknown_geometry(unknown_geometry_info):
    pool                = Pool(mp.cpu_count(), init_worker)
    xyz                 = unknown_geometry_info[0]
    normals             = unknown_geometry_info[1]
    optim_config_params = unknown_geometry_info[2]
    voxel_size          = optim_config_params["voxel_size"]
    scales              = optim_config_params["scales"]
    N                   = len(scales)
    i                   = 0
    geometry            = []
    original_xyz        = xyz
    color               = [0, 0, 0]
    while i < N:
        no_pts          = xyz.shape[0]
        print(f"number of points {no_pts} at {i} level of decomposition")
        if no_pts < 300:
            break
        kdtree          = KDTree(xyz)
        box_dim_xy      = np.max(xyz[:, :2], axis = 0) - np.min(xyz[:, :2], axis = 0)
        voxel_size_in_z = np.max(xyz[:,  2], axis = 0) - np.min(xyz[:, 2] , axis = 0)
        voxel_size      = np.max(box_dim_xy) / scales[i]
        radius          = voxel_size * 1.25
        box_dim         = [voxel_size, voxel_size, voxel_size_in_z]
        voxel_key_xy    = ((xyz[:, :2] - np.min(xyz[:, :2], axis = 0)) // voxel_size)
        voxel_key       = np.hstack((voxel_key_xy, np.zeros((no_pts, 1))))
        non_empty_voxel_keys, inverse, no_pts_per_voxel = np.unique(voxel_key.astype(int), axis = 0, return_inverse = True, return_counts = True)
        idx_pts_vox_sorted     =  np.argsort(inverse)
        avg_no_pts_per_voxel   =  no_pts / len(non_empty_voxel_keys)
        last_seen              = 0
        grid_candidates_center = []
        inner_rings            = []
        outter_rings           = []
        for idx, vox in enumerate(non_empty_voxel_keys):
            if no_pts_per_voxel[idx] >= 200:
                cluster_xyz     = xyz[idx_pts_vox_sorted[last_seen:last_seen+no_pts_per_voxel[idx]]]
                cluster_normals = normals[idx_pts_vox_sorted[last_seen:last_seen+no_pts_per_voxel[idx]]]
                inner_rings.append((cluster_xyz, cluster_normals, optim_config_params))
                grid_candidate_center = cluster_xyz[np.linalg.norm(cluster_xyz - np.mean(cluster_xyz, axis = 0), axis = 1).argmin()].reshape(1, 3)
                ind = kdtree.query_radius(grid_candidate_center, r = radius)
                outter_rings.append(xyz[ind[0]])
                ## uncomment to visualize bounding boxes
                # ref_pt   = non_empty_voxel_keys[idx] * box_dim + np.min(xyz, axis = 0)
                # T        = np.eye(4)
                # T[:3, 3] = ref_pt
                # bounding_box = draw_bounding_box(T, box_dim, color)
                # geometry.append(bounding_box)
            last_seen += no_pts_per_voxel[idx]
        i += 1
        if len(inner_rings) != 0:
            inner_ring_geometry = pool.map(compute_geometry, inner_rings)
            geom_meta_info = []
            for j in range(len(inner_ring_geometry)):
                type  = inner_ring_geometry[j]["type"]
                model = inner_ring_geometry[j]["refined"]["model"]
                geom_meta_info.append((type, model, outter_rings[j]))

            outter_rings_adherence = pool.map(compute_adherence, geom_meta_info)
            outter_rings_adherence = np.array(outter_rings_adherence)
            idx_max = np.argmax(outter_rings_adherence)
            if outter_rings_adherence[idx_max] >= 0.78:
                print(f"found: {inner_ring_geometry[idx_max]['type']}, adherence: {outter_rings_adherence[idx_max]} at {i - 1} level of decomposition")
                print(inner_ring_geometry[idx_max]["refined"], "initial saliency")
                saliency_meta_info = [xyz, normals, inner_ring_geometry[idx_max], optim_config_params]
                salient_geometry = iterative_region_growing(saliency_meta_info)
                if salient_geometry["type"] != "undefined":
                    inliers  = salient_geometry["refined"]["inliers"]
                    geometry.append(salient_geometry)
                    xyz     =  np.delete(xyz, inliers, 0)
                    normals =  np.delete(normals, inliers, 0)
                    i = 0
        print("-" * 55)
    return geometry

def shape_recognition(pool, threadpool, logger, pcd, clusters, clusters_ids, optim_config_params):
    xyz             = np.asarray(pcd.points)
    normals         = np.asarray(pcd.normals)
    cluster_info    = []
    voxel_size      = optim_config_params["voxel_size"]
    safety_margin   = 0.003
    outlier_threshold_factor = 2
    angle_threshold = 20 * np.pi / 180
    # coarse geometry computation
    logger.warning("start preliminary segmentation")
    for i in range(len(cluster_ids)):
        cluster_xyz       =  xyz[clusters[i]]
        cluster_normals   =  normals[clusters[i]]
        cluster_info.append((cluster_xyz, cluster_normals, optim_config_params))
    geometry  = pool.map(compute_geometry, cluster_info)
    logger.warning("complete preliminary segmentation")
    # compute the overall adherence
    for i in range(len(geometry)):
        if geometry[i]["refined"]["adherence"] >= 0.7:
            model           = geometry[i]["refined"]["model"]
            T               = geometry[i]["refined"]["T"]
            cluster_xyz     = cluster_info[i][0]
            cluster_normals = cluster_info[i][1]
            inliers = cylinder_domain_volume_growing(voxel_size, T, model, cluster_xyz, cluster_normals, safety_margin, outlier_threshold_factor, angle_threshold)
            geometry[i]["refined"]["no_inliers"] = inliers.shape[0]

    logger.warning("start intermediate segmentation")
    geometry_info = []
    for i in range(len(cluster_ids)):
        geometry_info.append((xyz[clusters[i]], normals[clusters[i]], geometry[i], optim_config_params))
    geometry = pool.map(iterative_region_growing, geometry_info)
    logger.warning("complete intermediate segmentation")

    # classify geometry according to its adherence
    adherence = []
    print(len(geometry))
    for i in range(len(geometry)):
        adherence.append(geometry[i]["refined"]["adherence"])
    adherence = np.array(adherence)
    unknown_geometry_idx = sorted(np.where(adherence < 0.7)[0], reverse = True)
    known_geometry_idx   = np.where(adherence >= 0.7)[0]
    logger.warning(f"found {len(known_geometry_idx)} known geometries and {len(unknown_geometry_idx)} unknown geometries")
    if len(unknown_geometry_idx) != 0:
        unknown_geometry_info = []
        for idx in unknown_geometry_idx:
            geometry.pop(idx)
            unknown_geometry_info.append(cluster_info[idx])
        results = threadpool.map(compute_unknown_geometry, unknown_geometry_info)
        for i in range(len(results)):
            if len(results[i]) != 0:
                geometry.extend(results[i])
    logger.warning("complete ultimate segmentation")
    draw_geometries(xyz, geometry)

    return geometry
# ------------------------------------------------------------------------------------------->
# ------------------------------------------------------------------------------------------->
# ------------------------------------------------------------------------------------------->

def outliers_removal(pcd, no_neighbors = 30, std_ratio = 1.0, vis = True):
    xyz              = np.asarray(pcd.points)
    tree             = KDTree(xyz)
    dist, ind        = tree.query(xyz[:], k = no_neighbors)
    local_density    = np.mean(dist, axis = 1)
    global_density   = np.mean(local_density)
    global_std       = np.std(local_density)
    inlier_indices   = np.where((local_density > global_density - std_ratio*global_std) & (local_density < global_density + std_ratio*global_std))
    xyz              = xyz[inlier_indices].astype('float32')
    if vis == True:
        pcd.paint_uniform_color([1, 0, 0])
        np.asarray(pcd.colors)[inlier_indices, :] = [0, 1, 0]
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(xyz)
    return filtered_pcd

def patch_normals_estimation(patch):
    centroid = np.mean(patch, axis = 0)
    k        = patch.shape[0]
    cov      = (1/k)*((patch-centroid).T).dot(patch-centroid)
    w, v     = np.linalg.eig(cov)
    idx      = w.argsort()[::-1]
    w        = w[idx]
    v        = v[:,idx]
    normals  = v[:,2]
    if normals.dot(centroid) > 0:
        normals = -normals
    return normals

def compute_normals(pool, pcd, no_neighbors = 30):
    xyz          = np.asarray(pcd.points)
    tree         = KDTree(xyz)
    dist, ind    = tree.query(xyz[:], k = no_neighbors)
    patches      = np.array([xyz[ind[i]] for i in range(ind.shape[0])])
    results      = pool.map(patch_normals_estimation, patches)
    results      = np.array(results)
    normals      = results
    return normals

def creases_removal(normals_patch):
    patch               = normals_patch[0]
    ind                 = normals_patch[1]
    cosine_similarities = np.array([np.dot(patch[0], patch[i]) for i in range(patch.shape[0])])
    min_idxes           = np.where(cosine_similarities <= 0.83)
    min_similarity      = cosine_similarities[min_idxes]
    outlier_idx         = ind[min_idxes]
    return outlier_idx

def removing_creases(pool, pcd, no_neighbors = 30, vis = False):
    xyz     = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    tree    = KDTree(xyz)
    dist, ind  = tree.query(xyz[:], k = no_neighbors)
    normals_patches            = [(normals[ind[i]], ind[i]) for i in range(ind.shape[0])]
    outliers                   = pool.map(creases_removal, normals_patches)
    outliers                   = np.concatenate(outliers).ravel()
    outliers                   = np.unique(outliers)

    if vis == True:
        sample_pcd         = o3d.geometry.PointCloud()
        sample_pcd.points  = o3d.utility.Vector3dVector(xyz)
        sample_pcd.normals = o3d.utility.Vector3dVector(normals)
        sample_pcd.paint_uniform_color([0, 1, 0])
        np.asarray(sample_pcd.colors)[outliers, :] = [1, 0, 0]
        sample_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([sample_pcd], point_show_normal=True)

    xyz     = np.delete(xyz, outliers, axis = 0).astype('float32')
    normals = np.delete(normals, outliers, axis=0)

    pcd.points            = o3d.utility.Vector3dVector(xyz)
    pcd.normals           = o3d.utility.Vector3dVector(normals)
    return pcd

def clustering(pool, pcd, eps = 0.005, min_points = 9, threshold = 300, no_neighbors = 30, vis = False):
    xyz     = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    labels            = np.array(pcd.cluster_dbscan(eps = eps, min_points = min_points, print_progress=False))
    noise_ind         = np.where(labels < 0)
    labels            = np.delete(labels, noise_ind[0])
    if vis == True:
        print("Highlight Noise")
        copied_pcd = copy.deepcopy(pcd)
        copied_pcd.paint_uniform_color([0, 1, 0])
        np.asarray(copied_pcd.colors)[noise_ind[0], :] = [1, 0, 0]
        copied_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([copied_pcd], point_show_normal=True)

    xyz             = np.delete(xyz, noise_ind[0], 0).astype('float32')
    normals         = np.delete(normals, noise_ind[0], 0)
    pcd.points      = o3d.utility.Vector3dVector(xyz)
    pcd.normals     = o3d.utility.Vector3dVector(normals)
    labels          = np.array(pcd.cluster_dbscan(eps = eps, min_points = min_points, print_progress=False))
    ids             = np.unique(labels)
    outliers        = []
    for i in range(len(ids)):
        cluster = np.where(labels == ids[i])[0]
        if cluster.shape[0] < threshold:
            outliers.append(cluster)
    if len(outliers) != 0:
        outliers             = np.concatenate(outliers).ravel()
    if vis == True:
        print("Highlight clusters having number of points smaller than given threshold")
        copied_pcd = copy.deepcopy(pcd)
        copied_pcd.paint_uniform_color([0, 1, 0])
        np.asarray(copied_pcd.colors)[outliers, :] = [1, 0, 0]
        copied_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([copied_pcd], point_show_normal=True)

    xyz             = np.delete(xyz, outliers, 0).astype('float32')
    normals         = np.delete(normals, outliers, 0)
    # update point clouds attributes
    pcd.points      = o3d.utility.Vector3dVector(xyz)
    pcd.normals     = o3d.utility.Vector3dVector(normals)

    # ---------------------------------> Final phase of clustering process
    # re-grouping retained clusters
    labels               = np.array(pcd.cluster_dbscan(eps = eps, min_points = min_points, print_progress = False))
    cluster_ids          = np.unique(labels)
    clusters             = []

    for i in range(len(cluster_ids)):
        cluster          = np.where(labels == cluster_ids[i])[0]
        cluster_xyz      = xyz[cluster]
        clusters.append(cluster)
        cluster_tree     = KDTree(cluster_xyz)
        dist, ind        = cluster_tree.query(cluster_xyz[:], k = no_neighbors)
        patches          = np.array([cluster_xyz[ind[i]] for i in range(ind.shape[0])])
        cluster_normals  = pool.map(patch_normals_estimation, patches)
        cluster_normals  = np.array(cluster_normals)
        normals[cluster] = cluster_normals

    # update point clouds attributes
    pcd.normals          = o3d.utility.Vector3dVector(normals)
    return pcd, clusters, cluster_ids

def down_sampled(pcd, voxel_size = 0.003):
    down_pcd        = pcd.o3d.geometry.PointCloud.voxel_down_sample(voxel_size = voxel_size)
    pcd.points      = down_pcd.points
    xyz             = np.asarray(pcd.points).astype('float32')
    return pcd

def plane_segmentation(pool, pcd, distance_threshold = 0.005, ransac_n = 3, num_iterations = 1000):
    samples = []
    xyz     = np.asarray(pcd.points)
    for i in range(num_iterations):
        index = np.random.choice(xyz.shape[0], ransac_n, replace=False)
        samples.append((xyz, index, distance_threshold))

    results  = pool.map(plane_fit, samples)
    no_inliers_statistics  = []
    for i in range(len(results)):
        no_inliers_statistics.append(results[i][1])
    no_inliers_statistics = np.array(no_inliers_statistics)
    idx = np.argmax(no_inliers_statistics)
    plane_model  = results[idx][0]
    plane_centroid = np.mean(xyz[results[idx][2]], axis = 0)
    return plane_model, plane_centroid

def plane_fit(sample):
    xyz                = sample[0]
    index              = sample[1]
    distance_threshold = sample[2]
    sample_pts         = xyz[index]
    # fitting plane into 3 random points
    n = np.cross(sample_pts[1, :] - sample_pts[0, :], sample_pts[2, :] - sample_pts[1, :])
    n = n / np.sqrt(np.sum(n**2))
    # align normals vector consistently towards camera's origin
    if n.dot(sample_pts[0,:]) > 0:
        n = -n
    # scalar component
    d = -np.dot(n, sample_pts[0, :])
    a = n[0]
    b = n[1]
    c = n[2]
    # calculate the distance of the point to the inlier plane
    dists        = np.abs(a*xyz[:,0] + b*xyz[:,1] + c*xyz[:,2] + d) / np.sqrt(a*a + b*b + c*c)
    inliers      = np.where(dists < distance_threshold)
    no_inliers   = len(inliers[0])
    plane_model  = [a, b, c, d]
    result       = [plane_model, no_inliers, inliers]
    return result

def clipping_from_plane(pcd, plane_model, threshold = 0.005):
    xyz = np.asarray(pcd.points)
    a                  = plane_model[0]
    b                  = plane_model[1]
    c                  = plane_model[2]
    d                  = plane_model[3]
    dist               = (a * xyz[:,0] + b*xyz[:,1] + c*xyz[:,2] + d) / np.sqrt(a*a + b*b + c*c)
    inliers            = np.where(dist > threshold)[0]
    xyz                = xyz[inliers].astype('float32')
    pcd.points         = o3d.utility.Vector3dVector(xyz)
    return pcd

if __name__ == "__main__":
    # ------------------------------> Set-up Logger
    logger               =   logging.getLogger("virtual_observer")
    logger.setLevel(logging.DEBUG)
    fmt       = "%(log_color)s%(levelname)s %(asctime)s.%(msecs)03d %(processName)s %(message)s"
    date_fmt  = "%y/%b/%Y %H:%M:%S"
    formatter = ColoredFormatter(fmt, datefmt=date_fmt, reset=True,
    log_colors={
        'DEBUG':    'bold_cyan',
        'INFO':     'bold_green',
        'WARNING':  'bold_yellow',
        'ERROR':    'bold_red',
        'CRITICAL': 'bold_red',   })
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)
    console_handler.setFormatter(formatter)
    logger.propagate = False
    # ------------------------------> Initialize Jacobian
    logger.warning("initializing jacobian ...")
    lambdified_jacobian_sphere                      = jacobian_sphere()
    lambdified_jacobian_cylinder                    = jacobian_cylinder()
    lambdified_jacobian_cone_axis                   = jacobian_cone_axis()
    lambdified_jacobian_cone                        = jacobian_cone()
    logger.info("complete initializing jacobian")
    logger.info("-"*77)
    # ------------------------------> Initialize Geometry Publisher
    geometry_publisher = rospy.Publisher('geometry_publisher', Path, queue_size=1)
    rospy.init_node('online_shape_perception', anonymous=True)
    # ------------------------------> Initialize Configuration Parameters
    voxel_size                                         = 0.003
    scales                                             = np.linspace(1.5, 7.0, num=30, endpoint=False).tolist()
    optim_config_params                                = {}

    optim_config_params["scales"]                      = scales
    optim_config_params["voxel_size"]                  = voxel_size

    optim_config_params["cylinder"]                    = {}
    optim_config_params["cylinder"]["damping_factor"]  = 100
    optim_config_params["cylinder"]["c"]               = 2
    optim_config_params["cylinder"]["threshold"]       = 10e-7

    # ----------------------> initializing multiprocessing
    pool        = Pool(mp.cpu_count(), init_worker)
    threadpool  = ThreadPool(processes = 100)
    # Initiating instance of camera class   
    cam         = IntelCamera([])
    counter     = 0
    # --------------------> Relative pose between camera and robot's base
    T_base_camera = np.array([[-0.06367482,  0.92597829, -0.37216893,  0.13944101],
                              [ 0.72274558, -0.21437222, -0.65702615, -0.26232728],
                              [-0.68817464, -0.31081947, -0.6555966,   1.10860615],
                              [ 0.        ,  0.        ,  0.       ,   1.        ]])
    try:
        while True:
            logger.info(f"---------------------> Computing frame No. {counter}")
            counter    += 1
            rgb_img, depth_img   = cam.stream()
            
            pcd         = cam.generate(depth_img,vox_size=0.002)
            pcd         = cam.crop_points()
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            # down sampling raw point cloud data
            # pcd         = down_sampled(pcd, voxel_size = 0.003)
            logger.warning("complete down sampling")
            # find supporting plane
            plane_model, plane_centroid = plane_segmentation(pool, pcd, distance_threshold = 0.005, ransac_n = 3, num_iterations = 1000)
            print(plane_model, "plane model")
            #plane_model = [-0.057420634, -0.39010644, -0.91897756, 0.8065622]
            logger.warning(f"found supporting plane: {plane_model[0]}x + {plane_model[1]}y + {plane_model[2]}z + {plane_model[3]} = 0")
            # clipping points above supporting plane
            pcd = clipping_from_plane(pcd, plane_model, threshold = 0.01)
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            logger.warning("complete clipping points above supporting plane")

            # ----------------------> filtering statistic outliers
            logger.warning("removing statistic outliers")
            pcd         = outliers_removal(pcd, no_neighbors = 30, std_ratio = 1.0, vis = False)
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            logger.warning("complete removing statistic outliers")
            logger.warning("-"*77)
            # ----------------------> compute normal vectors
            logger.warning("computing normal vectors")
            normals     = compute_normals(pool, pcd, no_neighbors = 30)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            logger.warning("complete computing normal vectors")
            logger.warning("-"*77)
            # ----------------------> creases removal
            logger.warning("removing creases")
            pcd = removing_creases(pool, pcd, no_neighbors = 30, vis = False)
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            logger.warning("complete removing creases")
            logger.warning("-"*77)
            # ----------------------> clustering
            logger.warning("clustering")
            pcd, clusters, cluster_ids = clustering(pool, pcd, eps = 0.005, min_points = 11, threshold = 300, no_neighbors = rmse, vis = False)
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            logger.warning("complete clustering")
            logger.warning(f"found {len(cluster_ids)} from scene")
            logger.warning("-"*77)
            
            # ----------------------> shape recognition
            geometry            = shape_recognition(pool, threadpool, logger, pcd, clusters, cluster_ids, optim_config_params)
            msg                 = Path()
            msg.header.frame_id = "map"
            msg.header.stamp    = rospy.Time.now()
            for k in range(len(geometry)):
                # Print out geometry information
                print(f"Geometry type {geometry[k]['type']}")
                print(f"Relative pose to camera")
                T_camera_object = geometry[k]["refined"]["T"]
                print(T_camera_object)
                if geometry[k]["type"] == "cylinder":
                    print(f"Length: {geometry[k]['refined']['height']}")
                    print(f"Radius: {abs(geometry[k]['refined']['model'][4,0])}")
                print("The relative pose between object and robot base")
                T_base_object = T_base_camera.dot(T_camera_object)
                print(T_base_object)
                print("-"*55)

                """
                Publish the geometry information using Path message
                """
                T = geometry[k]["refined"]["T"]
                p = SE3(T = T).p
                quaternion = SE3(T = T).q
                pose = PoseStamped()
                pose.header.frame_id    = geometry[k]["type"]
                if geometry[k]["type"] == "cylinder":
                    pose.header.stamp.secs  = int(abs(geometry[k]["refined"]["model"][4, 0]) * 1000)
                    pose.header.stamp.nsecs = int(geometry[k]["refined"]["height"] * 1000)
                if geometry[k]["type"] == "cone":
                    pose.header.stamp.secs  = int(geometry[k]["refined"]["cone_radius_bottom"] * 1000)
                    pose.header.stamp.nsecs = int(geometry[k]["refined"]["cone_height_bottom"] * 1000)
                if geometry[k]["type"] == "sphere":
                    pose.header.stamp.secs  = int(abs(geometry[k]["refined"]["model"][3,0]))
                if geometry[k]["type"] == "plane":
                    pose.header.stamp.secs  = geometry[k]["refined"]["length"]
                    pose.header.stamp.nsecs = geometry[k]["refined"]["width"]

                pose.header.seq         = k
                pose.pose.position.x    = p[0]
                pose.pose.position.y    = p[1]
                pose.pose.position.z    = p[2]
                pose.pose.orientation.x = quaternion[1]
                pose.pose.orientation.y = quaternion[2]
                pose.pose.orientation.z = quaternion[3]
                pose.pose.orientation.w = quaternion[0]
                msg.poses.append(pose)

                # robot_base = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)  
                # cylinder_frame = o3d.geometry.TriangleMesh.create_cylinder(radius=float(r), height=float(h), resolution=100, split=4)
                
                # cylinder_rot = copy.deepcopy(cylinder_frame).transform(t_init)
                # cylinder_rot.paint_uniform_color([1,0,0])
                # cylinder_rot_fin=copy.deepcopy(cylinder_rot).transform(T_B_O.dot(T))
                # o3d.visualization.draw_geometries([cylinder_rot, cylinder_frame, robot_base])
                # o3d.visualization.draw_geometries([robot_base, cylinder_rot_fin])
                
            geometry_publisher.publish(msg)


    except KeyboardInterrupt:
        print("stop")
        pool.terminate()
        pool.join()
        pid = os.getpid()                   # get process id
        os.kill(int(pid), signal.SIGKILL)
