import sys
import os
import yaml
import numpy as np

import cv2
import copy
import pyrealsense2 as rs

from cv2 import aruco
from camera import Camera

class Store_workspace_info:
    ref_path = os.getcwd()
    config = {}
    with open(ref_path+"/suction_config.yml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cam = Camera(cfg)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    marker_size = 0.068 # in meters
    white_area = 0.015 # in meters

    # if a == "1":
    READY100 = False
    READY101 = False
    READY102 = False
    used_ids = {"Reference":101, "Width":100, "Length":102, "type":"6x6"}

    # offset_from_corner = 0.01
    offset_from_corner = marker_size/2
    widths = []
    lengths = []
    iterations = 50
    loop = 0
    while True:
        if cam.device_product_line == "L500":
            rgb, _ = cam.stream()
        else:
            rgb, _ = cam.stream()
        gray_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(rgb.copy(), corners, ids)
        if np.shape(corners)[0] > 0:
                for i in range(np.shape(corners)[0]):
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_size, cameraMatrix=cam.camera_mat, distCoeffs=cam.dist_coeffs)
                    frame_markers = cv2.drawFrameAxes(frame_markers, cameraMatrix=cam.camera_mat, distCoeffs=cam.dist_coeffs, rvec=rvecs, tvec=tvecs, length=0.080, thickness=2)
                    ## for SE3 trasnformation matrix (marker with respect to the camera)
                    R, _ = cv2.Rodrigues(rvecs)
                    tvecs = np.reshape(tvecs, (3, 1))
                    cam2marker = np.concatenate((R, tvecs), axis = 1)
                    cam2marker = np.concatenate((cam2marker, np.array([[0, 0, 0, 1]])), axis = 0)
                    ## move point in the marker to the corner of the marker in the image
                    ## used marker id : 100, 101(reference), 102
                    if ids[i] == 101:
                        corner = np.reshape([-offset_from_corner, offset_from_corner, 0, 1], (4, 1))
                        corner101_from_cam = np.dot(cam2marker, corner)[:3]
                        pixel = (np.dot(cam.camera_mat, corner101_from_cam)/corner101_from_cam[-1])[:2]
                        pixel = pixel.astype(np.int64)
                        pixel = np.reshape(pixel, (2,))
                        pixel101 = tuple(pixel)
                        READY101 = True

                    elif ids[i] == 100:
                        corner = np.reshape([-offset_from_corner, offset_from_corner, 0, 1], (4, 1))
                        corner100_from_cam = np.dot(cam2marker, corner)[:3]
                        pixel = (np.dot(cam.camera_mat, corner100_from_cam)/corner100_from_cam[-1])[:2]
                        pixel = pixel.astype(np.int64)
                        pixel = np.reshape(pixel, (2,))
                        pixel100 = tuple(pixel)
                        READY100 = True
                    

                    elif ids[i] == 102:
                        corner = np.reshape([-offset_from_corner, -offset_from_corner, 0, 1], (4, 1))
                        corner102_from_cam = np.dot(cam2marker, corner)[:3]
                        pixel = (np.dot(cam.camera_mat, corner102_from_cam)/corner102_from_cam[-1])[:2]
                        pixel = pixel.astype(np.int64)
                        pixel = np.reshape(pixel, (2,))
                        pixel102 = tuple(pixel)
                        READY102 = True
                    
                    else:
                        pass
        
        if READY100 == True and READY101 == True and READY102 == True:
            vector_width = (corner100_from_cam - corner101_from_cam)
            width = np.math.sqrt(np.sum(vector_width**2))
            unitvertor_width = vector_width/width
            vector_length = (corner102_from_cam - corner101_from_cam)
            length = np.math.sqrt(np.sum(vector_length**2))
            unitvertor_length = vector_length/length     

            vector2corner4th = vector_width + vector_length
            corner_4th = copy.deepcopy(corner101_from_cam)
            corner_4th = corner_4th + vector2corner4th

            pixel = (np.dot(cam.camera_mat, corner_4th)/corner_4th[-1])[:2]
            pixel = pixel.astype(np.int64)
            pixel = np.reshape(pixel, (2,))
            pixel_4th = tuple(pixel)

            cv2.circle(img=frame_markers, center=pixel101, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(img=frame_markers, center=pixel102, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.circle(img=frame_markers, center=pixel_4th, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.line(img=frame_markers, pt1=pixel101, pt2=pixel100, color=(0, 0, 255), thickness=2)
            cv2.line(img=frame_markers, pt1=pixel101, pt2=pixel102, color=(0, 0, 255), thickness=2)
            cv2.line(img=frame_markers, pt1=pixel_4th, pt2=pixel102, color=(0, 0, 255), thickness=2)
            cv2.line(img=frame_markers, pt1=pixel_4th, pt2=pixel100, color=(0, 0, 255), thickness=2)
            cv2.putText(img=frame_markers, text="Bin Width: {:.3f}m".format(width), org=(10, 20), fontFace=0, fontScale=0.5, color=(0, 255, 0))
            cv2.putText(img=frame_markers, text="Bin Length: {:.3f}m".format(length), org=(10, 40), fontFace=0, fontScale=0.5, color=(0, 255, 0))

            cv2.imshow("res", frame_markers)
            cv2.waitKey(1)
            
            loop += 1
            widths.append(width)
            lengths.append(length)
            print("stacked: {}/{}".format(loop, iterations), end='\r')
            READY100, READY101, READY102 = False, False, False

        else:
            cv2.imshow("res", frame_markers)
            cv2.waitKey(1)
        
        if loop == iterations:
            print("")
            print("Record done!")
            break

    ## store bin info to yaml file
    W = np.average(widths).tolist()
    L = np.average(lengths).tolist()
    print("W, L = {:.3f}m, {:.3f}m".format(W, L))

    Bin = {}
    N432 = {}
    N432[cam.device_name] = {}
    N432[cam.device_name]["Width"] = W
    N432[cam.device_name]["Length"] = L
    N432[cam.device_name]["Marker_ids"] = used_ids
    N432[cam.device_name]["Marker_size"] = marker_size
    Bin["N432"] = N432
    with open(ref_path+"/bin.yml") as f:
        bin_info = yaml.load(f, Loader=yaml.FullLoader)
        # Add bin information to bin information data
        if bin_info is not None:
        ## There's pre-recorded info
            with open(ref_path+"/bin.yml", "w") as f:
                bin_info["N432"][cam.device_name] = N432[cam.device_name]
                yaml.dump(bin_info, f, default_flow_style=None)
        else:
        ## There's no pre-recorded info
            with open(ref_path+"/bin.yml", "w") as f:
                yaml.dump(Bin, f, default_flow_style=None)
    cam.stop