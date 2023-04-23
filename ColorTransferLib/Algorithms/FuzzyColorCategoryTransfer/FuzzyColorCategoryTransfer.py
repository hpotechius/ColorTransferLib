"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from numba import cuda
import math
import time
import cv2

import os
os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
from oct2py import octave, Oct2Py

import csv
import copy
import itertools
import open3d as o3d
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.BaseOptions import BaseOptions
from ColorTransferLib.ImageProcessing.Image import Image as Img
from copy import deepcopy
#from ColorTransferLib.Utils.Helper import check_compatibility
from .FaissKNeighbors import FaissKNeighbors
from pyhull.convex_hull import ConvexHull
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Color Transfer between Images
#   Author: Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley
#   Published in: IEEE Computer Graphics and Applications
#   Year of Publication: 2001
#
# Abstract:
#   We use a simple statistical analysis to impose one image's color characteristics on another. We can achieve color
#   correction by choosing an appropriate source image and apply its characteristic to another image.
#
# Link: https://doi.org/10.1109/38.946629
#
# Source:
#   - https://arnabfly.github.io/arnab_blog/fknn/
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class FuzzyColorCategoryTransfer:
    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }

    # each color has an id depending on its position in the "color_terms"-variable, i.e.
    # Red = 0
    # Yellow = 1
    # Green = 2
    # Blue = 3
    # Black = 4
    # White = 5
    # Grey = 6
    # Orange = 7
    # Brown = 8
    # Pink = 9
    # Purple = 10
    color_terms = np.array(["Red", "Yellow", "Green", "Blue", "Black", "White", "Grey", "Orange", "Brown", "Pink", "Purple"])
    color_terms_id = {"Red":0, "Yellow":1, "Green":2, "Blue":3, "Black":4, "White":5, "Grey":6, "Orange":7, "Brown":8, "Pink":9, "Purple":10}

    # this variable is only used for rendering, not for the actual algorithm
    color_samples = {
        "Red": np.array([1.0,0.0,0.0]),
        "Yellow":np.array([1.0,1.0,0.0]),
        "Green": np.array([0.0,1.0,0.0]),
        "Blue": np.array([0.0,0.0,1.0]),
        "Black": np.array([0.0,0.0,0.0]),
        "White": np.array([1.0,1.0,1.0]),
        "Grey": np.array([0.5,0.5,0.5]),
        "Orange": np.array([1.0,0.5,0.0]),
        "Brown": np.array([0.4,0.2,0.1]),
        "Pink": np.array([0.85,0.5,0.75]),
        "Purple": np.array([0.4,0.01,0.77]),
    }
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "FuzzyColorCategoryTransfer",
            "title": "...",
            "year": 2023,
            "abstract": "..."
        }

        return info
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_colormapping_dataset(path):
        color_mapping = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    color_mapping.append([float(row[0]), float(row[1]), float(row[2]), float(np.where(FCCT.color_terms == row[3])[0][0])])
                line_count += 1

        color_mapping = np.asarray(color_mapping)
        colors = color_mapping[:,:3] / 255
        colors = np.expand_dims(colors, axis=1).astype("float32")
        hsv_colors = FCCT.RGB2cartHSV(colors)
        hsv_colors = np.squeeze(hsv_colors)
        labels = color_mapping[:,3].astype("int64")
        return hsv_colors, labels

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def fuzzy_knn(colors, labels, src_color, ref_color, k=100):
        src_num = src_color.shape[0]
        ref_num = ref_color.shape[0]

        neigh = FaissKNeighbors(k=k)
        neigh.fit(colors, labels)

        src_preds, src_votes, src_distances = neigh.predict(src_color) 
        ref_preds, ref_votes, ref_distances = neigh.predict(ref_color)

        # shape -> (#points, #labels)
        src_membership = FCCT.__calc_membership(src_votes, src_distances, src_num, 2)
        ref_membership = FCCT.__calc_membership(ref_votes, ref_distances, ref_num, 2)


        # sort colors by their categories with membership
        # [1] color_cats_src["Red"]: colors which belongs to the category Red
        #     -> shape: (#colors, 3)
        # [2] color_cats_src_ids["Red"]: initial positions within the original color array 
        #     -> shape: (#colors, 1)
        #     -> necessary in order to get the initial image
        # [3] color_cats_ref_mem["Red"]: membership to labels per color
        #     -> shape: (#colors, 11)
        color_cats_src, color_cats_src_ids, color_cats_src_mem = FCCT.__sort_by_category(src_preds, src_color, src_membership)
        color_cats_ref, color_cats_ref_ids, color_cats_ref_mem = FCCT.__sort_by_category(ref_preds, ref_color, ref_membership)

        return color_cats_src, color_cats_src_ids, color_cats_src_mem, color_cats_ref, color_cats_ref_ids, color_cats_ref_mem

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def write_colors_as_PC(colors, rgb_colors, path):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(colors)
        pc.colors = o3d.utility.Vector3dVector(rgb_colors)
        o3d.io.write_point_cloud(path, pc)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def polar_to_cartesian(hsv_colors):
        hue_angle = hsv_colors[:,0]
        sat_radius = hsv_colors[:,1]
        value = hsv_colors[:,2]

        # weighting if the radius in order to get the HSV-cone
        weighted_radius = sat_radius * value / 255.0

        x_pos = weighted_radius * np.cos(np.radians(hue_angle))
        y_pos = weighted_radius * np.sin(np.radians(hue_angle))
        z_pos = value

        hsv_cart = np.concatenate((np.expand_dims(x_pos,1), np.expand_dims(y_pos,1), np.expand_dims(z_pos,1)), axis=1)
        return hsv_cart

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def cartesian_to_polar(hsv_colors):
        x = hsv_colors[:,0]
        y = hsv_colors[:,1]
        z = hsv_colors[:,2]

        # if z == 0:
        #     hue = 0
        #     sat = 0
        #     val = 0
        # else:
        # weighting if the radius in order to get the HSV-cone
        radius_weighting = 255.0 / z


        # hue has to be converted to degrees
        # Note: hue is in range [-180, 180] -> value smaller than 0 hat to be mapped to [180, 360]
        hue = np.degrees(np.arctan2(y, x))
        hue = (hue + 360) % 360

        sat = np.sqrt(x ** 2 + y ** 2) * radius_weighting
        val = z

        hsv_polar = np.concatenate((np.expand_dims(hue,1), np.expand_dims(sat,1), np.expand_dims(val,1)), axis=1)
        return hsv_polar
    
    # ------------------------------------------------------------------------------------------------------------------
    # Transfers RGB values to cartesian HSV
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def RGB2cartHSV(rgb):
        # if src_color is in range [0,1] the SV channels are also in range [0,1] but H channel is in range [0,360]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)[:,0,:]
        hsv[:,1:3] = hsv[:,1:3] * 255
        hsv_cart = FCCT.polar_to_cartesian(hsv)
        return hsv_cart
    
    # ------------------------------------------------------------------------------------------------------------------
    # Transfers RGB values to cartesian HSV
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def HSV2cartRGB(hsv):
        #hsv[:,1:3] = hsv[:,1:3] * 255
        hsv_polar = FCCT.cartesian_to_polar(hsv)
        # if src_color is in range [0,1] the SV channels are also in range [0,1] but H channel is in range [0,360]
        hsv_polar[:,1:3] = hsv_polar[:,1:3] / 255
        # returns normalized rgb values in range [0, 1]
        rgb = cv2.cvtColor(np.expand_dims(hsv_polar, axis=1), cv2.COLOR_HSV2RGB_FULL)

        return rgb
    
    
    # ------------------------------------------------------------------------------------------------------------------
    # Get transfer direction between source and reference
    # Note: White, Grey and Black will be transformed to White, Grey and Black
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_transfer_direction(CV_src, CV_ref, EVV_src, EVV_ref):
        predefined_pairs = [
            #("White", "White"), ("Grey","Grey"), ("Black","Black"), ("Purple", "Pink"), ("Green", "Green"), ("Blue", "Blue")
        ]

        volumes_src = []
        volumes_ref = []
        for c in FCCT.color_terms:
            # (Color, Volume, Center, Eigenvectors, Eigenvalues)
            volumes_src.append((c, CV_src[c][1], CV_src[c][0], EVV_src[c][0], EVV_src[c][1]))
            volumes_ref.append((c, CV_ref[c][1], CV_ref[c][0], EVV_ref[c][0], EVV_ref[c][1]))

        # create class pairs for white-white, grey-grey and black-black
        class_pairs_wgb = []
        #fixed = ["White", "Grey", "Black"]
        for elem_src, elem_ref in predefined_pairs:
            col_src = next(filter(lambda x : x[0]==elem_src, volumes_src))
            col_ref = next(filter(lambda x : x[0]==elem_ref, volumes_ref))
            class_pairs_wgb.append([col_src, col_ref])

        # get transfer direction of similar classes if the volumes have only a 20% difference
        # TODO

        # remove white, grey and black for the sorting procedure
        # col_src = list(filter(lambda x : x[0]!="White" and x[0]!="Grey" and x[0]!="Black", volumes_src))
        # col_ref = list(filter(lambda x : x[0]!="White" and x[0]!="Grey" and x[0]!="Black", volumes_ref))
        col_src = list(filter(lambda x : x[0] not in [a for a, _ in predefined_pairs], volumes_src))
        col_ref = list(filter(lambda x : x[0] not in [b for _, b in predefined_pairs], volumes_ref))

        sorted_volumes_src = sorted(col_src, key=lambda x: x[1])
        sorted_volumes_ref = sorted(col_ref, key=lambda x: x[1])
        class_pairs = [[s, r] for s, r in zip(sorted_volumes_src,sorted_volumes_ref)]
        return class_pairs_wgb + class_pairs
    
    # ------------------------------------------------------------------------------------------------------------------
    # calculates eigenvectors and -values
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def getEigen(CH):
        EVV = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for c in FCCT.color_terms:
            c_hull, validity = CH[c]
            if validity:
                # Resampling of the convex hulls as uniformly distributed point cloud
                pc_src = c_hull.sample_points_uniformly(number_of_points=1000)
                pc_src.colors = o3d.utility.Vector3dVector(np.full((1000,3), FCCT.color_samples[c]))
                # apply PCA to get eigenvectors and -values
                pca_src = PCA(n_components = 3)
                pca_src.fit_transform(np.asarray(pc_src.points))
                eigenvectors_src = pca_src.components_
                eigenvalues_src = pca_src.explained_variance_
            else:
                eigenvectors_src = [(0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0)]
                eigenvalues_src = [0.0, 0.0, 0.0]
            EVV[c] = (eigenvectors_src, eigenvalues_src)
        return EVV

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rotation(class_pairs, color_cats_src):
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]
            if src_vol == 0.0 or ref_vol == 0.0:
                continue
            # get rotation angle
            src_xy = src_cen[:2]
            ref_xy = ref_cen[:2]

            # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
            dot = src_xy[0] * ref_xy[0] + src_xy[1] * ref_xy[1]
            det = src_xy[0] * ref_xy[1] - src_xy[1] * ref_xy[0]
            radians = math.atan2(det, dot)

            # Rotation in xy plane. z-Axis will be ignored
            x = color_cats_src[src_col][:,0]
            y = color_cats_src[src_col][:,1]
            xx = x * np.cos(radians) - y * np.sin(radians)
            yy = x * np.sin(radians) + y * np.cos(radians)

            color_cats_src[src_col][:,0] = xx
            color_cats_src[src_col][:,1] = yy
        return color_cats_src
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_rotation_matrix(class_pairs):
        rots = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]
            if src_vol == 0.0 or ref_vol == 0.0:
                rots[src_col] = np.eye(4)
                continue

            # get rotation angle
            src_xy = src_cen[:2]
            ref_xy = ref_cen[:2]

            # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
            dot = src_xy[0] * ref_xy[0] + src_xy[1] * ref_xy[1]
            det = src_xy[0] * ref_xy[1] - src_xy[1] * ref_xy[0]
            radians = math.atan2(det, dot)

            rotation_mat = np.array([[np.cos(radians), -np.sin(radians), 0.0, 0.0],
                                     [np.sin(radians), np.cos(radians) , 0.0, 0.0],
                                     [0.0            , 0.0                , 1.0, 0.0],
                                     [0.0             , 0.0                , 0.0, 1.0]])

            rots[src_col] = rotation_mat
        return rots
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_translation_matrix(class_pairs, CV_src_new):
        trans = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            src_cen_new, src_vol_new = CV_src_new[src_col]

            if src_vol_new == 0.0 or ref_vol == 0.0:
                trans[src_col] = np.eye(4)
                continue

            translation = ref_cen - src_cen_new

            translation_mat = np.array([[1.0, 0.0, 0.0, translation[0]],
                                        [0.0, 1.0, 0.0, translation[1]],
                                        [0.0, 0.0, 1.0, translation[2]],
                                        [0.0, 0.0, 0.0, 1.0]])

            trans[src_col] = translation_mat
        return trans

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def translation(class_pairs, color_cats_src, CV_src_new):
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            src_cen_new, src_vol_new = CV_src_new[src_col]

            if src_vol_new == 0.0 or ref_vol == 0.0:
                continue

            translation = ref_cen - src_cen_new

            rep = np.tile(translation, (color_cats_src[src_col].shape[0],1))

            color_cats_src[src_col] += rep
        return color_cats_src
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_scaling_matrix(class_pairs, color_cats_src, color_cats_ref):
        scal = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_src_temp = copy.deepcopy(color_cats_src)
        color_cats_ref_temp = copy.deepcopy(color_cats_ref)

        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            if src_vol == 0.0 or ref_vol == 0.0:
                scal[src_col] = np.eye(4)
                continue

            
            # x_min, y_min, z_min = np.amin(color_cats_src_temp[src_col], axis=0)
            # x_max, y_max, z_max = np.amax(color_cats_src_temp[src_col], axis=0)

            # center_src = np.array([(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2])

            # trans_center = np.array([[1.0, 0.0, 0.0, -center_src[0]],
            #                          [0.0, 1.0, 0.0, -center_src[1]],
            #                          [0.0, 0.0, 1.0, -center_src[2]],
            #                          [0.0, 0.0, 0.0, 1.0        ]])
            
            # x_min, y_min, z_min = np.amin(color_cats_ref_temp[ref_col], axis=0)
            # x_max, y_max, z_max = np.amax(color_cats_ref_temp[ref_col], axis=0)

            # center_ref = np.array([(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2])

            
            # rep = np.tile(center_src, (color_cats_src_temp[src_col].shape[0],1))
            # color_cats_src_temp[src_col] -= rep
            # rep = np.tile(center_ref, (color_cats_ref_temp[ref_col].shape[0],1))
            # color_cats_ref_temp[ref_col] -= rep

            trans_center = np.array([[1.0, 0.0, 0.0, -ref_cen[0]],
                                     [0.0, 1.0, 0.0, -ref_cen[1]],
                                     [0.0, 0.0, 1.0, -ref_cen[2]],
                                     [0.0, 0.0, 0.0, 1.0        ]])
            


            rep = np.tile(ref_cen, (color_cats_src_temp[src_col].shape[0],1))
            color_cats_src_temp[src_col] -= rep
            rep = np.tile(ref_cen, (color_cats_ref_temp[ref_col].shape[0],1))
            color_cats_ref_temp[ref_col] -= rep

            total_scaling = np.eye(4)

            for r in range(50):
                ran_rotation = R.random().as_matrix()
                ran_rot_mat = np.eye(4)
                ran_rot_mat[:3,:3] = ran_rotation


                color_cats_src_temp[src_col] = FCCT.transform_single(color_cats_src_temp[src_col], ran_rot_mat)

                #color_cats_ref_temp[ref_col] = color_cats_ref_temp[ref_col].dot(ran_rotation)
                color_cats_ref_temp[ref_col] = FCCT.transform_single(color_cats_ref_temp[ref_col], ran_rot_mat)

                x_min, y_min, z_min = np.amin(color_cats_src_temp[src_col], axis=0)
                x_max, y_max, z_max = np.amax(color_cats_src_temp[src_col], axis=0)

                x_stretch = x_max - x_min
                y_stretch = y_max - y_min
                z_stretch = z_max - z_min
                scale_down = np.array([1.0/x_stretch, 1.0/y_stretch, 1.0/z_stretch])
                
                scale_down_m = np.tile(scale_down, (color_cats_src_temp[src_col].shape[0],1))
                color_cats_src_temp[src_col] = color_cats_src_temp[src_col] * scale_down_m

                #scale_down_mat = np.eye(4)
                scale_down_mat = np.array([[scale_down[0], 0.0            , 0.0            , 0.0],
                                           [0.0            , scale_down[1], 0.0            , 0.0],
                                           [0.0            , 0.0            , scale_down[2], 0.0],
                                           [0.0            , 0.0            , 0.0            , 1.0]])


                x_min, y_min, z_min = np.amin(color_cats_ref_temp[ref_col], axis=0)
                x_max, y_max, z_max = np.amax(color_cats_ref_temp[ref_col], axis=0)

                x_stretch = x_max - x_min
                y_stretch = y_max - y_min
                z_stretch = z_max - z_min
                scale_up = np.array([x_stretch, y_stretch, z_stretch])

                scale_up_m = np.tile(scale_up, (color_cats_src_temp[src_col].shape[0],1))
                color_cats_src_temp[src_col] = color_cats_src_temp[src_col] * scale_up_m

                scale_up_mat = np.array([[scale_up[0], 0.0          , 0.0          , 0.0],
                                         [0.0          , scale_up[1], 0.0          , 0.0],
                                         [0.0          , 0.0          , scale_up[2], 0.0],
                                         [0.0          , 0.0          , 0.0          , 1.0]])
                

                ran_rotation_inv = np.transpose(ran_rotation)
                ran_rotinv_mat = np.eye(4)
                ran_rotinv_mat[:3,:3] = ran_rotation_inv

                # color_cats_src_temp[src_col] = color_cats_src_temp[src_col].dot(ran_rotation_inv)
                # color_cats_ref_temp[ref_col] = color_cats_ref_temp[ref_col].dot(ran_rotation_inv)
                color_cats_src_temp[src_col] = FCCT.transform_single(color_cats_src_temp[src_col], ran_rotinv_mat)
                color_cats_ref_temp[ref_col] = FCCT.transform_single(color_cats_ref_temp[ref_col], ran_rotinv_mat)

                total_scaling = ran_rotinv_mat @ scale_down_mat @ scale_up_mat @ ran_rot_mat @ total_scaling

            # move back to original position
            trans_center_back = np.array([[1.0, 0.0, 0.0, ref_cen[0]],
                                          [0.0, 1.0, 0.0, ref_cen[1]],
                                          [0.0, 0.0, 1.0, ref_cen[2]],
                                          [0.0, 0.0, 0.0, 1.0]])
            
            rep = np.tile(ref_cen, (color_cats_src_temp[src_col].shape[0],1))
            color_cats_src_temp[src_col] += rep
            rep = np.tile(ref_cen, (color_cats_ref_temp[ref_col].shape[0],1))
            color_cats_ref_temp[ref_col] += rep
                                     

            total_transform = trans_center_back @ total_scaling @ trans_center
   

            scal[src_col] = total_transform


        return scal
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def scaling(class_pairs, color_cats_src, color_cats_ref):
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, ref_evec, ref_eval = elem[1]

            if src_vol == 0.0 or ref_vol == 0.0:
                continue

            # move temporarily to the origin for proper scaling
            rep = np.tile(ref_cen, (color_cats_src[src_col].shape[0],1))
            color_cats_src[src_col] -= rep
            rep = np.tile(ref_cen, (color_cats_ref[ref_col].shape[0],1))
            color_cats_ref[ref_col] -= rep

            # rotate source by eigenvectors of reference
            """
            rotation_ref = np.transpose(ref_evec)
            color_cats_src[src_col] = rotation_ref.dot(color_cats_src[src_col].T).T
            """

            # scale source to unit
            for r in range(50):
                ran_rotation = R.random().as_matrix()
                color_cats_src[src_col] = color_cats_src[src_col].dot(ran_rotation)
                color_cats_ref[ref_col] = color_cats_ref[ref_col].dot(ran_rotation)

                x_min, y_min, z_min = np.amin(color_cats_src[src_col], axis=0)
                x_max, y_max, z_max = np.amax(color_cats_src[src_col], axis=0)

                x_stretch = x_max - x_min
                y_stretch = y_max - y_min
                z_stretch = z_max - z_min
                stretch = np.array([x_stretch, y_stretch, z_stretch])
                scale_down = np.tile(stretch, (color_cats_src[src_col].shape[0],1))
                color_cats_src[src_col] = color_cats_src[src_col] / scale_down

                # scaling via eigenvectors and -values
                x_min, y_min, z_min = np.amin(color_cats_ref[ref_col], axis=0)
                x_max, y_max, z_max = np.amax(color_cats_ref[ref_col], axis=0)

                x_stretch = x_max - x_min
                y_stretch = y_max - y_min
                z_stretch = z_max - z_min
                stretch = np.array([x_stretch, y_stretch, z_stretch])
                scale_up = np.tile(stretch, (color_cats_src[src_col].shape[0],1))
                color_cats_src[src_col] = color_cats_src[src_col] * scale_up

                ran_rotation_inv = np.transpose(ran_rotation)
                color_cats_src[src_col] = color_cats_src[src_col].dot(ran_rotation_inv)
                color_cats_ref[ref_col] = color_cats_ref[ref_col].dot(ran_rotation_inv)
                """
                scale = np.tile(ref_eval, (color_cats_src[src_col].shape[0],1))
                color_cats_src[src_col] = color_cats_src[src_col] * scale
                """

            # move back to original position
            rep = np.tile(ref_cen, (color_cats_src[src_col].shape[0],1))
            color_cats_src[src_col] += rep
            rep = np.tile(ref_cen, (color_cats_ref[ref_col].shape[0],1))
            color_cats_ref[ref_col] += rep

        return color_cats_src
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def applyTransformation(class_pairs, color_cats_src, rotation_matrix, translation_matrix, scaling_matrix):
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            if color_cats_src[src_col].shape[0] == 0:
                continue

            print(src_col + " " + ref_col)
            M = scaling_matrix[src_col] @ (translation_matrix[src_col] @ rotation_matrix[src_col])
            #M = translation_matrix[src_col] @ rotation_matrix[src_col]
            rep = np.tile(M, (color_cats_src[src_col].shape[0],1,1))

            # extend source by one dimension to get homogenous coordinates
            color_cats_src[src_col] = np.concatenate((color_cats_src[src_col], np.ones((color_cats_src[src_col].shape[0], 1))), axis=1)

            color_cats_src[src_col] = np.einsum('ijk,ik->ij', rep, color_cats_src[src_col])

            #remove last dimension to ger cartesian coordinates
            color_cats_src[src_col] = color_cats_src[src_col][:,:3]

            # print(color_cats_src[src_col].shape)
            # print(M.shape)
            # print(rep.shape)
            # exit()

            
        return color_cats_src
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def transform(points, transform):
        points_out = copy.deepcopy(points)
        for c in FCCT.color_terms:
            # check if category contains points
            if points_out[c].shape[0] == 0:
                continue

            rep = np.tile(transform[c], (points_out[c].shape[0],1,1))
            # extend source by one dimension to get homogenous coordinates
            points_out[c] = np.concatenate((points_out[c], np.ones((points_out[c].shape[0], 1))), axis=1)

            points_out[c] = np.einsum('ijk,ik->ij', rep, points_out[c])

            #remove last dimension to ger cartesian coordinates
            points_out[c] = points_out[c][:,:3]
        return points_out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def transform_weighted(points, memberships, transform):
        points_temp = copy.deepcopy(points)
        points_out = copy.deepcopy(points)
        for cx in FCCT.color_terms:
            # check if category contains points
            if points_temp[cx].shape[0] == 0:
                continue


            cat_temp = np.zeros_like(points_out[cx])


            # iterate over each transformation matrix (11 in total)
            for c in FCCT.color_terms:
                rep = np.tile(transform[c], (points_temp[cx].shape[0],1,1))
                # extend source by one dimension to get homogenous coordinates
                temp_points = np.concatenate((points_temp[cx], np.ones((points_temp[cx].shape[0], 1))), axis=1)

                temp_points = np.einsum('ijk,ik->ij', rep, temp_points)

                #remove last dimension to get cartesian coordinates
                temp_points = temp_points[:,:3]

                # weighting of the result with the membership value
                # membership values for all point within main category cx -> (#points, 1)
                membership_vec = memberships[cx][:,FCCT.color_terms_id[c]]

                membership_vec = np.concatenate((np.expand_dims(membership_vec,1),np.expand_dims(membership_vec,1),np.expand_dims(membership_vec,1)), axis=1)

                cat_temp += temp_points * membership_vec
            points_out[cx] = cat_temp

        return points_out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def transform_single(points, transform):
        points_out = copy.deepcopy(points)
        # check if category contains points
        if points_out.shape[0] == 0:
            return points_out

        rep = np.tile(transform, (points_out.shape[0],1,1))
        # extend source by one dimension to get homogenous coordinates
        points_out = np.concatenate((points_out, np.ones((points_out.shape[0], 1))), axis=1)

        points_out = np.einsum('ijk,ik->ij', rep, points_out)

        #remove last dimension to ger cartesian coordinates
        points_out = points_out[:,:3]
        return points_out
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()

        # check if method is compatible with provided source and reference objects
        #output = check_compatibility(src, ref, FCCT.compatibility)
        output = {
            "status_code": 0,
            "response": "",
            "object": None
        }

        # Preprocessing: Convert colorspace from RGB to HSV
        hsv_cart_src = FCCT.RGB2cartHSV(src.get_colors())
        hsv_cart_ref = FCCT.RGB2cartHSV(ref.get_colors())
        rgb_out_orig = deepcopy(src.get_colors()) 
        out_img = deepcopy(src)


        # rgb_src = FCCT.HSV2cartRGB(hsv_cart_src)
        # output_colors = np.clip(rgb_src, 0, 1)
        # out_img.set_colors(output_colors)
        # out_img.write("/home/potechius/Downloads/result.png")
        # exit()

        # VISUALIZATION 00
        """
        FCCT.write_colors_as_PC(hsv_cart_src, src.get_colors()[:,0,:],"/home/potechius/Downloads/FCCT_Tests/00_src_points.ply")
        FCCT.write_colors_as_PC(hsv_cart_ref, ref.get_colors()[:,0,:],"/home/potechius/Downloads/FCCT_Tests/00_ref_points.ply")
        """

        # Read Color Dataset in HSV color space
        colors, labels = FCCT.get_colormapping_dataset("Models/BasicColorCategoryTransfer/colormapping.csv")

        # Apply Fuzzy KNN
        # color_cats_src = {"Red": np.array([...]), "Yellow": np.array([...])} 
        # -- Contains pixel colors
        # color_cats_src_ids {"Red": np.array([...]), "Yellow": np.array([...])}
        # -- contains positions within the original color array, i.e. src.get_colors()
        # color_cats_src_mem color_cats_src_ids {"Red": np.array([...]), "Yellow": np.array([...])}
        # -- contains per point 11 values with probabilities
        color_cats_src, color_cats_src_ids, color_cats_src_mem, color_cats_ref, color_cats_ref_ids, color_cats_ref_mem = FCCT.fuzzy_knn(colors, labels, hsv_cart_src, hsv_cart_ref)
        
        # VISUALIZATION 01
        """
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            # check if any points belong to this category
            if color_cats_src[c].shape[0] != 0:
                rep = np.tile(colorv, (color_cats_src[c].shape[0],1))
                FCCT.write_colors_as_PC(color_cats_src[c], rep,"/home/potechius/Downloads/FCCT_Tests/01_src_KNN/01_src_points_" + c + ".ply")
            if color_cats_ref[c].shape[0] != 0:
                rep = np.tile(colorv, (color_cats_ref[c].shape[0],1))
                FCCT.write_colors_as_PC(color_cats_ref[c], rep,"/home/potechius/Downloads/FCCT_Tests/01_ref_KNN/01_ref_points_" + c + ".ply")
        """

        # Convex Hull Calculation
        # CH_src["Red"] -> (mesh, validity)
        # mesh: the convex hull as triangle mesh
        # validity: True if a convex hull is computed
        CH_src = FCCT.__calc_convex_hulls(color_cats_src)
        CH_ref = FCCT.__calc_convex_hulls(color_cats_ref)

        # VISUALIZATION 02
        """
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                FCCT.__write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCCT_Tests/02_src_CH/02_src_CH_"+c+".ply", 
                                              color=colorv)
            if CH_ref[c][1]:
                FCCT.__write_convex_hull_mesh(mesh=CH_ref[c][0],
                                              path="/home/potechius/Downloads/FCCT_Tests/02_ref_CH/02_ref_CH_"+c+".ply", 
                                              color=colorv)
        """

        # Estimate Transfer Directions based on volume
        # Note: White, Grey and Black will only be transferred to the same color
        CV_src = FCCT.__calc_bary_center_volume(CH_src)
        CV_ref = FCCT.__calc_bary_center_volume(CH_ref)

        # Calculate Eigenvectors and -values
        EVV_src = FCCT.getEigen(CH_src)
        EVV_ref = FCCT.getEigen(CH_ref)

        # get transfer directions
        class_pairs = FCCT.get_transfer_direction(CV_src, CV_ref, EVV_src, EVV_ref)

        # Print Transfer Directions and volumes
        for elem in class_pairs:
            print(elem[0][0] + " - " + elem[1][0])
            print(str(elem[0][1]) + " - " + str(elem[1][1]))
            print("\n")

        # Apply Rotation and Translation
        #color_cats_src = FCCT.rotation(class_pairs, color_cats_src)
        rotation_matrix = FCCT.get_rotation_matrix(class_pairs)

        # VISUALIZATION 03
        """
        CH_src = FCCT.__calc_convex_hulls(FCCT.transform(color_cats_src, rotation_matrix))
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                FCCT.__write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCCT_Tests/03_src_CH_rotated/03_src_CH_rotated_"+c+".ply", 
                                              color=colorv)
        exit()
        """

        # ReCalculate center and volume of rotated source
        #CH_src_new = FCCT.__calc_convex_hulls(color_cats_src)
        #CV_src_new = FCCT.__calc_bary_center_volume(CH_src_new)
        
        # Apply Translation
        color_cats_src_temp = FCCT.transform(color_cats_src, rotation_matrix)

        CH_src_new = FCCT.__calc_convex_hulls(color_cats_src_temp)
        CV_src_new = FCCT.__calc_bary_center_volume(CH_src_new)
        
        #color_cats_src = FCCT.translation(class_pairs, color_cats_src, CV_src_new)
        translation_matrix = FCCT.get_translation_matrix(class_pairs, CV_src_new)

        # Scaling
        #color_cats_src = FCCT.scaling(class_pairs, color_cats_src, color_cats_ref)

        color_cats_src_temp = FCCT.transform(color_cats_src_temp, translation_matrix)    

        # VISUALIZATION 04
        """
        CH_src = FCCT.__calc_convex_hulls(color_cats_src_temp)
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                FCCT.__write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCCT_Tests/04_src_CH_translated/04_src_CH_translated_"+c+".ply", 
                                              color=colorv)
        exit()
        """

        scaling_matrix = FCCT.get_scaling_matrix(class_pairs, color_cats_src_temp, color_cats_ref)
        #color_cats_src = FCCT.scaling(class_pairs, color_cats_src_temp, color_cats_ref)

        color_cats_src_temp = FCCT.transform(color_cats_src_temp, scaling_matrix)

        # VISUALIZATION 05
        """
        CH_src = FCCT.__calc_convex_hulls(color_cats_src)
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                FCCT.__write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCCT_Tests/05_src_CH_scaled/05_src_CH_scaled_"+c+".ply", 
                                              color=colorv)
        """

        # get the 11 transformation matrices
        affine_transform = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for c in FCCT.color_terms:
            affine_transform[c] = scaling_matrix[c] @ translation_matrix[c] @ rotation_matrix[c]


        #color_cats_src= FCCT.transform(color_cats_src, affine_transform)
        color_cats_src= FCCT.transform_weighted(color_cats_src, color_cats_src_mem, affine_transform)

        # Histogram Matching per category
        # for elem_src, elem_ref in class_pairs:
        #     if color_cats_src[elem_src[0]].shape[0] == 0:
        #         continue
        #     color_cats_src[elem_src[0]] = FCCT.histogram_matching(color_cats_src[elem_src[0]], color_cats_ref[elem_ref[0]])


        #color_cats_src = FCCT.applyTransformation(class_pairs, color_cats_src, rotation_matrix, translation_matrix, scaling_matrix)
        
        #-------------------------------------------

        # write mesh

        for c in FCCT.color_terms:
            for color, idx in zip(color_cats_src[c], color_cats_src_ids[c]):
                hsv_cart_src[idx] = color

        # Histogram Matching
        #hsv_cart_src = FCCT.histogram_matching(hsv_cart_src, hsv_cart_ref)

        # # mex -g  mex_mgRecolourParallel_1.cpp COMPFLAGS="/openmp $COMPFLAGS"
        # octave.addpath(octave.genpath('.'))
        # #octave.addpath(octave.genpath('module/Algorithms/TpsColorTransfer/L2RegistrationForCT'))
        # octave.eval('pkg load image')
        # octave.eval('pkg load statistics')
        # octave.eval("dir")


    
        rgb_out = FCCT.HSV2cartRGB(hsv_cart_src)

        #rgb_out = octave.regrain(rgb_out_orig * 255, rgb_out * 255) / 255
        # INFO Histogram Matching in RGB gives better results than in HSV
        rgb_out = np.expand_dims(FCCT.histogram_matching(rgb_out[:,0,:]*255, ref.get_colors()[:,0,:]*255), 1) / 255
        print(rgb_out.shape)
        """
        FCCT.write_colors_as_PC(hsv_cart_src, rgb_out[:,0,:], "/home/potechius/Downloads/FCCT_Tests/00_out_points_2.ply")
        """


        output_colors = np.clip(rgb_out, 0, 1)
        out_img.set_colors(output_colors)

        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output



    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------ 
    # def __write_convex_hull_mesh(colors, shape, path, color, color_space="LAB"):
    #     if color_space == "RGB":
    #         ex = np.asarray(colors)[:, np.newaxis]
    #         cex = cv2.cvtColor(ex, cv2.COLOR_Lab2RGB)
    #         mesh, validity = FCCT.__calc_convex_hull(cex.squeeze())
    #     else:
    #         mesh, validity = FCCT.__calc_convex_hull(colors)

    #     if validity:
    #         colors = np.full(shape, color)
    #         mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    #         o3d.io.write_triangle_mesh(filename=path, 
    #                                 mesh=mesh, 
    #                                 write_ascii=True,
    #                                 write_vertex_normals=False,
    #                                 write_vertex_colors=True,
    #                                 write_triangle_uvs=False)
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------ 
    def __write_convex_hull_mesh(mesh, path, color):
        num_vert = np.asarray(mesh.vertices).shape[0]

        colors = np.tile(color, (num_vert,1))
        #colors = np.full(num_vert, color)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_triangle_mesh(filename=path, 
                                   mesh=mesh, 
                                   write_ascii=True,
                                   write_vertex_normals=False,
                                   write_vertex_colors=True,
                                   write_triangle_uvs=False)
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    def __calc_bary_center_volume(CHs):
        CV = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for c in FCCT.color_terms:
            mesh, validity = CHs[c] 
            if not validity:
                b_center_src = (0.0,0.0,0.0)
                vol_src = 0.0
                CV[c] = (b_center_src, vol_src)
                continue

            # calculate gravitational center of convex hull
            # (1) get geometrical center
            coord_center = mesh.get_center()
            #meshw = meshw.translate(-coord_center)
            # (2) iterate over triangles and calculate tetrahaedon mass and center using the coordinate center of the whole mesh
            vol_center = 0
            vertices = np.asarray(mesh.vertices)
            mesh_volume = 0
            for tri in mesh.triangles:
                # calculate center
                pos0 = vertices[tri[0]]
                pos1 = vertices[tri[1]]
                pos2 = vertices[tri[2]]
                pos3 = coord_center
                geo_center = np.sum([pos0, pos1, pos2, pos3], axis=0) / 4
                # calculate volume using the formula:
                # V = |(a-b) * ((b-d) x (c-d))| / 6
                vol = np.abs(np.dot((pos0 - pos3), np.cross((pos1 - pos3), (pos2-pos3)))) / 6
                vol_center += vol * geo_center
                mesh_volume += vol
            # (3) calculate mesh center based on:
            # mass_center = sum(tetra_volumes*tetra_centers)/sum(volumes)
            mass_center = vol_center / mesh_volume
            CV[c] = (mass_center, mesh_volume)
        return CV
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def histogram_matching(src_color, ref_color):
        # [1] Change range from [0.0, 1.0] to [0, 255] and copy source and reference to GPU and create output
        device_src = copy.deepcopy(src_color)
        device_ref = copy.deepcopy(ref_color)
        device_src[:2] = src_color[:2] + 255.0
        device_ref[:2] = ref_color[:2] + 255.0

        m = 1.0
        soft_m = 1.0 / m
        max_range = 900
        stretch = round(math.pow(max_range, soft_m))
        c_range = int(stretch * 2 + 1)

        for t in range(10):
            print(t)
            sci_mat = R.random()#random_state=5)
            mat_rot = sci_mat.as_matrix()
            mat_rot_inv = sci_mat.inv().as_matrix()

             # [2] Create random 3x3 rotation matrix
            mat_rot_tile = np.tile(mat_rot,(src_color.shape[0], 1, 1))
            mat_rot_inv_tile = np.tile(mat_rot_inv,(src_color.shape[0], 1, 1))

            mat_rot_tile_ref = np.tile(mat_rot,(ref_color.shape[0], 1, 1))
            mat_rot_inv_tile_ref = np.tile(mat_rot_inv,(ref_color.shape[0], 1, 1))

            # [3] Rotate source and reference colors with random rotation matrix
            src_rotated = np.einsum('ikl,ik->il', mat_rot_tile, device_src)
            ref_rotated = np.einsum('ikl,ik->il', mat_rot_tile_ref, device_ref)

            # [4] Get 1D marginal
            src_marg_x = src_rotated[:,0]
            src_marg_y = src_rotated[:,1]
            src_marg_z = src_rotated[:,2]
            ref_marg_x = ref_rotated[:,0]
            ref_marg_y = ref_rotated[:,1]
            ref_marg_z = ref_rotated[:,2]

            # [5] Calculate 1D pdf for range [-255, 255] which has to be shifted to [0, 884] (without stretching) in order
            # to allow indexing. The points can be rotated into another octant, therefore the range has to be extended from
            # [0, 255] (256 color values) to [-442, 442] (885 color values). The value 442 was chosen because a color value
            # of (255, 255, 255) can be rotated to (441.7, 0, 0).
            src_cum_marg_x = np.histogram(src_marg_x, bins=c_range, range=(-max_range, max_range), density=True)[0]
            src_cum_marg_y = np.histogram(src_marg_y, bins=c_range, range=(-max_range, max_range), density=True)[0]
            src_cum_marg_z = np.histogram(src_marg_z, bins=c_range, range=(-max_range, max_range), density=True)[0]

            ref_cum_marg_x = np.histogram(ref_marg_x, bins=c_range, range=(-max_range, max_range), density=True)[0]
            ref_cum_marg_y = np.histogram(ref_marg_y, bins=c_range, range=(-max_range, max_range), density=True)[0]
            ref_cum_marg_z = np.histogram(ref_marg_z, bins=c_range, range=(-max_range, max_range), density=True)[0]


            # [6] Calculate cumulative 1D pdf
            src_cum_marg_x = np.cumsum(src_cum_marg_x)
            src_cum_marg_y = np.cumsum(src_cum_marg_y)
            src_cum_marg_z = np.cumsum(src_cum_marg_z)

            ref_cum_marg_x = np.cumsum(ref_cum_marg_x)
            ref_cum_marg_y = np.cumsum(ref_cum_marg_y)
            ref_cum_marg_z = np.cumsum(ref_cum_marg_z)


            # Create LUT
            lut_x = np.zeros(c_range)
            lut_y = np.zeros(c_range)
            lut_z = np.zeros(c_range)

            for i, elem in enumerate(src_cum_marg_x):
                absolute_val_array = np.abs(ref_cum_marg_x - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_x[int(i)] = smallest_difference_index
            for i, elem in enumerate(src_cum_marg_y):
                absolute_val_array = np.abs(ref_cum_marg_y - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_y[int(i)] = smallest_difference_index
            for i, elem in enumerate(src_cum_marg_z):
                absolute_val_array = np.abs(ref_cum_marg_z - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_z[int(i)] = smallest_difference_index

            # Adapt src values
            transferred_rotated_x = lut_x[src_marg_x.astype("int64") + stretch]
            transferred_rotated_y = lut_y[src_marg_y.astype("int64") + stretch]
            transferred_rotated_z = lut_z[src_marg_z.astype("int64") + stretch]
            transferred_rotated = np.concatenate((transferred_rotated_x[:,np.newaxis], transferred_rotated_y[:,np.newaxis]), axis=1)
            transferred_rotated = np.concatenate((transferred_rotated, transferred_rotated_z[:,np.newaxis]), axis=1)

            # [7] Rotate Back
            #transferred_rotated = np.power(transferred_rotated, 1 / soft_m) - stretch
            output = np.einsum('ikl,ik->il', mat_rot_inv_tile, transferred_rotated - stretch)

            # dist_x = np.linalg.norm(transferred_rotated_x - src_rotated[:,0])
            # dist_y = np.linalg.norm(transferred_rotated_y - src_rotated[:,1])
            # dist_z = np.linalg.norm(transferred_rotated_z - src_rotated[:,2])
            # dist = [dist_x, dist_y, dist_z]
            # print(dist)

            output[:2] = output[:2] - 255
            device_src = np.clip(output, -255, 255)

        return device_src.astype("float32")

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    @staticmethod   
    def __sort_by_category(predictions, colors, membership):
        color_cats = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_mem = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_ids = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}

        for i, (pred, color, mem) in enumerate(zip(predictions, colors, membership)):
            color_cats[FCCT.color_terms[int(pred)]].append(color)
            color_cats_mem[FCCT.color_terms[int(pred)]].append(mem)
            color_cats_ids[FCCT.color_terms[int(pred)]].append(i)

        # converst the color lists to arrays
        for col in FCCT.color_terms:
            color_cats[col] = np.asarray(color_cats[col])
            color_cats_mem[col] = np.asarray(color_cats_mem[col])
            color_cats_ids[col] = np.asarray(color_cats_ids[col])

        return color_cats, color_cats_ids, color_cats_mem

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    @staticmethod   
    def __calc_convex_hulls(points):
        CH = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for c in FCCT.color_terms:
            # Check if array has enough points to create convex hull
            if len(points[c]) <= 4:
                CH[c] = (None, False)
                continue

            chull_red_src = ConvexHull(points[c])
            chull_red_src_p = np.expand_dims(chull_red_src.points, axis=1).astype("float32")
            chull_red_src_p = np.squeeze(chull_red_src_p)

            mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(chull_red_src_p),
                                             triangles=o3d.utility.Vector3iVector(chull_red_src.vertices))
            CH[c] = (mesh, True)
        return CH
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod   
    def __calc_membership(votes, distances, num, m):
        epsilon = 1e-5 # prevents division by 0
        dd = 1.0 / (np.power(distances, 2.0/(m-1.0)) + epsilon)
        denominator = np.sum(dd, axis=1)

        class_num = 11
        membership = np.empty((num, 0))
        for c in range(class_num):
            class_votes = (votes == c).astype(int)
            numerator = np.sum(class_votes * dd, axis=1)
            mem_class = numerator / denominator
            membership = np.concatenate((membership, mem_class[:,np.newaxis]), axis=1)
        return membership

        # neigh = FaissKNeighbors(k=1)
        # neigh.fit(colors, labels)

        # with open('/home/potechius/Downloads/LUT.txt', 'w') as f:
        #     arrays = [np.fromiter(range(256), dtype=int), np.fromiter(range(256), dtype=int), np.fromiter(range(256), dtype=int)]
        #     f.write("red green blue label\n")
        #     for res in itertools.product(*arrays):
        #         print(res)
        #         test_c = cv2.cvtColor(np.asarray(res)[np.newaxis, np.newaxis, :].astype("float32"), cv2.COLOR_RGB2Lab)
        #         src_preds = neigh.predict(test_c[:,0,:])
        #         f.write(str(res[0]) + " " + str(res[1]) + " " + str(res[2]) + " " + str(src_preds[0]) + "\n")
        #         #break

        # exit()

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    src = Img(file_path="/media/potechius/Active_Disk/Datasets/ACM-MM-Evaluation-Dataset/interior/256_interior-02.png")
    ref = Img(file_path="/media/potechius/Active_Disk/Datasets/ACM-MM-Evaluation-Dataset/interior/256_interior-03.png")
    #src = Img(file_path="/media/potechius/Active_Disk/SORTING/RES/source.png")
    #ref = Img(file_path="/media/potechius/Active_Disk/SORTING/RES/reference.png")
    #src = Img(file_path="/media/potechius/Active_Disk/SORTING/RES/psource_new.png")
    #ref = Img(file_path="/media/potechius/Active_Disk/SORTING/RES/preference_new.png")
    out = FCCT.apply(src, ref, None)
    out["object"].write("/media/potechius/Active_Disk/SORTING/RES/result_histomatch.png")




FCCT = FuzzyColorCategoryTransfer

if __name__ == "__main__":
    main()

