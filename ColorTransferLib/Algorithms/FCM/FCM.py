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

import csv
import copy
import itertools
import open3d as o3d
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.BaseOptions import BaseOptions
from ColorTransferLib.ImageProcessing.Image import Image as Img
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility
from .FaissKNeighbors import FaissKNeighbors
from .Transform import Transform
from .Export import Export
from .HistogramMatching import HistogramMatching
from .ColorClustering import ColorClustering
from .ColorSpace import ColorSpace

from pyhull.convex_hull import ConvexHull
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: A software test bed for sharing and evaluating color transfer algorithms for images and 3D objects
#   Author: Herbert Potechius, Gunasekaran Raja, Thomas Sikora, Sebastian Knorr
#   Published in: ACM MM
#   Year of Publication: 2023
#
# Abstract:
#   Over the past decades, an overwhelming number of scientific contributions have been published related to the topic 
#   of color transfer where the color statistic of an image is transferred to another image. Recently, this idea was 
#   further extended to 3D point clouds. Due to the fact that the results are normally evaluated subjectively, an 
#   objective comparison of multiple algorithms turns out to be difficult. Therefore, this paper introduces the 
#   ColorTransferLab, a web based test bed which offers a large set of state of the art color transfer implementations. 
#   Furthermore, it allows users to integrate their own implementations with the ultimate goal of providing a library 
#   of state of the art algorithms for the scientific community. This test bed is capable of manipulating both 2D images 
#   and 3D point clouds, and it allows to objectively evaluate and compare color transfer algorithms by providing a 
#   large set of objective metrics. Finally, as part of ColorTransferLab, we propose a new color transfer method, which 
#   integrates concepts of color categories, fuzzy classification, and histogram matching.
#
# Info:
#   Name: FuzzyCategoryTransfer
#   Identifier: FCM
#   Link: ...
#
# Misc:
#   Source: https://arnabfly.github.io/arnab_blog/fknn/
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class FCM:
    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }

    # each color has an id depending on its position in the "color_terms"-variable, i.e.
    # --------------------------------------------------------------------------------------------
    # |  Red  | Yellow | Green | Blue  | Black | White | Grey  | Orange | Brown | Pink  | Purple |
    # --------------------------------------------------------------------------------------------
    # |   0   |   1    |   2   |   3   |   4   |   5   |   6   |   7    |   8   |   9   |   10   |
    # --------------------------------------------------------------------------------------------
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
            "identifier": "FCM",
            "title": "A software test bed for sharing and evaluating color transfer algorithms for images and 3D objects",
            "year": 2023,
            "abstract": "Over the past decades, an overwhelming number of scientific contributions have been published "
                        "related to the topic of color transfer where the color statistic of an image is transferred "
                        "to another image. Recently, this idea was further extended to 3D point clouds. Due to the fact "
                        "that the results are normally evaluated subjectively, an objective comparison of multiple "
                        "algorithms turns out to be difficult. Therefore, this paper introduces the ColorTransferLab, a "
                        "web based test bed which offers a large set of state of the art color transfer "
                        "implementations. Furthermore, it allows users to integrate their own implementations with the "
                        "ultimate goal of providing a library of state of the art algorithms for the scientific "
                        "community. This test bed is capable of manipulating both 2D images and 3D point clouds, and "
                        "it allows to objectively evaluate and compare color transfer algorithms by providing a large "
                        "set of objective metrics. Finally, as part of ColorTransferLab, we propose a new color "
                        "transfer method, which integrates concepts of color categories, fuzzy classification, and "
                        "histogram matching."
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()

        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, FCM.compatibility)

        output = {
            "status_code": 0,
            "response": "",
            "object": None
        }

        src_cpy = copy.deepcopy(src) 
        ref_cpy = copy.deepcopy(ref) 

        if np.unique(src.get_colors(), axis=0).shape[0] <= 16:
            print("SMALL_S")
            
            dst = cv2.GaussianBlur(src_cpy.get_raw(),(7,7),cv2.BORDER_DEFAULT)
            src_cpy.set_raw(dst, normalized=True)

        # if np.unique(ref.get_colors(), axis=0).shape[0] <= 16:
        #     print("SMALL_R")
        #     dstref = cv2.GaussianBlur(ref_cpy.get_raw(),(7,7),cv2.BORDER_DEFAULT)
        #     ref_cpy.set_raw(dstref, normalized=True)

        # --------------------------------------------------------------------------------------------------------------
        # Preprocessing: Convert colorspace from RGB to HSV
        # Ranges: cartesian HSV in [-???:???, -??:??, 0:255]
        # --------------------------------------------------------------------------------------------------------------
        hsv_cart_src = ColorSpace.RGB2cartHSV(src_cpy.get_colors())
        hsv_cart_ref = ColorSpace.RGB2cartHSV(ref_cpy.get_colors())
        rgb_out_orig = deepcopy(src.get_colors()) 
        out_img = deepcopy(src)

        # rgb_src = FCCT.HSV2cartRGB(hsv_cart_src)
        # output_colors = np.clip(rgb_src, 0, 1)
        # out_img.set_colors(output_colors)
        # out_img.write("/home/potechius/Downloads/result.png")
        # exit()

        # VISUALIZATION 00
        # export color pointcloud in HSV
        """
        Export.write_colors_as_PC(hsv_cart_src, src.get_colors()[:,0,:],"/home/potechius/Downloads/FCM_Test/00_src_points.ply")
        Export.write_colors_as_PC(hsv_cart_ref, ref.get_colors()[:,0,:],"/home/potechius/Downloads/FCM_Test/00_ref_points.ply")
        exit()
        """
        # --------------------------------------------------------------------------------------------------------------
        # Read Color Dataset in HSV color space
        # calculates also centers for each category
        # --------------------------------------------------------------------------------------------------------------
        colors, labels, color_cats_db = ColorClustering.get_colormapping_dataset("Models/FCM/colormapping.csv")
        # VISUALIZATION 00
        # color database
        """
        #color_cats_ref = ColorSpace.CAT_cartesian_to_polar(color_cats_db)
        color_cats_ref = ColorSpace.CAT_HSV2cartRGB(color_cats_db)
        for c in FCM.color_terms:
            colorv = FCM.color_samples[c]
            if color_cats_ref[c].shape[0] != 0:
                rep = np.tile(colorv, (color_cats_ref[c].shape[0],1))
                Export.write_colors_as_PC(color_cats_ref[c], rep,"/home/potechius/Downloads/CS/RGB/db_" + c + ".ply")
        exit()
        """
        # --------------------------------------------------------------------------------------------------------------
        # Apply Fuzzy KNN
        # color_cats_src = {"Red": np.array([...]), "Yellow": np.array([...])} 
        # -- Contains pixel colors
        # color_cats_src_ids {"Red": np.array([...]), "Yellow": np.array([...])}
        # -- contains positions within the original color array, i.e. src.get_colors()
        # color_cats_src_mem color_cats_src_ids {"Red": np.array([...]), "Yellow": np.array([...])}
        # -- contains per point 11 values with probabilities
        # --------------------------------------------------------------------------------------------------------------
        color_cats_src, color_cats_src_ids, color_cats_src_mem, color_cats_ref, color_cats_ref_ids, color_cats_ref_mem = \
                                                    ColorClustering.fuzzy_knn(colors, labels, hsv_cart_src, hsv_cart_ref, k=100)

        # VISUALIZATION 00
        # save image with main clustering
        """
        for c in FCCT.color_terms:
            for idx in color_cats_src_ids[c]:
                hsv_cart_src[idx] = FCCT.color_samples[c]
            for idx in color_cats_ref_ids[c]:
                hsv_cart_ref[idx] = FCCT.color_samples[c]

        hsv_cart_src = np.clip(hsv_cart_src, 0, 1)
        out_img.set_colors(hsv_cart_src)
        out_img.write("/home/potechius/Downloads/FCM_Test/src_clustering.png")

        hsv_cart_ref = np.clip(hsv_cart_ref, 0, 1)
        ref.set_colors(hsv_cart_ref)
        ref.write("/home/potechius/Downloads/FCM_Test/ref_clustering.png")
        #exit()
        """

        # VISUALIZATION 01
        # Point Clouds with color coding
        """
        #color_cats_src = ColorSpace.CAT_cartesian_to_polar(color_cats_src)
        #color_cats_ref = ColorSpace.CAT_cartesian_to_polar(color_cats_ref)
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            # check if any points belong to this category
            if color_cats_src[c].shape[0] != 0:
                rep = np.tile(colorv, (color_cats_src[c].shape[0],1))
                Export.write_colors_as_PC(color_cats_src[c], rep,"/home/potechius/Downloads/FCM_Test/01_src_KNN/01_src_points_" + c + ".ply")
            if color_cats_ref[c].shape[0] != 0:
                rep = np.tile(colorv, (color_cats_ref[c].shape[0],1))
                Export.write_colors_as_PC(color_cats_ref[c], rep,"/home/potechius/Downloads/FCM_Test/01_ref_KNN/01_ref_points_" + c + ".ply")
        exit()
        """
        # --------------------------------------------------------------------------------------------------------------
        # Convex Hull Calculation
        # CH_src["Red"] -> (mesh, validity)
        # mesh: the convex hull as triangle mesh
        # validity: True if a convex hull is computed
        # --------------------------------------------------------------------------------------------------------------
        CH_src = ColorClustering.calc_convex_hulls(color_cats_src)
        CH_ref = ColorClustering.calc_convex_hulls(color_cats_ref)
        #CH_db = ColorClustering.calc_convex_hulls(color_cats_db)

        # VISUALIZATION 02
        # Export Convex Hulls
        """
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                Export.write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCM_Test/02_src_CH/02_src_CH_"+c+".ply", 
                                              color=colorv)
            if CH_ref[c][1]:
                Export.write_convex_hull_mesh(mesh=CH_ref[c][0],
                                              path="/home/potechius/Downloads/FCM_Test/02_ref_CH/02_ref_CH_"+c+".ply", 
                                              color=colorv)
        #exit()     
        """
        # --------------------------------------------------------------------------------------------------------------
        # Estimate Transfer Directions based on volume
        # Note: White, Grey and Black will only be transferred to the same color
        # --------------------------------------------------------------------------------------------------------------
        CV_src = ColorClustering.calc_bary_center_volume(CH_src)
        CV_ref = ColorClustering.calc_bary_center_volume(CH_ref)
        #CV_db = ColorClustering.calc_bary_center_volume(CH_db)

        # --------------------------------------------------------------------------------------------------------------
        # Calculate Eigenvectors and -values
        # --------------------------------------------------------------------------------------------------------------
        # EVV_src = ColorClustering.getEigen(CH_src)
        # EVV_ref = ColorClustering.getEigen(CH_ref)

        # --------------------------------------------------------------------------------------------------------------
        # get transfer directions based on number of points
        # NOTE: if one class within a class pair is empty, the class pair has to be removed and the memberships
        # have to be scaled.
        # Example: Memberships: [0.2, 0.2, 0.6] <- sums to onem but 0.6 has no correspondence, i.e. it has to be
        # removed. [0.2, 0.2] --scaling-to-one--> [0.5, 0.5] 
        # --------------------------------------------------------------------------------------------------------------
        #print(color_cats_src["Purple"].shape)

        class_pairs = ColorClustering.get_transfer_direction(CV_src, CV_ref, color_cats_src, color_cats_ref)
        color_cats_src_mem, color_cats_ref_mem = ColorClustering.updateMembership(color_cats_src_mem, color_cats_ref_mem, class_pairs)
        color_cats_src, color_cats_ref, color_cats_src_ids, color_cats_ref_ids, color_cats_src_mem, color_cats_ref_mem = ColorClustering.updateKNN(color_cats_src, color_cats_ref, color_cats_src_ids, color_cats_ref_ids, color_cats_src_mem, color_cats_ref_mem)
        
        # CH_src_new = ColorClustering.calc_convex_hulls(color_cats_src)
        # CV_src_new = ColorClustering.calc_bary_center_volume(CH_src_new)
        # CH_ref_new = ColorClustering.calc_convex_hulls(color_cats_ref)
        # CV_ref_new = ColorClustering.calc_bary_center_volume(CH_ref_new)
        # class_pairs = ColorClustering.updateCenters(class_pairs, CV_src_new, CV_ref_new)

        # VISUALIZATION 00
        # save image with updated clustering
        """
        for c in FCCT.color_terms:
            for idx in color_cats_src_ids[c]:
                hsv_cart_src[idx] = FCCT.color_samples[c]
            for idx in color_cats_ref_ids[c]:
                hsv_cart_ref[idx] = FCCT.color_samples[c]

        hsv_cart_src = np.clip(hsv_cart_src, 0, 1)
        out_img.set_colors(hsv_cart_src)
        out_img.write("/home/potechius/Downloads/FCM_Test/src_clustering_updated.png")

        hsv_cart_ref = np.clip(hsv_cart_ref, 0, 1)
        ref.set_colors(hsv_cart_ref)
        ref.write("/home/potechius/Downloads/FCM_Test/ref_clustering_updated.png")
        exit()
        """
        
        #exit()
        # --------------------------------------------------------------------------------------------------------------
        # Print Transfer Directions and volumes
        # --------------------------------------------------------------------------------------------------------------
        for elem in class_pairs:
            print(elem[0][0] + " - " + elem[1][0])
            print(str(elem[0][1]) + " - " + str(elem[1][1]))
        print("-----------------------------------------")
        # --------------------------------------------------------------------------------------------------------------
        # Apply Rotation and Translation
        # --------------------------------------------------------------------------------------------------------------
        """
        rotation_matrix = Transform.get_rotation_matrix(class_pairs)
        """

        CH_src_new = ColorClustering.calc_convex_hulls(color_cats_src)
        CV_src_new = ColorClustering.calc_bary_center_volume(CH_src_new)
        CH_ref_new = ColorClustering.calc_convex_hulls(color_cats_ref)
        CV_ref_new = ColorClustering.calc_bary_center_volume(CH_ref_new)
        rotation_angles = Transform.get_rotation_angles(class_pairs, CV_src_new, CV_ref_new)

        #hue_matrix = Transform.get_hue_translation(class_pairs, CV_db, color_cats_db)
        # color_cats_src = FCCT.rotation(class_pairs, color_cats_src)

        #color_cats_src = Transform.transform(ColorSpace.CAT_cartesian_to_polar(color_cats_src), hue_matrix)
        #color_cats_src = Transform.transform_weighted(ColorSpace.CAT_cartesian_to_polar(color_cats_src), color_cats_src_mem, hue_matrix)
        #color_cats_src = ColorSpace.CAT_polar_to_cartesian(color_cats_src)


        # --------------------------------------------------------------------------------------------------------------
        # ReCalculate center and volume of rotated source
        # --------------------------------------------------------------------------------------------------------------
        """
        CH_src_new = ColorClustering.calc_convex_hulls(color_cats_src)
        CV_src_new = ColorClustering.calc_bary_center_volume(CH_src_new)
        """
        # --------------------------------------------------------------------------------------------------------------
        # Apply Translation
        # --------------------------------------------------------------------------------------------------------------
        """
        color_cats_src_temp = Transform.transform(color_cats_src, rotation_matrix)
        CH_src_new = ColorClustering.calc_convex_hulls(color_cats_src_temp)
        CV_src_new = ColorClustering.calc_bary_center_volume(CH_src_new)
        """

        translation_sat, translation_val = Transform.get_translation_vec(class_pairs)
        # --------------------------------------------------------------------------------------------------------------
        #color_cats_src = Transform.translation(class_pairs, color_cats_src, CV_src_new)
        # --------------------------------------------------------------------------------------------------------------
        """
        translation_matrix = Transform.get_translation_matrix(class_pairs, CV_src_new)
        """
        # --------------------------------------------------------------------------------------------------------------
        # Scaling
        # --------------------------------------------------------------------------------------------------------------
        """
        #color_cats_src = FCCT.scaling(class_pairs, color_cats_src, color_cats_ref)
        color_cats_src_temp = Transform.transform(color_cats_src_temp, translation_matrix)  
        scaling_matrix = Transform.get_scaling_matrix(class_pairs, color_cats_src_temp, color_cats_ref)  
        """

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
        #color_cats_src_temp = Transform.transform(ColorSpace.CAT_cartesian_to_polar(color_cats_src), hue_matrix)
        #color_cats_ref_temp = ColorSpace.CAT_cartesian_to_polar(color_cats_ref)
        #scaling_matrix = Transform.get_hue_scaling(class_pairs, color_cats_src_temp, color_cats_ref_temp)

        #color_cats_src = Transform.scaling(class_pairs, color_cats_src_temp, color_cats_ref)

        #color_cats_src_temp = Transform.transform(color_cats_src_temp, scaling_matrix)



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

        # --------------------------------------------------------------------------------------------------------------
        # get the 11 transformation matrices
        # --------------------------------------------------------------------------------------------------------------
        """
        affine_transform = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for c in FCCT.color_terms:
            #affine_transform[c] = np.eye(4)
            #affine_transform[c] = scaling_matrix[c] @ hue_matrix[c]
            #affine_transform[c] = hue_matrix[c]
            affine_transform[c] = rotation_matrix[c]
            #affine_transform[c] = translation_matrix[c] @ rotation_matrix[c]
            #affine_transform[c] = scaling_matrix[c] @ rotation_matrix[c]
            #affine_transform[c] = scaling_matrix[c] @ translation_matrix[c] @ rotation_matrix[c]
        """

        # --------------------------------------------------------------------------------------------------------------
        # readjust weights so category with highest probability has more impact 
        # Original: [0.5, 0.3, 0.2] -> After adjustment: [0.75, 0.15, 0.1]
        # Note: The highest value is always scaled by log_2(x+1)^scale_factor 
        # Note: Scale factor has to in the range [0.0, 1.0] -> 0.1 describes a strong scaling
        # --------------------------------------------------------------------------------------------------------------
        """
        scale_factor = 0.3
        for c in FCCT.color_terms:
            continue
            if color_cats_src_mem[c].shape[0] == 0:
                continue

            # print(color_cats_src_mem[c][0])
            # print(np.sum(color_cats_src_mem[c][0]))
            # print("\n")

            max_mem = np.max(color_cats_src_mem[c], axis=1)
            max_pos = color_cats_src_mem[c].argmax(axis=1)
            
            new_max_mem = np.log(max_mem + 1) ** scale_factor

            # the value will downscale all the other probabilities to get a total prob. of 1
            # Eps is for stabilization
            eps = 0.00001
            rest_probs = (np.sum(color_cats_src_mem[c], axis=1) - max_mem + eps) / (1.0 - new_max_mem + eps)
            rest_probs = np.repeat(rest_probs[:, np.newaxis], 11, axis=1)
            color_cats_src_mem[c] /= rest_probs

            # print(color_cats_src_mem[c][0])
            # print(np.sum(color_cats_src_mem[c][0]))
            # print("\n")


            # upscale max value
            for i in range(color_cats_src_mem[c].shape[0]):
                color_cats_src_mem[c][i][max_pos[i]] = new_max_mem[i]


            # print(color_cats_src_mem[c][0])
            # print(np.sum(color_cats_src_mem[c][0]))
            # print("\n")

            #print(max_mem[0])
            #print(max_pos[0])
           # print(new_max_mem[0])
            #exit()
        """

        # VISUALIZATION 03
        # recategorized points as convex hulls
        """
        CH_src = ColorClustering.calc_convex_hulls(color_cats_src)
        CH_ref = ColorClustering.calc_convex_hulls(color_cats_ref)
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                Export.write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCM_Test/03_src_CH_reassigned/03_src_CH_reassigned_"+c+".ply", 
                                              color=colorv)
            if CH_ref[c][1]:
                Export.write_convex_hull_mesh(mesh=CH_ref[c][0],
                                              path="/home/potechius/Downloads/FCM_Test/03_ref_CH_reassigned/03_ref_CH_reassigned_"+c+".ply", 
                                              color=colorv)
        """

        color_cats_src = Transform.transform_weighted_rotation(color_cats_src, color_cats_src_mem, rotation_angles)
        color_cats_src = Transform.transform_weighted_translation(color_cats_src, color_cats_src_mem, translation_sat, 1)
        #color_cats_src = Transform.transform_weighted_translation(color_cats_src, color_cats_src_mem, translation_val, 2)

        # color_cats_src = Transform.transform_rotation(color_cats_src, color_cats_src_mem, rotation_angles)
        # color_cats_src = Transform.transform_translation(color_cats_src, color_cats_src_mem, translation_sat, 1)
        # color_cats_src = Transform.transform_translation(color_cats_src, color_cats_src_mem, translation_val, 2)

        CH_src = ColorClustering.calc_convex_hulls(color_cats_src)
        CV_src = ColorClustering.calc_bary_center_volume(CH_src)
        CH_ref = ColorClustering.calc_convex_hulls(color_cats_ref)
        CV_ref = ColorClustering.calc_bary_center_volume(CH_ref)
        scaling_matrix = Transform.get_scaling_matrix(class_pairs, color_cats_src, color_cats_ref, CV_src, CV_ref)  
        #color_cats_src = Transform.transform(color_cats_src, scaling_matrix, cartesian=False)  
        color_cats_src = Transform.transform_weighted(color_cats_src, color_cats_src_mem, scaling_matrix, cartesian=False)

        # VISUALIZATION 04
        """
        CH_src = ColorClustering.calc_convex_hulls(color_cats_src)
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                Export.write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCM_Test/04_src_CH_scaled/04_src_CH_scaled_"+c+".ply", 
                                              color=colorv)
        #exit()
        """

        # NOTE: weighted transformation in polar coordinates not possible because the FKNN was done in cartesian coordinates
        #color_cats_src_pol = ColorSpace.CAT_cartesian_to_polar(color_cats_src)
        #color_cats_src_ttt = Transform.transform(color_cats_src_pol, affine_transform)
        #color_cats_src_ttt = Transform.transform_weighted(color_cats_src_pol, color_cats_src_mem, affine_transform)
        #color_cats_src = ColorSpace.CAT_polar_to_cartesian(color_cats_src_ttt)
        
        
        # for c in FCCT.color_terms:
        #     if color_cats_src[c].shape[0] == 0:
        #         continue
        #     color_cats_src_pol[c][:,0:] = color_cats_src_ttt[c][:,0:]
        
       

        
        # VISUALIZATION 04
        # rotated
        """
        CH_src = ColorClustering.calc_convex_hulls(color_cats_src)
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                Export.write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCM_Test/04_src_CH_rotated/04_src_CH_rotated_"+c+".ply", 
                                              color=colorv)
        """
        # translation saturation
        """
        CH_src = ColorClustering.calc_convex_hulls(color_cats_src)
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                Export.write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCM_Test/04_src_CH_translated_sat/04_src_CH_translated_sat_"+c+".ply", 
                                              color=colorv)
        """
        # translation value
        """
        CH_src = ColorClustering.calc_convex_hulls(color_cats_src)
        for c in FCCT.color_terms:
            colorv = FCCT.color_samples[c]
            if CH_src[c][1]:
                Export.write_convex_hull_mesh(mesh=CH_src[c][0],
                                              path="/home/potechius/Downloads/FCM_Test/04_src_CH_translated_val/04_src_CH_translated_val_"+c+".ply", 
                                              color=colorv)
        """
        # for c in FCCT.color_terms:
        #     for color, idx in zip(color_cats_src[c], color_cats_src_ids[c]):
        #         #hsv_cart_src[idx] = color
        #         # reduce HSV to valid ranges
        #         hsv_cart_src[idx,0] = (359 + color[0]) % 359
        #         hsv_cart_src[idx,1] = np.clip(color[1], 0, 255)
        #         hsv_cart_src[idx,2] = np.clip(color[2], 0, 255)

        # #hsv_cart_src = ColorSpace.polar_to_cartesian(hsv_cart_src)

        # rgb_out = ColorSpace.HSV2cartRGB(hsv_cart_src)
        # output_colors = np.clip(rgb_out, 0, 1)
        # out_img.set_colors(output_colors)

        # out_img.write("/home/potechius/Downloads/FCCT_Tests/result_translation_weighted.png")
        #exit()

        #color_cats_src = Transform.transform_weighted(color_cats_src, color_cats_src_mem, affine_transform)
        #color_cats_src = Transform.transform(color_cats_src, affine_transform)

        # --------------------------------------------------------------------------------------------------------------
        # Histogram Matching per category
        # --------------------------------------------------------------------------------------------------------------

        """
        for elem_src, elem_ref in class_pairs:
            if color_cats_src[elem_src[0]].shape[0] <= 3 or color_cats_ref[elem_ref[0]].shape[0] <= 3:
                continue
            #color_cats_src[elem_src[0]] = HistogramMatching.histogram_matching(color_cats_src[elem_src[0]], color_cats_ref[elem_ref[0]], 10)
        """

        #color_cats_src = FCCT.applyTransformation(class_pairs, color_cats_src, rotation_matrix, translation_matrix, scaling_matrix)
        
        #-------------------------------------------

        # write mesh

        #hsv_cart_src_te = copy.deepcopy(hsv_cart_src)
        for c in FCM.color_terms:
            for color, idx in zip(color_cats_src[c], color_cats_src_ids[c]):
                hsv_cart_src[idx] = color

        # color_cats_src, color_cats_src_ids, color_cats_src_mem, _, _, _ = ColorClustering.fuzzy_knn(colors, labels, hsv_cart_src, hsv_cart_ref)

        # for elem_src, elem_ref in class_pairs:
        #     if color_cats_src[elem_src[0]].shape[0] <= 3 or color_cats_ref[elem_ref[0]].shape[0] <= 3:
        #         continue
        #     color_cats_src[elem_src[0]] = HistogramMatching.histogram_matching(color_cats_src[elem_src[0]], color_cats_ref[elem_ref[0]], 10)

        # for c in FCCT.color_terms:
        #     for color, idx in zip(color_cats_src[c], color_cats_src_ids[c]):
        #         hsv_cart_src[idx] = color

        # TEMP
        # ttt = hsv_cart_src_te - hsv_cart_src
        # print(np.min(ttt[:,0]))
        # print(np.max(ttt[:,0]))
        # print(np.min(ttt[:,1]))
        # print(np.max(ttt[:,1]))
        # print(np.min(ttt[:,2]))
        # print(np.max(ttt[:,2]))

        # --------------------------------------------------------------------------------------------------------------
        # Histogram Matching
        # --------------------------------------------------------------------------------------------------------------    
        hsv_cart_src[:,0] = np.clip(hsv_cart_src[:,0], 0, 255)    
        hsv_cart_src[:,1] = np.clip(hsv_cart_src[:,1], 0, 255)    
        hsv_cart_src[:,2] = np.clip(hsv_cart_src[:,2], 0, 255)    
        # print(np.min(hsv_cart_src[:,0]))
        # print(np.max(hsv_cart_src[:,0]))
        # print(np.min(hsv_cart_src[:,1]))
        # print(np.max(hsv_cart_src[:,1]))
        # print(np.min(hsv_cart_src[:,2]))
        # print(np.max(hsv_cart_src[:,2]))
        
        #hsv_cart_src = HistogramMatching.histogram_matching(hsv_cart_src, hsv_cart_ref, 10)
        #hsv_cart_src = HistogramMatching.histogram_matching(ColorSpace.cartesian_to_polar(hsv_cart_src), ColorSpace.cartesian_to_polar(hsv_cart_ref), 10)
        #hsv_cart_src = ColorSpace.polar_to_cartesian(hsv_cart_src)


        rgb_out = ColorSpace.HSV2cartRGB(hsv_cart_src)

        #rgb_out = np.expand_dims(HistogramMatching.histogram_matching(rgb_out[:,0,:]*255, ref.get_colors()[:,0,:]*255, 10), 1) / 255
        rgb_out = np.expand_dims(HistogramMatching.histogram_matching2(rgb_out[:,0,:], ref.get_colors()[:,0,:], 10), 1)
        
        
        """
        FCCT.write_colors_as_PC(hsv_cart_src, rgb_out[:,0,:], "/home/potechius/Downloads/FCCT_Tests/00_out_points_2.ply")
        """

        # --------------------------------------------------------------------------------------------------------------
        # Regraining
        # --------------------------------------------------------------------------------------------------------------
        out_res = rgb_out
        out_res = HistogramMatching.regrain(src_cpy, rgb_out, 100)
        
        output_colors = np.clip(out_res, 0, 1)
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
    @staticmethod   
    def add_gaussian_noise(image):
        row,col,ch= image.shape
        mean = 0
        var = 0.00001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
        
    # ------------------------------------------------------------------------------------------------------------------
    # returns the 3D color histogram
    # ------------------------------------------------------------------------------------------------------------------
    def get_color_statistic_3D(img, bins=[256,256,256], normalized=False):
        color = img.get_colors()
        rgb_c = (color * 255.0).astype(np.int).reshape(color.shape[0], color.shape[2])
        histo = np.asarray(np.histogramdd(rgb_c, bins)[0])

        if normalized:
            sum_h = np.sum(histo)
            histo /= sum_h
        return histo
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def histogramintersection(src, ref, bins=[10,10,10]):
        histo1 = FCM.get_color_statistic_3D(src, bins=bins, normalized=True)
        histo2 = FCM.get_color_statistic_3D(ref, bins=bins, normalized=True)
        minimum = np.minimum(histo1, histo2)
        intersection = np.sum(minimum)
        #intersection = np.true_divide(np.sum(minima), np.sum(histo2))
        return round(intersection, 4)


