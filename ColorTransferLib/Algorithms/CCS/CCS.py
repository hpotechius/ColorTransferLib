"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import time
from copy import deepcopy

from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Color Transfer in Correlated Color Space
#   Author: Xuezhong Xiao, Lizhuang Ma
#   Published in: Proceedings of the 2006 ACM international conference on Virtual reality continuum and its applications
#   Year of Publication: 2006
#
# Abstract:
#   In this paper we present a process called color transfer which can borrow one image's color characteristics from 
#   another. Recently Reinhard and his colleagues reported a pioneering work of color transfer. Their technology can 
#   produce very believable results, but has to transform pixel values from RGB to lab . Inspired by their work, we 
#   advise an approach which can directly deal with the color transfer in any 3D space. From the view of statistics, 
#   we consider pixel's value as a threedimension stochastic variable and an image as a set of samples, so the 
#   correlations between three components can be measured by covariance. Our method imports covariance between three 
#   components of pixel values while calculate the mean along each of the three axes. Then we decompose the covariance 
#   matrix using SVD algorithm and get a rotation matrix. Finally we can scale, rotate and shift pixel data of target 
#   image to fit data points' cluster of source image in the current color space and get resultant image which takes on 
#   source image's look and feel. Besides the global processing, a swatch-based method is introduced in order to 
#   manipulate images' color more elaborately. Experimental results confirm the validity and usefulness of our method.
#
# Info:
#   Name: CorrelatedColorSpaceTransfer
#   Identifier: CSS
#   Link: https://doi.org/10.1145/1128923.1128974
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CCS:
    # ------------------------------------------------------------------------------------------------------------------
    # Checks source and reference compatibility
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        output = {
            "status_code": 0,
            "response": "",
            "object": None,
            "process_time": 0
        }

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = CCS.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = CCS.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = CCS.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = CCS.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = CCS.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = CCS.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = CCS.__apply_mesh(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_color, ref_color, opt):

        reshaped_src = np.reshape(src_color, (-1,3))
        reshaped_ref = np.reshape(ref_color, (-1,3))

        num_pts_src = reshaped_src.shape[0]
        num_pts_ref = reshaped_ref.shape[0]

        # [] Calculate mean of each channel (for src and ref)
        mean_src = np.mean(src_color, axis=(0, 1))
        mean_ref = np.mean(ref_color, axis=(0, 1))

        # [] Calculate covariance matrix between the three components (for src and ref)
        cov_src = np.cov(reshaped_src, rowvar=False)
        cov_ref = np.cov(reshaped_ref, rowvar=False)

        # [] SVD of covariance matrices
        U_src, L_src, _ = np.linalg.svd(cov_src)
        U_ref, L_ref, _ = np.linalg.svd(cov_ref)
        
        T_ref = np.eye(4)
        T_ref[:3,3] = mean_ref

        R_ref = np.eye(4)
        R_ref[:3,:3] = U_ref

        S_ref = np.array([[np.sqrt(L_ref[0]), 0, 0, 0],
                          [0, np.sqrt(L_ref[1]), 0, 0],
                          [0, 0, np.sqrt(L_ref[2]), 0],
                          [0, 0, 0, 1]])
        
        T_src = np.eye(4)
        T_src[:3,3] = -mean_src
        
        R_src = np.eye(4)
        R_src[:3,:3] = np.linalg.inv(U_src)
        
        S_src = np.array([[1/np.sqrt(L_src[0]), 0, 0, 0],
                          [0, 1/np.sqrt(L_src[1]), 0, 0],
                          [0, 0, 1/np.sqrt(L_src[2]), 0],
                          [0, 0, 0, 1]])
        
        # [] turn euclidean points into homogeneous points
        ones = np.ones((num_pts_src, 1))
        homogeneous_src = np.hstack((reshaped_src, ones))

        # [] Apply Transformation: out = T_ref * R_ref * S_ref * S_src * R_src * T_src * src
        transformation_matrix = T_ref @ R_ref @ S_ref @ S_src @ R_src @ T_src
        out = (transformation_matrix @ homogeneous_src.T).T

        # turn homogeneous points into euclidean points 
        out_colors = out[:,:3]
        out_colors = np.clip(out_colors, 0, 1)

        return out_colors
  
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = CCS.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_video(src, ref, opt): 
        # check if type is video
        out_colors_arr = []
        src_colors = src.get_colors()

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_images()[0])

            out_colors = CCS.__color_transfer(src_color, ref_color, opt)

            out_img.set_colors(out_colors)
            out_colors_arr.append(out_img)

        outp = Video(imgs=out_colors_arr, fps=src.get_fps())

        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_volumetricvideo(src, ref, opt): 
        out_colors_arr = []
        src_colors = src.get_colors()

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = CCS.__color_transfer(src_color, ref_color, opt)

            out_img.set_colors(out_colors)
            out_colors_arr.append(out_img)
            outp = VolumetricVideo(meshes=out_colors_arr, file_name=src.get_file_name())

        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_lightfield(src, ref, opt):
        src_lightfield_array = src.get_image_array()
        out = deepcopy(src)
        out_lightfield_array = out.get_image_array()

        for row in range(src.get_grid_size()[0]):
            for col in range(src.get_grid_size()[1]):
                src_color = src_lightfield_array[row][col].get_colors()
                ref_color = ref.get_colors()

                out_colors = CCS.__color_transfer(src_color, ref_color, opt)

                out_lightfield_array[row][col].set_colors(out_colors)

        return out
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_gaussiansplatting(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = CCS.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_pointcloud(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = CCS.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = CCS.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

