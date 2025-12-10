"""
Copyright 2026 by Herbert Potechius,
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
from ColorTransferLib.Utils.ColorSpaces import ColorSpaces  # new

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
#   Identifier: Xiao06
#   Link: https://doi.org/10.1145/1128923.1128974
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Xiao06:
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
            out_obj = Xiao06.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = Xiao06.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = Xiao06.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = Xiao06.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = Xiao06.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = Xiao06.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = Xiao06.__apply_mesh(src, ref, opt)
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
        # ensure float32 arrays
        src_arr = np.asarray(src_color, dtype=np.float32)
        ref_arr = np.asarray(ref_color, dtype=np.float32)

        if src_arr.shape[-1] != 3 or ref_arr.shape[-1] != 3:
            raise ValueError(
                f"__color_transfer expects last dimension = 3, "
                f"got src_shape={src_arr.shape}, ref_shape={ref_arr.shape}"
            )

        # flatten RGB to (N,3)
        flat_src_rgb = src_arr.reshape(-1, 3)
        flat_ref_rgb = ref_arr.reshape(-1, 3)

        num_pts_src = flat_src_rgb.shape[0]
        num_pts_ref = flat_ref_rgb.shape[0]

        # too few points -> nothing to do (return original RGB flatten)
        if num_pts_src < 3 or num_pts_ref < 3:
            return flat_src_rgb

        # ------------------------------------------------------------------
        # choose working colorspace
        # ------------------------------------------------------------------
        if opt.colorspace == "cielab":
            src_flat = ColorSpaces.rgb_to_cielab(flat_src_rgb)
            ref_flat = ColorSpaces.rgb_to_cielab(flat_ref_rgb)
        elif opt.colorspace == "lalphabeta":
            src_flat = ColorSpaces.rgb_to_lalphabeta(flat_src_rgb)
            ref_flat = ColorSpaces.rgb_to_lalphabeta(flat_ref_rgb)
        elif opt.colorspace == "rgb":
            src_flat = flat_src_rgb
            ref_flat = flat_ref_rgb
        else:
            raise ValueError(f"Unsupported colorspace: {opt.colorspace}")

        src_flat = np.asarray(src_flat, dtype=np.float32)
        ref_flat = np.asarray(ref_flat, dtype=np.float32)

        # [1] channel means in working colorspace
        mean_src = np.mean(src_flat, axis=0)
        mean_ref = np.mean(ref_flat, axis=0)

        # [2] covariance matrices in working colorspace
        cov_src = np.cov(src_flat, rowvar=False)
        cov_ref = np.cov(ref_flat, rowvar=False)

        # regularize to avoid singular/ill-conditioned matrices
        eps = 1e-6
        cov_src_reg = cov_src + eps * np.eye(3, dtype=np.float32)
        cov_ref_reg = cov_ref + eps * np.eye(3, dtype=np.float32)

        # [3] SVD of covariance matrices
        U_src, L_src, _ = np.linalg.svd(cov_src_reg)
        U_ref, L_ref, _ = np.linalg.svd(cov_ref_reg)

        # clamp eigenvalues for numerical stability
        L_src_clamped = np.maximum(L_src, eps)
        L_ref_clamped = np.maximum(L_ref, eps)

        # [4] build 4x4 transforms as in Xiao-2006 (in working colorspace)
        T_ref = np.eye(4, dtype=np.float32)
        T_ref[:3, 3] = mean_ref

        R_ref = np.eye(4, dtype=np.float32)
        R_ref[:3, :3] = U_ref

        S_ref = np.array(
            [
                [np.sqrt(L_ref_clamped[0]), 0.0, 0.0, 0.0],
                [0.0, np.sqrt(L_ref_clamped[1]), 0.0, 0.0],
                [0.0, 0.0, np.sqrt(L_ref_clamped[2]), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        T_src = np.eye(4, dtype=np.float32)
        T_src[:3, 3] = -mean_src

        R_src = np.eye(4, dtype=np.float32)
        # U_src is orthogonal -> inverse = transpose
        R_src[:3, :3] = U_src.T

        S_src = np.array(
            [
                [1.0 / np.sqrt(L_src_clamped[0]), 0.0, 0.0, 0.0],
                [0.0, 1.0 / np.sqrt(L_src_clamped[1]), 0.0, 0.0],
                [0.0, 0.0, 1.0 / np.sqrt(L_src_clamped[2]), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # [5] homogeneous coordinates in working colorspace
        ones = np.ones((num_pts_src, 1), dtype=np.float32)
        homogeneous_src = np.hstack((src_flat, ones))  # (N,4)

        # [6] apply full transform: T_ref * R_ref * S_ref * S_src * R_src * T_src
        transformation_matrix = T_ref @ R_ref @ S_ref @ S_src @ R_src @ T_src
        out_h = (transformation_matrix @ homogeneous_src.T).T  # (N,4)

        # result in working colorspace
        out_flat_cs = out_h[:, :3].astype(np.float32)

        # ------------------------------------------------------------------
        # convert back to RGB
        # ------------------------------------------------------------------
        if opt.colorspace == "cielab":
            out_flat_rgb = ColorSpaces.cielab_to_rgb(out_flat_cs)
        elif opt.colorspace == "lalphabeta":
            out_flat_rgb = ColorSpaces.lalphabeta_to_rgb(out_flat_cs)
        else:  # "rgb"
            out_flat_rgb = out_flat_cs

        out_flat_rgb = np.clip(out_flat_rgb, 0.0, 1.0).astype(np.float32)

        # return flattened (N,3) as expected by set_colors
        return out_flat_rgb
  
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = Xiao06.__color_transfer(src_color, ref_color, opt)

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

            out_colors = Xiao06.__color_transfer(src_color, ref_color, opt)

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

            out_colors = Xiao06.__color_transfer(src_color, ref_color, opt)

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

                out_colors = Xiao06.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Xiao06.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Xiao06.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Xiao06.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

