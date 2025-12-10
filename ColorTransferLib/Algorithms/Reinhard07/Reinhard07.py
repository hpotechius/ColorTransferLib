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
from scipy.linalg import fractional_matrix_power
from copy import deepcopy

from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo
from ColorTransferLib.Utils.ColorSpaces import ColorSpaces

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: The Linear Monge-Kantorovitch Linear Colour Mapping forExample-Based Colour Transfer
#   Author: Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley
#   Published in: 4th European Conference on Visual Media Production
#   Year of Publication: 2007
#
# Abstract:
#   A common task in image editing is to change the colours of a picture to match the desired colour grade of another 
#   picture. Finding the correct colour mapping is tricky because it involves numerous interrelated operations, like 
#   balancing the colours, mixing the colour channels or adjusting the contrast. Recently, a number of automated tools 
#   have been proposed to find an adequate one-to-one colour mapping. The focus in this paper is on finding the best 
#   linear colour transformation. Linear transformations have been proposed in the literature but independently. The aim 
#   of this paper is thus to establish a common mathematical background to all these methods. Also, this paper proposes 
#   a novel transformation, which is derived from the Monge-Kantorovitch theory of mass transportation. The proposed 
#   solution is optimal in the sense that it minimises the amount of changes in the picture colours. It favourably 
#   compares theoretically and experimentally with other techniques for various images and under various colour spaces.
#
# Info:
#   Name: MongeKLColorTransfer
#   Identifier: Reinhard07
#   Link: https://doi.org/10.1049/cp:20070055
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Reinhard07:
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
            out_obj = Reinhard07.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = Reinhard07.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = Reinhard07.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = Reinhard07.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = Reinhard07.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = Reinhard07.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = Reinhard07.__apply_mesh(src, ref, opt)
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
        # ensure float32 arrays and flatten to (N,3)
        src_arr = np.asarray(src_color, dtype=np.float32)
        ref_arr = np.asarray(ref_color, dtype=np.float32)

        if src_arr.shape[-1] != 3 or ref_arr.shape[-1] != 3:
            raise ValueError(
                f"__color_transfer expects last dimension = 3, "
                f"got src_shape={src_arr.shape}, ref_shape={ref_arr.shape}"
            )

        src_flat_rgb = src_arr.reshape(-1, 3)  # (N_s,3)
        ref_flat_rgb = ref_arr.reshape(-1, 3)  # (N_r,3)

        # handle degenerate cases: too few pixels -> do nothing
        if src_flat_rgb.shape[0] < 2 or ref_flat_rgb.shape[0] < 2:
            return src_arr

        # choose working colorspace
        if opt.colorspace == "lalphabeta":
            src_flat = ColorSpaces.rgb_to_lalphabeta(src_flat_rgb)
            ref_flat = ColorSpaces.rgb_to_lalphabeta(ref_flat_rgb)
        elif opt.colorspace == "cielab":
            src_flat = ColorSpaces.rgb_to_cielab(src_flat_rgb)
            ref_flat = ColorSpaces.rgb_to_cielab(ref_flat_rgb)
        elif opt.colorspace == "rgb":
            src_flat = src_flat_rgb
            ref_flat = ref_flat_rgb
        else:
            raise ValueError(f"Unsupported colorspace: {opt.colorspace}")

        src_flat = np.asarray(src_flat, dtype=np.float32)
        ref_flat = np.asarray(ref_flat, dtype=np.float32)

        # covariance matrices (3x3) in working colorspace
        src_cov = np.cov(src_flat, rowvar=False)
        ref_cov = np.cov(ref_flat, rowvar=False)

        # regularize to avoid singular matrices
        eps = 1e-6
        src_cov_reg = src_cov + eps * np.eye(3, dtype=np.float32)
        ref_cov_reg = ref_cov + eps * np.eye(3, dtype=np.float32)

        # compute Monge–Kantorovich/Bures mapping in working colorspace
        src_cov_sqrt = fractional_matrix_power(src_cov_reg, 0.5)
        src_cov_inv_sqrt = fractional_matrix_power(src_cov_reg, -0.5)

        inner = src_cov_sqrt @ ref_cov_reg @ src_cov_sqrt
        inner_sqrt = fractional_matrix_power(inner, 0.5)

        T = src_cov_inv_sqrt @ inner_sqrt @ src_cov_inv_sqrt
        T = np.real_if_close(T).astype(np.float32)

        mean_src = np.mean(src_flat, axis=0)
        mean_ref = np.mean(ref_flat, axis=0)

        out_flat_cs = (src_flat - mean_src) @ T + mean_ref
        out_flat_cs = np.real_if_close(out_flat_cs).astype(np.float32)

        # convert back to RGB if needed
        if opt.colorspace == "lalphabeta":
            out_flat_rgb = ColorSpaces.lalphabeta_to_rgb(out_flat_cs)
        elif opt.colorspace == "cielab":
            out_flat_rgb = ColorSpaces.cielab_to_rgb(out_flat_cs)
        else:
            out_flat_rgb = out_flat_cs

        out_flat_rgb = np.asarray(out_flat_rgb, dtype=np.float32)
        out_flat_rgb = np.minimum(np.maximum(out_flat_rgb, 0.0), 1.0)

        out = out_flat_rgb.reshape(src_arr.shape)
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = Reinhard07.__color_transfer(src_color, ref_color, opt)

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

            out_colors = Reinhard07.__color_transfer(src_color, ref_color, opt)

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

            out_colors = Reinhard07.__color_transfer(src_color, ref_color, opt)

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

                out_colors = Reinhard07.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Reinhard07.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Reinhard07.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Reinhard07.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

