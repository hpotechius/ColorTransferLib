"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import time
from copy import deepcopy
import torch
import cv2

import ColorTransferLib.Algorithms.CAM.color_aware_st as cwst
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: CAMS: Color-Aware Multi-Style Transfer
#   Author: Mahmoud Afifi, Abdullah Abuolaim, Mostafa Hussien, Marcus A. Brubaker, Michael S. Brown
#   Published in: I...
#   Year of Publication: 2021
#
# Abstract:
#   Image style transfer aims to manipulate the appearance of a source image, or "content" image, to share similar
#   texture and colors of a target "style" image. Ideally, the style transfer manipulation should also preserve the
#   semantic content of the source image. A commonly used approach to assist in transferring styles is based on Gram
#   matrix optimization. One problem of Gram matrix-based optimization is that it does not consider the correlation
#   between colors and their styles. Specifically, certain textures or structures should be associated with specific
#   colors. This is particularly challenging when the target style image exhibits multiple style types. In this work,
#   we propose a color-aware multi-style transfer method that generates aesthetically pleasing results while preserving
#   the style-color correlation between style and generated images. We achieve this desired outcome by introducing a
#   simple but efficient modification to classic Gram matrix-based style transfer optimization. A nice feature of our
#   method is that it enables the users to manually select the color associations between the target style and content
#   image for more transfer flexibility. We validated our method with several qualitative comparisons, including a user
#   study conducted with 30 participants. In comparison with prior work, our method is simple, easy to implement, and
#   achieves visually appealing results when targeting images that have multiple styles. Source code is available at
#   this https URL.
#
# Info:
#   Name: CamsTransfer
#   Identifier: CAM
#   Link: https://doi.org/10.48550/arXiv.2106.13920
#   Source: https://github.com/mahmoudnafifi/color-aware-style-transfer
#
# Implementation Details:
#   in ComputeHistogram add small value to prevent division by zero when using images with small color depth
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CAM:
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
            out_obj = CAM.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = CAM.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = CAM.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = CAM.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = CAM.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = CAM.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = CAM.__apply_mesh(src, ref, opt)
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
    def __color_transfer(src_img, ref_img, options):
        if not torch.cuda.is_available():
            options.device = "cpu"

        #ref.resize(src.get_width(), src.get_height())

        # Preprocessing
        src_img = src_img * 255.0
        ref_img = ref_img * 255.0

        ref_img = cv2.resize(ref_img, dsize=(src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_CUBIC)

        out = cwst.apply(src_img, ref_img, options)

        # out_img.set_raw(out, normalized=True)
        # output = {
        #     "status_code": 0,
        #     "response": "",
        #     "object": out_img,
        #     "process_time": time.time() - start_time
        # }

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = CAM.__color_transfer(src_img, ref_img, opt)
        out_img.set_raw(out_colors)

        outp = out_img
        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_video(src, ref, opt): 
        # check if type is video
        out_raw_arr = []
        src_raws = src.get_raw()

        for i, src_raw in enumerate(src_raws):
            # Preprocessing
            ref_raw = ref.get_raw()
            out_img = deepcopy(src.get_images()[0])

            out_colors = CAM.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_raw(out_colors)
            out_raw_arr.append(out_img)

        outp = Video(imgs=out_raw_arr, fps=src.get_fps())

        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_volumetricvideo(src, ref, opt): 
        out_raw_arr = []
        src_raws = src.get_raw()

        for i, src_raw in enumerate(src_raws):
            # Preprocessing
            ref_raw = ref.get_raw()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = CAM.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_raw(out_colors)
            out_raw_arr.append(out_img)
            outp = VolumetricVideo(meshes=out_raw_arr, file_name=src.get_file_name())

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
                src_raw = src_lightfield_array[row][col].get_raw()
                ref_raw = ref.get_raw()

                out_colors = CAM.__color_transfer(src_raw, ref_raw, opt)

                out_lightfield_array[row][col].set_raw(out_colors)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = CAM.__color_transfer(src_img, ref_img, opt)

        out_img.set_raw(out_colors)
        outp = out_img
        return outp


