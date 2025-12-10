"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import time
from copy import deepcopy
import sys
import os

from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo
from ColorTransferLib.Utils.Helper import init_model_files

from .third_party.test import predict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: StyTr2: Image Style Transfer with Transformers
#   Author: Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, Changsheng Xu
#   Published in: CVPR
#   Year of Publication: 2022
#
# Info:
#   Name: StyTr2
#   Identifier: Deng22
#   Link: https://doi.org/10.48550/arXiv.2105.14576
#
# Source:
#   https://github.com/diyiiyiii/StyTR-2
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Deng22:
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

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField" or ref.get_type() == "GaussianSplatting" or ref.get_type() == "PointCloud":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = Deng22.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = Deng22.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = Deng22.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = Deng22.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = Deng22.__apply_mesh(src, ref, opt)
        else:
            out_obj = None
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_img, ref_img, opt):
        model_file_paths = init_model_files("Deng22", ["decoder_iter_160000.pth", "embedding_iter_160000.pth", "transformer_iter_160000.pth", "vgg_normalised.pth"])

        # suppress output
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull

        out_img = predict(model_file_paths, src_img, ref_img, opt)

        sys.stdout = old_stdout
        devnull.close()

        return out_img

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = Deng22.__color_transfer(src_img, ref_img, opt)
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

            out_colors = Deng22.__color_transfer(src_raw, ref_raw, opt)

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

            out_colors = Deng22.__color_transfer(src_raw, ref_raw, opt)

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

                out_colors = Deng22.__color_transfer(src_raw, ref_raw, opt)

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

        out_colors = Deng22.__color_transfer(src_img, ref_img, opt)


        out_img.set_raw(out_colors)
        outp = out_img
        return outp


