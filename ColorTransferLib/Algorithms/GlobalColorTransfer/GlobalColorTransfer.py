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
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.BaseOptions import BaseOptions
from ColorTransferLib.ImageProcessing.Image import Image as Img
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility


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
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class GlobalColorTransfer:
    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # HOST METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "GlobalColorTransfer",
            "title": "Color Transfer between Images",
            "year": 2001,
            "abstract": "We use a simple statistical analysis to impose one images color characteristics on another. "
                        "We can achieve color correction by choosing an appropriate source image and apply its "
                        "characteristic to another image."
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, GlobalColorTransfer.compatibility)

        # Preprocessing
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        # [1] Copy source and reference to GPU and create output
        device_src = cuda.to_device(src_color)
        device_ref = cuda.to_device(ref_color)
        device_out = cuda.device_array(src_color.shape)

        # [2] Convert RGB to lab color space
        if opt.colorspace == "lalphabeta":
            lab_src = ColorSpaces.rgb_to_lab(device_src)
            lab_ref = ColorSpaces.rgb_to_lab(device_ref)
        elif opt.colorspace == "rgb":
            lab_src = device_src
            lab_ref = device_ref

        # [3] Get mean, standard deviation and ratio of standard deviations
        mean_lab_src = cuda.to_device(np.mean(lab_src, axis=(0, 1)))
        std_lab_src = np.std(lab_src, axis=(0, 1))
        mean_lab_ref = cuda.to_device(np.mean(lab_ref, axis=(0, 1)))
        std_lab_ref = np.std(lab_ref, axis=(0, 1))

        device_div_std = cuda.to_device(std_lab_ref / std_lab_src)

        # [4] Apply Global Color Transfer on GPU
        threadsperblock = (32, 32)
        blockspergrid_x = int(math.ceil(device_out.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(device_out.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        GlobalColorTransfer.__kernel_apply[blockspergrid, threadsperblock](lab_src,
                                                                           device_out,
                                                                           mean_lab_src,
                                                                           mean_lab_ref,
                                                                           device_div_std)

        # [5] Convert lab to RGB color space
        if opt.colorspace == "lalphabeta":
            device_out = ColorSpaces.lab_to_rgb(device_out)

        # [6] Copy output from GPU to RAM
        out_colors = device_out.copy_to_host()
        out_colors = np.clip(out_colors, 0, 1)

        out_img.set_colors(out_colors)

        output = {
            "status_code": 0,
            "response": "",
            "object": out_img
        }

        return output

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # DEVICE METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # method description
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    @cuda.jit
    def __kernel_apply(in_mat, out_mat, mean_lab_src, mean_lab_ref, div_std):
        pos = cuda.grid(2)
        x = pos[1] % in_mat.shape[1]
        y = pos[0] % in_mat.shape[0]

        out_mat[y, x, 0] = div_std[0] * (in_mat[y, x, 0] - mean_lab_src[0]) + mean_lab_ref[0]
        out_mat[y, x, 1] = div_std[1] * (in_mat[y, x, 1] - mean_lab_src[1]) + mean_lab_ref[1]
        out_mat[y, x, 2] = div_std[2] * (in_mat[y, x, 2] - mean_lab_src[2]) + mean_lab_ref[2]

