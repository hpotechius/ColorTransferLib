"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import time
from copy import deepcopy

from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
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
# Info:
#   Name: GlobalColorTransfer
#   Identifier: GLO
#   Link: https://doi.org/10.1109/38.946629
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class GLO:
    compatibility = {
        "src": ["Image", "Mesh", "PointCloud"],
        "ref": ["Image", "Mesh", "PointCloud"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Returns basic information of the corresponding publication.
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "GLO",
            "title": "Color Transfer between Images",
            "year": 2001,
            "abstract": "We use a simple statistical analysis to impose one images color characteristics on another. "
                        "We can achieve color correction by choosing an appropriate source image and apply its "
                        "characteristic to another image.",
            "types": ["Image", "Mesh", "PointCloud"]
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()

        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, GLO.compatibility)

        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output
        
        # Preprocessing
        src_color = src.get_colors()
        ref_color = ref.get_colors()

        out_img = deepcopy(src)

        out_colors = out_img.get_colors()

        # [2] Convert RGB to lab color space
        if opt.colorspace == "lalphabeta":
            lab_src = ColorSpaces.rgb_to_lab_cpu(src_color)
            lab_ref = ColorSpaces.rgb_to_lab_cpu(ref_color)
        elif opt.colorspace == "rgb":
            lab_src = src_color
            lab_ref = ref_color

        # [3] Get mean, standard deviation and ratio of standard deviations
        mean_lab_src = np.mean(lab_src, axis=(0, 1))
        std_lab_src = np.std(lab_src, axis=(0, 1))
        mean_lab_ref = np.mean(lab_ref, axis=(0, 1))
        std_lab_ref = np.std(lab_ref, axis=(0, 1))

        device_div_std = std_lab_ref / std_lab_src

        # [4] Apply Global Color Transfer
        out_colors[:,:,0] = device_div_std[0] * (lab_src[:,:,0] - mean_lab_src[0]) + mean_lab_ref[0]
        out_colors[:,:,1] = device_div_std[1] * (lab_src[:,:,1] - mean_lab_src[1]) + mean_lab_ref[1]
        out_colors[:,:,2] = device_div_std[2] * (lab_src[:,:,2] - mean_lab_src[2]) + mean_lab_ref[2]

        # [5] Convert lab to RGB color space
        if opt.colorspace == "lalphabeta":
            out_colors = ColorSpaces.lab_to_rgb_cpu(out_colors)

        # [6] Clip color to range [0,1]
        out_colors = np.clip(out_colors, 0, 1)

        out_img.set_colors(out_colors)

        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output
