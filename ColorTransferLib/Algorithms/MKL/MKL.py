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
from scipy.linalg import fractional_matrix_power
from copy import deepcopy

from ColorTransferLib.Utils.Helper import check_compatibility

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
#   Identifier: MKL
#   Link: https://doi.org/10.1049/cp:20070055
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class MKL:
    compatibility = {
        "src": ["Image", "Mesh", "PointCloud"],
        "ref": ["Image", "Mesh", "PointCloud"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "MKL",
            "title": "The Linear Monge-Kantorovitch Linear Colour Mapping forExample-Based Colour Transfer",
            "year": 2007,
            "abstract": "A common task in image editing is to change the colours of a picture to match the desired " 
                        "colour grade of another picture. Finding the correct colour mapping is tricky because it "
                        "involves numerous interrelated operations, like balancing the colours, mixing the colour "
                        "channels or adjusting the contrast. Recently, a number of automated tools have been proposed "
                        "to find an adequate one-to-one colour mapping. The focus in this paper is on finding the best "
                        "linear colour transformation. Linear transformations have been proposed in the literature but "
                        "independently. The aim of this paper is thus to establish a common mathematical background to "
                        "all these methods. Also, this paper proposes a novel transformation, which is derived from "
                        "the Monge-Kantorovitch theory of mass transportation. The proposed solution is optimal in the "
                        "sense that it minimises the amount of changes in the picture colours. It favourably compares "
                        "theoretically and experimentally with other techniques for various images and under various "
                        "colour spaces.",
            "types": ["Image", "Mesh", "PointCloud"]
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, MKL.compatibility)
    
        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output

        # Preprocessing
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        src_color_rgb = src_color.reshape(src_color.shape[0], 3)
        ref_color_rgb = ref_color.reshape(ref_color.shape[0], 3)

        src_cov = np.cov(src_color_rgb, rowvar=False)
        ref_cov = np.cov(ref_color_rgb, rowvar=False)

        src_covs = fractional_matrix_power(src_cov, 0.5)
        src_covsr = fractional_matrix_power(src_cov, -0.5)

        T = np.dot(src_covsr, np.dot(fractional_matrix_power(np.dot(src_covs, np.dot(ref_cov, src_covs)), 0.5), src_covsr))

        mean_src = np.mean(src_color_rgb, axis=0)
        mean_ref = np.mean(ref_color_rgb, axis=0)
        out = (src_color_rgb - mean_src) @ T + mean_ref
        out = np.reshape(out, src_color.shape)
        out = np.real(out)
        out = np.clip(out, 0, 1)

        out_img.set_colors(out)
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output
