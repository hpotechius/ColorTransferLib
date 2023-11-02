"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import math
import time
from copy import deepcopy
import numpy as np

from ColorTransferLib.Utils.Helper import check_compatibility

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: N-dimensional probability density function transfer and its application to color transfer
#   Author: Francois Pitie, Anil C. Kokaram, Rozenn Dahyot
#   Published in: Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1
#   Year of Publication: 2005
#
# Abstract:
#   This article proposes an original method to estimate a continuous transformation that maps one N-dimensional
#   distribution to another. The method is iterative, non-linear, and is shown to converge. Only 1D marginal
#   distribution is used in the estimation process, hence involving low computation costs. As an illustration this
#   mapping is applied to color transfer between two images of different contents. The paper also serves as a central
#   focal point for collecting together the research activity in this area and relating it to the important problem of
#   automated color grading.
#
# Info:
#   Name: PdfColorTransfer
#   Identifier: PDF
#   Link: https://doi.org/10.1109/ICCV.2005.166
#
# Implementation Details:
#   m = 1
#   iterations = 20
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PDF:
    compatibility = {
        "src": ["Image", "Mesh", "PointCloud"],
        "ref": ["Image", "Mesh", "PointCloud"]
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
            "identifier": "PDF",
            "title": "N-dimensional probability density function transfer and its application to color transfer",
            "year": 2005,
            "abstract": "This article proposes an original method to estimate a continuous transformation that maps "
                        "one N-dimensional distribution to another. The method is iterative, non-linear, and is shown "
                        "to converge. Only 1D marginal distribution is used in the estimation process, hence involving "
                        "low computation costs. As an illustration this mapping is applied to color transfer between "
                        "two images of different contents. The paper also serves as a central focal point for "
                        "collecting together the research activity in this area and relating it to the important "
                        "problem of automated color grading.",
            "types": ["Image", "Mesh", "PointCloud"]
        }

        return info
    
    # ------------------------------------------------------------------------------------------------------------------
    # Generate a random 3x3 rotation matrix
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def random_rotation_matrix():
        random_state = np.random.default_rng()
        H = np.eye(3) + random_state.standard_normal((3, 3))
        Q, R = np.linalg.qr(H)
        return Q

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, PDF.compatibility)

        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output

        output = {
            "status_code": 0,
            "response": "",
            "object": None
        }

        # Preprocessing
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        # Change range from [0.0, 1.0] to [0, 255]
        src_color = src_color.squeeze() * 255.0
        ref_color = ref_color.squeeze() * 255.0

        m = 1.0
        max_range = 442
        stretch = round(math.pow(max_range, 1.0 / m))
        c_range = int(stretch * 2 + 1)

        for t in range(opt.iterations):
            mat_rot = PDF.random_rotation_matrix()
            mat_rot_inv = np.linalg.inv(mat_rot)

            src_rotated = np.einsum('ij,kj->ki', mat_rot, src_color)
            ref_rotated = np.einsum('ij,kj->ki', mat_rot, ref_color)

            # Calculate 1D pdf
            src_marginals = [np.histogram(src_rotated[:, i], bins=c_range, range=(-max_range, max_range), density=True)[0] for i in range(3)]
            ref_marginals = [np.histogram(ref_rotated[:, i], bins=c_range, range=(-max_range, max_range), density=True)[0] for i in range(3)]

            # Calculate cumulative 1D pdf
            src_cum_marginals = [np.cumsum(marg) for marg in src_marginals]
            ref_cum_marginals = [np.cumsum(marg) for marg in ref_marginals]

            lut = []
            for src_marg, ref_marg in zip(src_cum_marginals, ref_cum_marginals):
                lut_channel = np.zeros(c_range)
                for i, elem in enumerate(src_marg):
                    lut_channel[i] = np.abs(ref_marg - elem).argmin()
                lut.append(lut_channel)

            src_rotated_marginals = [(src_rotated[:, i].astype("int64") + stretch) for i in range(3)]
            transferred_rotated = np.stack([lut_channel[marginal] for lut_channel, marginal in zip(lut, src_rotated_marginals)], axis=-1)

            src_color = np.einsum('ij,kj->ki', mat_rot_inv, transferred_rotated - stretch)
            src_color = np.clip(src_color, 0, 255)

        out_img.set_colors(src_color[:,np.newaxis,:]/255.0)

        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output