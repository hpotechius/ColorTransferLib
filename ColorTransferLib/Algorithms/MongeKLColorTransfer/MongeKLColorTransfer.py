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
from scipy.linalg import fractional_matrix_power
from ColorTransferLib.Utils.Helper import check_compatibility
from copy import deepcopy

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
# Link: https://doi.org/10.1049/cp:20070055
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class MongeKLColorTransfer:
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
            "identifier": "MongeKLColorTransfer",
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
                        "colour spaces."
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, MongeKLColorTransfer.compatibility)

        # Preprocessing
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)
        """
        EPS = 2.2204e-16
        def MKL(A, B):
            Da2, Ua = np.linalg.eig(A)

            Da2 = np.diag(Da2)
            Da2[Da2 < 0] = 0
            Da = np.sqrt(Da2 + EPS)
            C = Da @ np.transpose(Ua) @ B @ Ua @ Da
            Dc2, Uc = np.linalg.eig(C)

            Dc2 = np.diag(Dc2)
            Dc2[Dc2 < 0] = 0
            Dc = np.sqrt(Dc2 + EPS)
            Da_inv = np.diag(1 / (np.diag(Da)))
            T = Ua @ Da_inv @ Uc @ Dc @ np.transpose(Uc) @ Da_inv @ np.transpose(Ua)
            return T

        X0 = np.reshape(src, (-1, 3), 'F')
        X1 = np.reshape(ref, (-1, 3), 'F')
        A = np.cov(X0, rowvar=False)
        B = np.cov(X1, rowvar=False)
        T = MKL(A, B)
        mX0 = np.mean(X0, axis=0)
        mX1 = np.mean(X1, axis=0)
        XR = (X0 - mX0) @ T + mX1
        IR = np.reshape(XR, src.shape, 'F')
        IR = np.real(IR)
        IR[IR > 1] = 1
        IR[IR < 0] = 0

        return IR
        """

        # Convert colors from RGB to lalphabeta color space
        src_color_rgb = src_color.reshape(src_color.shape[0], 3)
        ref_color_rgb = ref_color.reshape(ref_color.shape[0], 3)


        #src_mean = np.mean(src_color_rgb, axis=0)
        #ref_mean = np.mean(ref_color_rgb, axis=0)
        src_cov = np.cov(src_color_rgb, rowvar=False)
        ref_cov = np.cov(ref_color_rgb, rowvar=False)


        src_covs = fractional_matrix_power(src_cov, 0.5)
        src_covsr = fractional_matrix_power(src_cov, -0.5)

        M = np.dot(src_covsr, np.dot(fractional_matrix_power(np.dot(src_covs, np.dot(ref_cov, src_covs)), 0.5), src_covsr))

        # has to be rewritten
        mean_src = np.mean(src_color_rgb, axis=0)
        mean_ref = np.mean(ref_color_rgb, axis=0)
        XR = (src_color_rgb - mean_src) @ M + mean_ref
        IR = np.reshape(XR, src_color.shape, 'F')
        IR = np.real(IR)
        IR[IR > 1] = 1
        IR[IR < 0] = 0



        out_img.set_colors(IR)
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img
        }

        return output

        out = np.dot(src_color_rgb, M)

        #lab_new = f_out[:,:3]
        #lab_new = lab_new.reshape(src.get_num_vertices(), 1, 3)
        #lab_new = ColorSpaces.lab_to_rgb_host(np.ascontiguousarray(lab_new, dtype=np.float32))
        #lab_new = np.clip(lab_new, 0.0, 1.0)

        return out
