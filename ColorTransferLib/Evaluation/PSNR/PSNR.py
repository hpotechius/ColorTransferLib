"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import math
import numpy as np

from ColorTransferLib.DataTypes.Image import Image

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: -
#   Author: -
#   Published in: -
#   Year of Publication: -
#
# Abstract:
#   -

# Info:
#   Name: Peak Signal-to-Noise Ratio
#   Shortname: PSNR
#   Identifier: PSNR
#   Link: -
#   Range: [0, inf] with higher values indicates better quality
#
# Implementation Details:
#   if source and output are identical, the PSNR will be not considered for the evaluation
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PSNR:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = args[0]
        ref = args[2]
        src_img = src.get_raw()
        ref_img = ref.get_raw()

        num_pix = src_img.shape[0] * src_img.shape[1]

        mse_c = np.sum(np.power(np.subtract(src_img, ref_img), 2), axis=(0,1)) / num_pix
        mse = np.sum(mse_c) / 3

        psnr_v = 10 * math.log10(math.pow(1, 2) / mse)

        return round(psnr_v, 4)