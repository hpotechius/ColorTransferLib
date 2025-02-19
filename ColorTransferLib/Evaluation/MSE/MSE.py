"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
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
# Mean Square Error (MSE)
# 
#
# Source: 
#
# Range []
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class MSE:
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

        return round(mse, 4)