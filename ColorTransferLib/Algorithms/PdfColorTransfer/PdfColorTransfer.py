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
import time
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.Math import get_random_3x3rotation_matrix
from ColorTransferLib.Utils.Math import device_mul_mat3x3_vec3

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from ColorTransferLib.Utils.BaseOptions import BaseOptions
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility



THREADSPERBLOCK = (32, 32)
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
# Link: https://doi.org/10.1109/ICCV.2005.166
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PdfColorTransfer:
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
            "identifier": "PdfColorTransfer",
            "title": "N-dimensional probability density function transfer and its application to color transfer",
            "year": 2005,
            "abstract": "This article proposes an original method to estimate a continuous transformation that maps "
                        "one N-dimensional distribution to another. The method is iterative, non-linear, and is shown "
                        "to converge. Only 1D marginal distribution is used in the estimation process, hence involving "
                        "low computation costs. As an illustration this mapping is applied to color transfer between "
                        "two images of different contents. The paper also serves as a central focal point for "
                        "collecting together the research activity in this area and relating it to the important "
                        "problem of automated color grading."
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, PdfColorTransfer.compatibility)

        # Preprocessing
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        # [1] Change range from [0.0, 1.0] to [0, 255] and copy source and reference to GPU and create output
        device_src = src_color.squeeze()*255.0
        device_ref = ref_color.squeeze()*255.0

        m = 1.0
        soft_m = 1.0 / m
        max_range = 442
        stretch = round(math.pow(max_range, soft_m))
        c_range = int(stretch * 2 + 1)

        for t in range(opt.iterations):
            print(t)
            sci_mat = R.random()#random_state=5)
            mat_rot = sci_mat.as_matrix()
            mat_rot_inv = sci_mat.inv().as_matrix()

             # [2] Create random 3x3 rotation matrix
            mat_rot_tile = np.tile(mat_rot,(src_color.shape[0], 1, 1))
            mat_rot_inv_tile = np.tile(mat_rot_inv,(src_color.shape[0], 1, 1))

            # [3] Rotate source and reference colors with random rotation matrix
            src_rotated = np.einsum('ikl,ik->il', mat_rot_tile, device_src)
            ref_rotated = np.einsum('ikl,ik->il', mat_rot_tile, device_ref)

            # [4] Get 1D marginal
            src_marg_x = src_rotated[:,0]
            src_marg_y = src_rotated[:,1]
            src_marg_z = src_rotated[:,2]
            ref_marg_x = ref_rotated[:,0]
            ref_marg_y = ref_rotated[:,1]
            ref_marg_z = ref_rotated[:,2]

            # [5] Calculate 1D pdf for range [-255, 255] which has to be shifted to [0, 884] (without stretching) in order
            # to allow indexing. The points can be rotated into another octant, therefore the range has to be extended from
            # [0, 255] (256 color values) to [-442, 442] (885 color values). The value 442 was chosen because a color value
            # of (255, 255, 255) can be rotated to (441.7, 0, 0).
            src_cum_marg_x = np.histogram(src_marg_x, bins=c_range, range=(-max_range, max_range), density=True)[0]
            src_cum_marg_y = np.histogram(src_marg_y, bins=c_range, range=(-max_range, max_range), density=True)[0]
            src_cum_marg_z = np.histogram(src_marg_z, bins=c_range, range=(-max_range, max_range), density=True)[0]

            ref_cum_marg_x = np.histogram(ref_marg_x, bins=c_range, range=(-max_range, max_range), density=True)[0]
            ref_cum_marg_y = np.histogram(ref_marg_y, bins=c_range, range=(-max_range, max_range), density=True)[0]
            ref_cum_marg_z = np.histogram(ref_marg_z, bins=c_range, range=(-max_range, max_range), density=True)[0]


            # [6] Calculate cumulative 1D pdf
            src_cum_marg_x = np.cumsum(src_cum_marg_x)
            src_cum_marg_y = np.cumsum(src_cum_marg_y)
            src_cum_marg_z = np.cumsum(src_cum_marg_z)

            ref_cum_marg_x = np.cumsum(ref_cum_marg_x)
            ref_cum_marg_y = np.cumsum(ref_cum_marg_y)
            ref_cum_marg_z = np.cumsum(ref_cum_marg_z)


            # Create LUT
            lut_x = np.zeros(c_range)
            lut_y = np.zeros(c_range)
            lut_z = np.zeros(c_range)

            for i, elem in enumerate(src_cum_marg_x):
                absolute_val_array = np.abs(ref_cum_marg_x - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_x[int(i)] = smallest_difference_index
            for i, elem in enumerate(src_cum_marg_y):
                absolute_val_array = np.abs(ref_cum_marg_y - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_y[int(i)] = smallest_difference_index
            for i, elem in enumerate(src_cum_marg_z):
                absolute_val_array = np.abs(ref_cum_marg_z - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_z[int(i)] = smallest_difference_index

            # Adapt src values
            transferred_rotated_x = lut_x[src_marg_x.astype("int64") + stretch]
            transferred_rotated_y = lut_y[src_marg_y.astype("int64") + stretch]
            transferred_rotated_z = lut_z[src_marg_z.astype("int64") + stretch]
            transferred_rotated = np.concatenate((transferred_rotated_x[:,np.newaxis], transferred_rotated_y[:,np.newaxis]), axis=1)
            transferred_rotated = np.concatenate((transferred_rotated, transferred_rotated_z[:,np.newaxis]), axis=1)

            # [7] Rotate Back
            #transferred_rotated = np.power(transferred_rotated, 1 / soft_m) - stretch
            output = np.einsum('ikl,ik->il', mat_rot_inv_tile, transferred_rotated - stretch)



            # dist_x = np.linalg.norm(transferred_rotated_x - src_rotated[:,0])
            # dist_y = np.linalg.norm(transferred_rotated_y - src_rotated[:,1])
            # dist_z = np.linalg.norm(transferred_rotated_z - src_rotated[:,2])
            # dist = [dist_x, dist_y, dist_z]
            # print(dist)

            device_src = output
            device_src = np.clip(device_src, 0, 255)

        print("DONE")

        device_src = np.clip(device_src, 0, 255)
        out_img.set_colors(device_src[:,np.newaxis,:]/255.0)

        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output