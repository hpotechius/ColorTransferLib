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
from ColorTransferLib.Utils.Math import get_random_3x3rotation_matrix
from ColorTransferLib.Utils.Math import device_mul_mat3x3_vec3

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

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
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, PdfColorTransfer.compatibility)

        # Preprocessing
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        # [1] Change range from [0.0, 1.0] to [0, 255] and copy source and reference to GPU and create output
        print("1. Read src and ref")
        device_src = cuda.to_device(src_color*255.0)
        device_ref = cuda.to_device(ref_color*255.0)
        device_temp_src = cuda.device_array(src_color.shape)
        device_temp_ref = cuda.device_array(ref_color.shape)

        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(241, projection='3d')
        ax.set_title('SRC', fontstyle='italic')
        xyz=np.array(np.random.random((100, 3)))
        # print(xyz.shape)
        # print(src.shape)
        # print(ref.shape)
        sss = src_color[0:src_color.shape[0]:100].copy()
        #print(sss.shape)
        ax.scatter(sss[:,0,0]*255, sss[:,0,1]*255, sss[:,0,2]*255, color=sss[:,0,:])
        ax.set_xlim([-255.0, 255.0])
        ax.set_ylim([-255.0, 255.0])
        ax.set_zlim([-255.0, 255.0])
        ax.view_init(elev=36, azim=-153)

        axr = fig.add_subplot(245, projection='3d')
        axr.set_title('REF', fontstyle='italic')
        rrr = ref_color[0:ref_color.shape[0]:100].copy()
        axr.scatter(rrr[:,0,0]*255, rrr[:,0,1]*255, rrr[:,0,2]*255, color=rrr[:,0,:])
        axr.set_xlim([-255.0, 255.0])
        axr.set_ylim([-255.0, 255.0])
        axr.set_zlim([-255.0, 255.0])
        axr.view_init(elev=36, azim=-153)

        for t in range(opt.iterations):
            print(t)
            # [2] Create random 3x3 rotation matrix
            mat_rot = cuda.to_device(get_random_3x3rotation_matrix())

            # [3] Rotate source and reference colors with random rotation matrix

            print("2. Rotate")
            blockspergrid_x = int(math.ceil(device_temp_src.shape[0] / THREADSPERBLOCK[0]))
            blockspergrid_y = int(math.ceil(device_temp_src.shape[1] / THREADSPERBLOCK[1]))
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            PdfColorTransfer.__kernel_apply[blockspergrid, THREADSPERBLOCK](device_src, mat_rot, device_temp_src)

            # TEMP
            ax1 = fig.add_subplot(242, projection='3d')
            ax1.set_title('SRC Rotated', fontstyle='italic')
            out = device_temp_src.copy_to_host()
            out = out[0:out.shape[0]:100].copy()
            ax1.scatter(out[:,0,0], out[:,0,1], out[:,0,2], color=sss[:,0,:]) # plot the point (2,3,4) on the figure
            ax1.set_xlim([-255.0, 255.0])
            ax1.set_ylim([-255.0, 255.0])
            ax1.set_zlim([-255.0, 255.0])
            ax1.view_init(elev=36, azim=-153)

            blockspergrid_x = int(math.ceil(device_temp_ref.shape[0] / THREADSPERBLOCK[0]))
            blockspergrid_y = int(math.ceil(device_temp_ref.shape[1] / THREADSPERBLOCK[1]))
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            PdfColorTransfer.__kernel_apply[blockspergrid, THREADSPERBLOCK](device_ref, mat_rot, device_temp_ref)

            ax1r = fig.add_subplot(246, projection='3d')
            ax1r.set_title('SRC Rotated', fontstyle='italic')
            outr = device_temp_ref.copy_to_host()
            print(outr.shape)
            outr = outr[0:outr.shape[0]:100].copy()
            ax1r.scatter(outr[:,0,0], outr[:,0,1], outr[:,0,2], color=rrr[:,0,:]) # plot the point (2,3,4) on the figure
            ax1r.set_xlim([-255.0, 255.0])
            ax1r.set_ylim([-255.0, 255.0])
            ax1r.set_zlim([-255.0, 255.0])
            ax1r.view_init(elev=36, azim=-153)

            # [4] Get 1D marginal and convert to int values

            print("3. Get marginals")
            src_marg_x = np.around(device_temp_src[:,0,0])
            src_marg_y = np.around(device_temp_src[:,0,1])
            src_marg_z = np.around(device_temp_src[:,0,2])
            ref_marg_x = np.around(device_temp_ref[:,0,0])
            ref_marg_y = np.around(device_temp_ref[:,0,1])
            ref_marg_z = np.around(device_temp_ref[:,0,2])

            ax2 = fig.add_subplot(243, projection='3d')
            ax2.set_title('Marginals', fontstyle='italic')
            outx = src_marg_x[0:src_marg_x.shape[0]:100].copy()
            ax2.scatter(outx, outx*0, outx*0, color=sss[:,0,:])
            outy = src_marg_y[0:src_marg_y.shape[0]:100].copy()
            ax2.scatter(outy*0, outy, outy*0, color=sss[:,0,:])
            outz = src_marg_z[0:src_marg_z.shape[0]:100].copy()
            ax2.scatter(outz*0, outz*0, outy, color=sss[:,0,:])
            ax2.set_xlim([-255.0, 255.0])
            ax2.set_ylim([-255.0, 255.0])
            ax2.set_zlim([-255.0, 255.0])
            ax2.view_init(elev=36, azim=-153)

            ax2r = fig.add_subplot(247, projection='3d')
            ax2r.set_title('Marginals', fontstyle='italic')
            outxr = ref_marg_x[0:ref_marg_x.shape[0]:100].copy()
            ax2r.scatter(outxr, outxr*0, outxr*0, color=rrr[:,0,:])
            outyr = ref_marg_y[0:ref_marg_y.shape[0]:100].copy()
            ax2r.scatter(outyr*0, outyr, outyr*0, color=rrr[:,0,:])
            outzr = ref_marg_z[0:ref_marg_z.shape[0]:100].copy()
            ax2r.scatter(outzr*0, outzr*0, outyr, color=rrr[:,0,:])
            ax2r.set_xlim([-255.0, 255.0])
            ax2r.set_ylim([-255.0, 255.0])
            ax2r.set_zlim([-255.0, 255.0])
            ax2r.view_init(elev=36, azim=-153)

            # [5] Calculate 1D pdf for range [-255, 255] which has to be shifted to [0, 884] (without stretching) in order
            # to allow indexing. The points can be rotated into another octant, therefore the range has to be extended from
            # [0, 255] (256 color values) to [-442, 442] (885 color values). The value 442 was chosen because a color value
            # of (255, 255, 255) can be rotated to (441.7, 0, 0).

            print("4. Get pdfs")
            src_cum_marg_x = np.zeros(885)
            for elem in src_marg_x: src_cum_marg_x[int(elem+442.0)] += 1.0
            src_cum_marg_x = src_cum_marg_x/np.sum(src_cum_marg_x)

            src_cum_marg_y = np.zeros(885)
            for elem in src_marg_y: src_cum_marg_y[int(elem+442.0)] += 1.0
            src_cum_marg_y = src_cum_marg_y/np.sum(src_cum_marg_y)

            src_cum_marg_z = np.zeros(885)
            for elem in src_marg_z: src_cum_marg_z[int(elem+442.0)] += 1.0
            src_cum_marg_z = src_cum_marg_z/np.sum(src_cum_marg_z)

            ref_cum_marg_x = np.zeros(885)
            for elem in ref_marg_x: ref_cum_marg_x[int(elem+442.0)] += 1.0
            ref_cum_marg_x = ref_cum_marg_x/np.sum(ref_cum_marg_x)

            ref_cum_marg_y = np.zeros(885)
            for elem in ref_marg_y: ref_cum_marg_y[int(elem+442.0)] += 1.0
            ref_cum_marg_y = ref_cum_marg_y/np.sum(ref_cum_marg_y)

            ref_cum_marg_z = np.zeros(885)
            for elem in ref_marg_z: ref_cum_marg_z[int(elem+442.0)] += 1.0
            ref_cum_marg_z = ref_cum_marg_z/np.sum(ref_cum_marg_z)



            # [6] Calculate cumulative 1D pdf
            print("5. Get cummulative pdfs")
            for i in range(1,src_cum_marg_x.shape[0]): src_cum_marg_x[i] += src_cum_marg_x[i-1]
            for i in range(1,src_cum_marg_y.shape[0]): src_cum_marg_y[i] += src_cum_marg_y[i-1]
            for i in range(1,src_cum_marg_z.shape[0]): src_cum_marg_z[i] += src_cum_marg_z[i-1]

            for i in range(1,ref_cum_marg_x.shape[0]): ref_cum_marg_x[i] += ref_cum_marg_x[i-1]
            for i in range(1,ref_cum_marg_y.shape[0]): ref_cum_marg_y[i] += ref_cum_marg_y[i-1]
            for i in range(1,ref_cum_marg_z.shape[0]): ref_cum_marg_z[i] += ref_cum_marg_z[i-1]

            #fig2 = plt.figure(figsize=(20, 10))
            #ax2d = fig2.add_subplot(241)
            #ax2d.set_title('SRC', fontstyle='italic')
            #ax2d.scatter(range(-442, 443), src_cum_marg_x, color=(0.0,1.0,0.0))
            #ax2d.set_xlim([-442.0, 442.0])
            #ax2d.set_ylim([0, 1])

            # [8] create inverse 1D cumulative pdf from reference color from range [0, 1] with step size of 0.01
            """src_cum_marg_x = np.round(src_cum_marg_x, 3)
            src_cum_marg_y = np.round(src_cum_marg_y, 3)
            src_cum_marg_z = np.round(src_cum_marg_z, 3)
            ref_cum_marg_x = np.round(ref_cum_marg_x, 3)
            ref_cum_marg_y = np.round(ref_cum_marg_y, 3)
            ref_cum_marg_z = np.round(ref_cum_marg_z, 3)"""

            # Create LUT
            print("6. Create LUTs")
            lut_x = np.zeros(885)
            lut_y = np.zeros(885)
            lut_z = np.zeros(885)

            for i, elem in enumerate(src_cum_marg_x):
                absolute_val_array = np.abs(ref_cum_marg_x - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_x[int(i)] = smallest_difference_index
                #print(str(i - 442.0) + " - " + str(smallest_difference_index-442.0))
            for i, elem in enumerate(src_cum_marg_y):
                absolute_val_array = np.abs(ref_cum_marg_y - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_y[int(i)] = smallest_difference_index
            for i, elem in enumerate(src_cum_marg_z):
                absolute_val_array = np.abs(ref_cum_marg_z - elem)
                smallest_difference_index = absolute_val_array.argmin()
                lut_z[int(i)] = smallest_difference_index

            fig2 = plt.figure(figsize=(20, 10))
            ax2d = fig2.add_subplot(241)
            ax2d.set_title('SRC CU', fontstyle='italic')
            ax2d.scatter(range(-442, 443), src_cum_marg_x, color=(0.0,1.0,0.0))
            ax2d.set_xlim([-442.0, 442.0])
            ax2d.set_ylim([0, 1])

            ax2d = fig2.add_subplot(242)
            ax2d.set_title('REF CU', fontstyle='italic')
            ax2d.scatter(range(-442, 443), ref_cum_marg_x, color=(0.0,1.0,0.0))
            ax2d.set_xlim([-442.0, 442.0])
            ax2d.set_ylim([0, 1])

            ax2d = fig2.add_subplot(243)
            ax2d.set_title('LUT X', fontstyle='italic')
            ax2d.scatter(range(-442, 443), lut_x, color=(0.0,1.0,0.0))
            ax2d.set_xlim([-442.0, 442.0])
            ax2d.set_ylim([-442.0, 442.0])

            """
            ax2dy = fig2.add_subplot(242)
            ax2dy.set_title('Y', fontstyle='italic')
            ax2dy.scatter(range(-442, 443), lut_y, color=(0.0,1.0,0.0))
            ax2dy.set_xlim([-442.0, 442.0])
            ax2dy.set_ylim([-442.0, 442.0])

            ax2dz = fig2.add_subplot(243)
            ax2dz.set_title('Z', fontstyle='italic')
            ax2dz.scatter(range(-442, 443), lut_z, color=(0.0,1.0,0.0))
            ax2dz.set_xlim([-442.0, 442.0])
            ax2dz.set_ylim([-442.0, 442.0])
            """

            # Adapt src values

            print("7. Transform")
            device_temp_srcXX = cuda.device_array(src_color.shape)
            blockspergrid_x = int(math.ceil(device_temp_src.shape[0] / THREADSPERBLOCK[0]))
            blockspergrid_y = int(math.ceil(device_temp_src.shape[1] / THREADSPERBLOCK[1]))
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            print(device_temp_src.shape)
            PdfColorTransfer.__kernel_apply2[blockspergrid, THREADSPERBLOCK](device_temp_src,
                                                                             cuda.to_device(lut_x),
                                                                             cuda.to_device(lut_y),
                                                                             cuda.to_device(lut_z),
                                                                             device_temp_srcXX)

            # [7] Rotate Back
            print("8. Rotate back")
            mat_rot_inv = mat_rot.transpose()
            device_temp_out = cuda.device_array(src_color.shape)

            blockspergrid_x = int(math.ceil(device_temp_src.shape[0] / THREADSPERBLOCK[0]))
            blockspergrid_y = int(math.ceil(device_temp_src.shape[1] / THREADSPERBLOCK[1]))
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            PdfColorTransfer.__kernel_apply[blockspergrid, THREADSPERBLOCK](device_temp_srcXX,
                                                                            mat_rot_inv,
                                                                            device_temp_out)

            device_temp_src = device_temp_out

            print("DONE")

        out = device_temp_src.copy_to_host() / 255.0
        #print(out)

        #plt.show()

        out_img.set_colors(out)

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
    def __kernel_apply(src_mat, rot_mat, out_mat):
        pos = cuda.grid(2)
        x = pos[1] % src_mat.shape[1]
        y = pos[0] % src_mat.shape[0]

        out_mat[y, x] = device_mul_mat3x3_vec3(rot_mat, src_mat[y, x])

    # ------------------------------------------------------------------------------------------------------------------
    # method description
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    @cuda.jit
    def __kernel_apply2(src_mat, lutx, luty, lutz, out_mat):
        pos = cuda.grid(2)
        x = pos[1] % src_mat.shape[1]
        y = pos[0] % src_mat.shape[0]

        idx, idy, idz = src_mat[y, x]
        temp = (lutx[int(idx+442.0)]- 442.0, luty[int(idy+442.0)]- 442.0, lutz[int(idz+442.0)]- 442.0)
        out_mat[y, x] = temp

