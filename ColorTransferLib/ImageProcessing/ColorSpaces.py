"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import math
from numba import cuda
from ColorTransferLib.Utils.Math import device_mul_mat3x3_vec3


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# TODO: DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class ColorSpaces:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # KERNEL METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    @cuda.jit
    def __kernel_lab_to_rgb(input, output, M_lms2rgb, M_lab2lms1, M_lab2lms2):
        pos = cuda.grid(2)
        x = pos[1] % input.shape[1]
        y = pos[0] % input.shape[0]

        temp = device_mul_mat3x3_vec3(M_lab2lms2, device_mul_mat3x3_vec3(M_lab2lms1, input[y, x]))
        temp = (math.exp(temp[0]), math.exp(temp[1]), math.exp(temp[2]))
        temp = device_mul_mat3x3_vec3(M_lms2rgb, temp)
        output[y, x] = (min(temp[0], 1.0), min(temp[1], 1.0), min(temp[2], 1.0))

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    @cuda.jit
    def __kernel_rgb_to_lab(in_obj, output, m_rgb2lms, m_lms2lab1, m_lms2lab2):
        pos = cuda.grid(2)
        x = pos[1] % in_obj.shape[1]
        y = pos[0] % in_obj.shape[0]

        temp = device_mul_mat3x3_vec3(m_rgb2lms, in_obj[y, x])

        temp = (max(0.000000000001, temp[0]), max(0.000000000001, temp[1]), max(0.000000000001, temp[2]))
        temp = (math.log(temp[0]), math.log(temp[1]), math.log(temp[2]))
        output[y, x] = device_mul_mat3x3_vec3(m_lms2lab2, device_mul_mat3x3_vec3(m_lms2lab1, temp))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # HOST METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rgb_to_lab_host(img):
        img = cuda.to_device(img)

        out = cuda.device_array(img.shape)

        device_m_rgb2lms = cuda.to_device(np.array([
                            [0.3811, 0.5783, 0.0402],
                            [0.1967, 0.7244, 0.0782],
                            [0.0241, 0.1288, 0.8444]]))

        device_m_lms2lab1 = cuda.to_device(np.array([
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, -2.0],
                            [1.0, -1.0, 0.0]]))

        device_m_lms2lab2 = cuda.to_device(np.array([
                            [1.0/math.sqrt(3.0), 0.0, 0.0],
                            [0.0, 1.0/math.sqrt(6.0), 0.0],
                            [0.0, 0.0, 1.0/math.sqrt(2.0)]]))

        threadsperblock = (32, 32)
        blockspergrid_x = int(math.ceil(out.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(out.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        # Now start the kernel
        ColorSpaces.__kernel_rgb_to_lab[blockspergrid, threadsperblock](img,
                                                                      out,
                                                                      device_m_rgb2lms,
                                                                      device_m_lms2lab1,
                                                                      device_m_lms2lab2)

        out = out.copy_to_host()

        return out
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rgb_to_lab_cpu(img):
        device_m_rgb2lms = np.array([
                            [0.3811, 0.5783, 0.0402],
                            [0.1967, 0.7244, 0.0782],
                            [0.0241, 0.1288, 0.8444]])

        device_m_lms2lab1 = np.array([
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, -2.0],
                            [1.0, -1.0, 0.0]])

        device_m_lms2lab2 = np.array([
                            [1.0/math.sqrt(3.0), 0.0, 0.0],
                            [0.0, 1.0/math.sqrt(6.0), 0.0],
                            [0.0, 0.0, 1.0/math.sqrt(2.0)]])
        

        eigen_device_m_rgb2lms = np.tile(device_m_rgb2lms.T, (img.shape[0], 1, 1))
        eigen_device_m_lms2lab1 = np.tile(device_m_lms2lab1.T, (img.shape[0], 1, 1))
        eigen_device_m_lms2lab2 = np.tile(device_m_lms2lab2.T, (img.shape[0], 1, 1))

        result = np.einsum("ijk,ij->ik", eigen_device_m_rgb2lms,  np.squeeze(img))

        result = np.log(result + 0.000000000001)
        result = np.einsum("ijk,ij->ik", eigen_device_m_lms2lab1,  result)
        result = np.einsum("ijk,ij->ik", eigen_device_m_lms2lab2,  result)  
        result = np.expand_dims(result, 1)

        return result
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rgb_to_lab(img):
        out = cuda.device_array(img.shape)

        device_m_rgb2lms = cuda.to_device(np.array([
                            [0.3811, 0.5783, 0.0402],
                            [0.1967, 0.7244, 0.0782],
                            [0.0241, 0.1288, 0.8444]]))

        device_m_lms2lab1 = cuda.to_device(np.array([
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, -2.0],
                            [1.0, -1.0, 0.0]]))

        device_m_lms2lab2 = cuda.to_device(np.array([
                            [1.0/math.sqrt(3.0), 0.0, 0.0],
                            [0.0, 1.0/math.sqrt(6.0), 0.0],
                            [0.0, 0.0, 1.0/math.sqrt(2.0)]]))

        threadsperblock = (32, 32)
        blockspergrid_x = int(math.ceil(out.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(out.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        # Now start the kernel
        ColorSpaces.__kernel_rgb_to_lab[blockspergrid, threadsperblock](img,
                                                                      out,
                                                                      device_m_rgb2lms,
                                                                      device_m_lms2lab1,
                                                                      device_m_lms2lab2)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def lab_to_rgb_cpu(img):
        device_m_lms2rgb = np.array([
                            [4.4679, -3.5873, 0.1193],
                            [-1.2186, 2.3809, -0.1624],
                            [0.0497, -0.2439, 1.2045]])

        device_m_lab2lms1 = np.array([
                            [math.sqrt(3.0)/3.0, 0.0, 0.0],
                            [0.0, math.sqrt(6.0)/6.0, 0.0],
                            [0.0, 0.0, math.sqrt(2.0)/2.0]])

        device_m_lab2lms2 = np.array([
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, -1.0],
                            [1.0, -2.0, 0.0]])
        

        eigen_device_m_lms2rgb = np.tile(device_m_lms2rgb.T, (img.shape[0], 1, 1))
        eigen_device_m_lab2lms1 = np.tile(device_m_lab2lms1.T, (img.shape[0], 1, 1))
        eigen_device_m_lab2lms2 = np.tile(device_m_lab2lms2.T, (img.shape[0], 1, 1))

        result = np.einsum("ijk,ij->ik", eigen_device_m_lab2lms1,  np.squeeze(img))
        result = np.einsum("ijk,ij->ik", eigen_device_m_lab2lms2,  result)
        result = np.exp(result)
        result = np.einsum("ijk,ij->ik", eigen_device_m_lms2rgb,  result)  
        result = np.expand_dims(result, 1)

        return result
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def lab_to_rgb(img):
        out = cuda.device_array(img.shape)

        device_m_lms2rgb = cuda.to_device(np.array([
                            [4.4679, -3.5873, 0.1193],
                            [-1.2186, 2.3809, -0.1624],
                            [0.0497, -0.2439, 1.2045]]))

        device_m_lab2lms1 = cuda.to_device(np.array([
                            [math.sqrt(3.0)/3.0, 0.0, 0.0],
                            [0.0, math.sqrt(6.0)/6.0, 0.0],
                            [0.0, 0.0, math.sqrt(2.0)/2.0]]))

        device_m_lab2lms2 = cuda.to_device(np.array([
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, -1.0],
                            [1.0, -2.0, 0.0]]))

        threadsperblock = (32, 32)
        blockspergrid_x = int(math.ceil(out.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(out.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        ColorSpaces.__kernel_lab_to_rgb[blockspergrid, threadsperblock](img,
                                                                      out,
                                                                      device_m_lms2rgb,
                                                                      device_m_lab2lms1,
                                                                      device_m_lab2lms2)


        return out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def lab_to_rgb_host(img):
        img = cuda.to_device(img)

        out = cuda.device_array(img.shape)

        device_m_lms2rgb = cuda.to_device(np.array([
                            [4.4679, -3.5873, 0.1193],
                            [-1.2186, 2.3809, -0.1624],
                            [0.0497, -0.2439, 1.2045]]))

        device_m_lab2lms1 = cuda.to_device(np.array([
                            [math.sqrt(3.0)/3.0, 0.0, 0.0],
                            [0.0, math.sqrt(6.0)/6.0, 0.0],
                            [0.0, 0.0, math.sqrt(2.0)/2.0]]))

        device_m_lab2lms2 = cuda.to_device(np.array([
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, -1.0],
                            [1.0, -2.0, 0.0]]))

        threadsperblock = (32, 32)
        blockspergrid_x = int(math.ceil(out.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(out.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        ColorSpaces.__kernel_lab_to_rgb[blockspergrid, threadsperblock](img,
                                                                      out,
                                                                      device_m_lms2rgb,
                                                                      device_m_lab2lms1,
                                                                      device_m_lab2lms2)

        out = out.copy_to_host()

        return out