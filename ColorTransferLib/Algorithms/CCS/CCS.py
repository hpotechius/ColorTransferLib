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
from ColorTransferLib.Utils.BaseOptions import BaseOptions
from ColorTransferLib.ImageProcessing.Image import Image as Img
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility
from scipy.sparse.linalg import spsolve
import pyamg
import tensorflow as tf
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

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
class CCS:
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
            "identifier": "CCS",
            "title": "Color Transfer in Correlated Color Space",
            "year": 2006,
            "abstract": "In this paper we present a process called color transfer which can borrow one image's color characteristics from another. Recently Reinhard and his colleagues reported a pioneering work of color transfer. Their technology can produce very believable results, but has to transform pixel values from RGB to lab . Inspired by their work, we advise an approach which can directly deal with the color transfer in any 3D space. From the view of statistics, we consider pixel's value as a threedimension stochastic variable and an image as a set of samples, so the correlations between three components can be measured by covariance. Our method imports covariance between three components of pixel values while calculate the mean along each of the three axes. Then we decompose the covariance matrix using SVD algorithm and get a rotation matrix. Finally we can scale, rotate and shift pixel data of target image to fit data points' cluster of source image in the current color space and get resultant image which takes on source image's look and feel. Besides the global processing, a swatch-based method is introduced in order to manipulate images' color more elaborately. Experimental results confirm the validity and usefulness of our method.",
            "types": ["Image", "Mesh", "PointCloud"]
        }

        return info
    

    @staticmethod
    def gradient_matrices(M, N):
        # Erstellt die Gradientenmatrizen Dx und Dy für ein Bild der Größe MxN
        size = M * N
        Dx = lil_matrix((size, size))
        Dy = lil_matrix((size, size))
        
        for i in range(size):
            # For the Sobel filter in x-direction:
            if (i-N-1) >= 0 and (i-N-1) % N != N-1: Dx[i, i-N-1] = 1
            #if i-N >= 0: Dx[i, i-N] = 0
            if (i-N+1) >= 0 and (i-N+1) % N != 0: Dx[i, i-N+1] = -1
                
            if (i-1) >= 0 and (i-1) % N != N-1: Dx[i, i-1] = 2
            #Dx[i, i] = 0
            if (i+1) < size and (i+1) % N != 0: Dx[i, i+1] = -2
                
            if (i+N-1) < size and ((i+N) % N)-1 >= 0: Dx[i, i+N-1] = 1
            #if i+N < size: Dx[i, i+N] = 0
            if (i+N+1) < size and ((i+N) % N)+1 < N: Dx[i, i+N+1] = -1


            # For the Sobel filter in y-direction:
            if (i-N-1) >= 0 and (i-N-1) % N != N-1: Dy[i, i-N-1] = 1
            if i-N >= 0: Dy[i, i-N] = 2
            if (i+1) < size and (i-N+1) >= 0 and (i-N+1) % N != 0: Dy[i, i-N+1] = 1
                
            #if (i-1) >= 0 and (i-1) % N != N-1: Dy[i, i-1] = 0
            #Dy[i, i] = 0
            #if (i+1) % N != 0: Dy[i, i+1] = 0
                
            if (i+N-1) < size and ((i+N) % N)-1 >= 0: Dy[i, i+N-1] = -1
            if i+N < size: Dy[i, i+N] = -2
            if (i+N+1) < size and ((i+N) % N)+1 < N: Dy[i, i+N+1] = -1


        # for i in range(M):
        #     for j in range(N):
        #         pos = i * N + j
        #         if j > 0:
        #             Dx[pos, pos - 1] = -1
        #             Dy[pos, pos - 1] = -2
                    
        #         if j < N - 1:
        #             Dx[pos, pos + 1] = 1
        #             Dy[pos, pos + 1] = 2
                    
        #         if i > 0:
        #             Dx[pos, pos - N] = -2
        #             Dy[pos, pos - N] = -1
                    
        #         if i < M - 1:
        #             Dx[pos, pos + N] = 2
        #             Dy[pos, pos + N] = 1

        #print(Dy)
        #exit()
        Dx = Dx.tocsr()
        Dy = Dy.tocsr()

                    
        return Dx, Dy   
    
    @staticmethod
    def solve_for_channel(channel_data_f, channel_data_s, M, N, lambda_val, Dx, Dy):
        # Hilfsfunktion, die die Gleichung für einen bestimmten Kanal löst
        print("Fuck1")
        #I = np.identity(M * N)
        size = M * N
        I = lil_matrix((size, size))
        I.setdiag(1)

        # for i in range(M):
        #     for j in range(N):
        #         if i == j:
        #             I[i, j] = 1
        #         else:
        #             I[i, j] = 0
        I = I.tocsr()

        print("Fuck2")
        A = I + lambda_val * (Dx.T @ Dx + Dy.T @ Dy)
        #A = I + lambda_val * (Dx.transpose() @ (Dx) + Dy.transpose() @ (Dy))
        #print(A.shape)
        #print(I.shape)
        print("Fuck3")
        b = channel_data_f + (lambda_val * (Dx.T @ Dx + Dy.T @ Dy) @ channel_data_s)
        #b = channel_data_f + lambda_val * ((Dx.transpose() @ (Dx) + Dy.transpose() @ (Dy))) @ (channel_data_s)
        print("Fuck4")

        #o = spsolve(A, b)
        print("HEHEKKKK")
        #print(I.shape)
        #print(A.shape)
        #print(b.shape)
        ml = pyamg.smoothed_aggregation_solver(A)
        o = ml.solve(b, tol=1e-10)
        #o = spsolve(A.tocsr(), b)
        #o, _ = cg(A.tocsr(), b)

        return o.reshape((M, N))
    
    @staticmethod
    def histogram_matching(source, reference):
        matched = np.empty_like(source)
        for channel in range(source.shape[2]):
            matched[:,:,channel] = CCS.match_single_channel(source[:,:,channel], reference[:,:,channel])
        return matched

    @staticmethod
    def match_single_channel(source, reference):
        s_values, s_counts = np.unique(source, return_counts=True)
        r_values, r_counts = np.unique(reference, return_counts=True)
        
        s_quants = np.cumsum(s_counts).astype(np.float64)
        s_quants /= s_quants[-1]
        
        r_quants = np.cumsum(r_counts).astype(np.float64)
        r_quants /= r_quants[-1]
        
        interp_r_values = np.interp(s_quants, r_quants, r_values)
        
        return interp_r_values[np.searchsorted(s_values, source)]
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()

        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, CCS.compatibility)

        if output["status_code"] == -1:
            return output

        # Preprocessing
        src_img = src.get_raw()
        ref_img = ref.get_raw()



        #histogram matching
        matched_img = CCS.histogram_matching(src_img, ref_img)

        # original src size
        size_src = (src.get_height(), src.get_width(), 3)

        out_img = deepcopy(src)
        out = out_img.get_colors()

        pad = 50

        M, N = src.get_height()+2*pad, src.get_width()+2*pad
        lambda_val = 1.0  # Setzen Sie hier den gewünschten Wert für Lambda ein
        Dx, Dy = CCS.gradient_matrices(M, N)

        o_rgb = np.zeros((M, N, 3))

        # Lösen Sie die Gleichung für jeden Kanal separat
        matched_img = np.pad(matched_img, ((pad,pad),(pad,pad),(0,0)), "reflect")
        src_img = np.pad(src_img, ((pad,pad),(pad,pad),(0,0)), "reflect")
        for channel in range(3):
            print(channel)
            o_rgb[:,:,channel] = CCS.solve_for_channel(matched_img[:,:,channel].flatten(), src_img[:,:,channel].flatten(), M, N, lambda_val, Dx, Dy)


        o_rgb = np.clip(o_rgb, 0, 1)

        #o_rgb = (o_rgb-np.min(o_rgb))/(np.max(o_rgb)-np.min(o_rgb))
        #print(o_rgb)

        #o_rgb = np.pad(o_rgb, ((50,50),(50,50),(0,0)), "reflect")
        #print(o_rgb.shape)
        o_rgb = o_rgb[pad:-1-pad,pad:-1-pad,:]
        out_imgage = Img(array=o_rgb, normalized=True, color="BGR")
        #out_img.set_colors(o_rgb)
        #out_img.set_raw(o_rgb, normalized=True)


        output = {
            "status_code": 0,
            "response": "",
            "object": out_imgage,
            "process_time": time.time() - start_time
        }

        return output
        exit()




        # 
        reshaped_src = np.reshape(src_color, (-1,3))
        reshaped_ref = np.reshape(ref_color, (-1,3))

        num_pts_src = reshaped_src.shape[0]
        num_pts_ref = reshaped_ref.shape[0]


        # [] Calculate mean of each channel (for src and ref)
        mean_src = np.mean(src_color, axis=(0, 1))
        mean_ref = np.mean(ref_color, axis=(0, 1))


        # [] Calculate covariance matrix between the three components (for src and ref)
        cov_src = np.cov(reshaped_src, rowvar=False)
        cov_ref = np.cov(reshaped_ref, rowvar=False)

        # [] SVD of covariance matrices
        U_src, L_src, _ = np.linalg.svd(cov_src)
        U_ref, L_ref, _ = np.linalg.svd(cov_ref)
        
        T_ref = np.eye(4)
        T_ref[:3,3] = mean_ref

        R_ref = np.eye(4)
        R_ref[:3,:3] = U_ref

        S_ref = np.array([[np.sqrt(L_ref[0]), 0, 0, 0],
                          [0, np.sqrt(L_ref[1]), 0, 0],
                          [0, 0, np.sqrt(L_ref[2]), 0],
                          [0, 0, 0, 1]])

        # S_ref = np.array([[L_ref[0], 0, 0, 0],
        #                   [0, L_ref[1], 0, 0],
        #                   [0, 0, L_ref[2], 0],
        #                   [0, 0, 0, 1]])
        
        T_src = np.eye(4)
        T_src[:3,3] = -mean_src
        
        R_src = np.eye(4)
        R_src[:3,:3] = np.linalg.inv(U_src)#U_src.T
        
        S_src = np.array([[1/np.sqrt(L_src[0]), 0, 0, 0],
                          [0, 1/np.sqrt(L_src[1]), 0, 0],
                          [0, 0, 1/np.sqrt(L_src[2]), 0],
                          [0, 0, 0, 1]])
        
        # S_src = np.array([[1/(L_src[0]), 0, 0, 0],
        #                   [0, 1/(L_src[1]), 0, 0],
        #                   [0, 0, 1/(L_src[2]), 0],
        #                   [0, 0, 0, 1]])
        
        # [] turn euclidean points into homogeneous points
        ones = np.ones((num_pts_src, 1))
        homogeneous_src = np.hstack((reshaped_src, ones))

        # [] Apply Transformation: out = T_ref * R_ref * S_ref * S_src * R_src * T_src * src
        transformation_matrix = T_ref @ R_ref @ S_ref @ S_src @ R_src @ T_src
        out = (transformation_matrix @ homogeneous_src.T).T

        out = np.reshape(out[:,:3], size_src)
        out_colors = np.clip(out, 0, 1)

        out_img.set_colors(out_colors)


        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply_old(src, ref, opt):
        start_time = time.time()

        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, CCS.compatibility)

        if output["status_code"] == -1:
            return output

        # Preprocessing
        src_color = src.get_colors()
        ref_color = ref.get_colors()

        # original src size
        size_src = (src.get_height(), src.get_width(), 3)

        out_img = deepcopy(src)
        out = out_img.get_colors()

        # 
        reshaped_src = np.reshape(src_color, (-1,3))
        reshaped_ref = np.reshape(ref_color, (-1,3))

        num_pts_src = reshaped_src.shape[0]
        num_pts_ref = reshaped_ref.shape[0]


        # [] Calculate mean of each channel (for src and ref)
        mean_src = np.mean(src_color, axis=(0, 1))
        mean_ref = np.mean(ref_color, axis=(0, 1))


        # [] Calculate covariance matrix between the three components (for src and ref)
        cov_src = np.cov(reshaped_src, rowvar=False)
        cov_ref = np.cov(reshaped_ref, rowvar=False)

        # [] SVD of covariance matrices
        U_src, L_src, _ = np.linalg.svd(cov_src)
        U_ref, L_ref, _ = np.linalg.svd(cov_ref)
        
        T_ref = np.eye(4)
        T_ref[:3,3] = mean_ref

        R_ref = np.eye(4)
        R_ref[:3,:3] = U_ref

        S_ref = np.array([[np.sqrt(L_ref[0]), 0, 0, 0],
                          [0, np.sqrt(L_ref[1]), 0, 0],
                          [0, 0, np.sqrt(L_ref[2]), 0],
                          [0, 0, 0, 1]])

        # S_ref = np.array([[L_ref[0], 0, 0, 0],
        #                   [0, L_ref[1], 0, 0],
        #                   [0, 0, L_ref[2], 0],
        #                   [0, 0, 0, 1]])
        
        T_src = np.eye(4)
        T_src[:3,3] = -mean_src
        
        R_src = np.eye(4)
        R_src[:3,:3] = np.linalg.inv(U_src)#U_src.T
        
        S_src = np.array([[1/np.sqrt(L_src[0]), 0, 0, 0],
                          [0, 1/np.sqrt(L_src[1]), 0, 0],
                          [0, 0, 1/np.sqrt(L_src[2]), 0],
                          [0, 0, 0, 1]])
        
        # S_src = np.array([[1/(L_src[0]), 0, 0, 0],
        #                   [0, 1/(L_src[1]), 0, 0],
        #                   [0, 0, 1/(L_src[2]), 0],
        #                   [0, 0, 0, 1]])
        
        # [] turn euclidean points into homogeneous points
        ones = np.ones((num_pts_src, 1))
        homogeneous_src = np.hstack((reshaped_src, ones))

        # [] Apply Transformation: out = T_ref * R_ref * S_ref * S_src * R_src * T_src * src
        transformation_matrix = T_ref @ R_ref @ S_ref @ S_src @ R_src @ T_src
        out = (transformation_matrix @ homogeneous_src.T).T

        out = np.reshape(out[:,:3], size_src)
        out_colors = np.clip(out, 0, 1)

        out_img.set_colors(out_colors)


        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output
  