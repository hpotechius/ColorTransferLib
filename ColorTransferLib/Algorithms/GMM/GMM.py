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
import cv2
from sklearn.neighbors import KDTree
from copy import deepcopy
#from ColorTransferLib.Utils.Helper import check_compatibility
import multiprocessing as mp
#from .FaissKNeighbors import FaissKNeighbors
from ColorTransferLib.ImageProcessing.Image import Image
import json

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Example-based Color Transfer with Gaussian Mixture Modeling
#   Author: Chunzhi Gu, Xuequan Lu,and Chao Zhang
#   Published in: Journal of Computer-Aided Design and Computer Graphics
#   Year of Publication: 2022
#
# Abstract:
#   Color transfer, which plays a key role in image editing, has attracted noticeable attention recently. It has
#   remained a challenge to date due to various issues such as time-consuming manual adjustments and prior segmentation
#   issues. In this paper, we propose to model color transfer under a probability framework and cast it as a parameter
#   estimation problem. In particular, we relate the transferred image with the example image under the Gaussian
#   Mixture Model (GMM) and regard the transferred image color as the GMM centroids. We employ the
#   Expectation-Maximization (EM) algorithm (E-step and M-step) for optimization. To better preserve gradient
#   information, we introduce a Laplacian based regularization term to the objective function at the M-step which is
#   solved by deriving a gradient descent algorithm. Given the input of a source image and an example image, our
#   method is able to generate multiple color transfer results with increasing EM iterations. Extensive experiments
#   show that our approach generally outperforms other competitive color transfer methods, both visually and
#   quantitatively.
#
# Info:
#   Name: GmmEmColorTransfer
#   Identifier: GMM
#   Link: https://doi.org/10.1016/j.patcog.2022.108716
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class GMM:
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
            "identifier": "GmmEmColorTransfer",
            "title": "Example-based Color Transfer with Gaussian Mixture Modeling",
            "year": 2022,
            "abstract": "Color transfer, which plays a key role in image editing, has attracted noticeable attention "
                        "recently. It has remained a challenge to date due to various issues such as time-consuming "
                        "manual adjustments and prior segmentation issues. In this paper, we propose to model color "
                        "transfer under a probability framework and cast it as a parameter estimation problem. In "
                        "particular, we relate the transferred image with the example image under the Gaussian Mixture "
                        "Model (GMM) and regard the transferred image color as the GMM centroids. We employ the "
                        "Expectation-Maximization (EM) algorithm (E-step and M-step) for optimization. To better "
                        "preserve gradient information, we introduce a Laplacian based regularization term to the "
                        "objective function at the M-step which is solved by deriving a gradient descent algorithm. "
                        "Given the input of a source image and an example image, our method is able to generate "
                        "multiple color transfer results with increasing EM iterations. Extensive experiments show "
                        "that our approach generally outperforms other competitive color transfer methods, both "
                        "visually and quantitatively."
        }

        return info


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        A_start_time = time.time()

        out_img = deepcopy(src)
        src = src.get_raw() * 255.0
        ref = ref.get_raw() * 255.0

        #src = np.array([[[0.0,0.0,255.0],[0.0,255.0,0.0]],[[255.0,255.0,0.0],[255.0,0.0,0.0]]])
        #ref = np.array([[[228.0,206.0,56.0],[183.0,55.0,68.0]],[[0.0,113.0,255.0],[78.0,255.0,142.0]]])

        # [1] Init parameters
        q_max = opt.iterations
        alpha = opt.data_weight
        var = np.tile(np.eye(3)*5,(src.shape[0], src.shape[1], 1, 1))

        mu = opt.regularization_weight#np.random.rand(src.shape[0], src.shape[1], 3) / 250.0 + 0.001  # returns random values of range [0.001, 0.005]
        neigh = opt.neigbhours

        # [2] RGB to CieLAB
        # scaling to range [0, 1]
        src = cv2.cvtColor(src.astype("float32") / 255.0, cv2.COLOR_RGB2LAB).astype("float64")
        ref = cv2.cvtColor(ref.astype("float32") / 255.0, cv2.COLOR_RGB2LAB).astype("float64")
        out = np.copy(src)

        # [3] get 5 nearest neighbor for each x in Y
        start_time = time.time()
        nn_val = GMM.__k_nearest_neighbors(src, ref, neigh)
        print("KNN: " + str(time.time() - start_time))

        for qq in range(q_max):
            print(qq)
            # E-step
            start_time = time.time()
            p_mat = GMM.__E_step(src, out, var, nn_val, neigh)
            print("E-STEP: " + str(time.time() - start_time))
            # M-step
            start_time = time.time()
            out, var = GMM.__M_step(src, out, alpha, mu, p_mat, nn_val, neigh)
            print("M-STEP: " + str(time.time() - start_time))


        out = cv2.cvtColor(out.astype("float32"), cv2.COLOR_LAB2RGB) * 255

        out_img.set_raw(out)
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - A_start_time
        }

        return output

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __E_step(src, out, var, neighbors, k):
        p_mat = np.zeros((src.shape[0], src.shape[1], k))
        # var is always a diagonal matrix -> inverse = reciprocal
        res = np.zeros_like(var)
        res = np.reciprocal(var, out=res, where=var != 0.0)

        for c in range(k):
            sub = neighbors[:, :, c] - out[:, :]
            to = np.einsum('ijkl,ijk->ijk', res, sub)
            top_p = np.exp((-1) * np.einsum('ijk,ijk->ij', sub, to) / 2.)
            bot_p = np.sum(top_p)

            if not np.isinf(bot_p) and not np.isnan(bot_p):
                p_mat[:,:,c] = np.multiply(top_p, bot_p, where=(np.logical_not(np.isinf(top_p))))  # prevent division by 0
        return p_mat
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __M_step(src, out, alpha, mu, probability_mat, neighbors, k):
        sum_p_top = np.zeros((src.shape[0], src.shape[1], 3))
        sum_p_bot = np.sum(probability_mat, axis=2)

        for c in range(k):
            sub = neighbors[:, :, c] - out[:, :]
            to = np.einsum('ij,ijk->ijk', probability_mat[:,:,c], sub)
            sum_p_top += to

        temp_sum_p_bot = np.zeros_like(sum_p_bot)
        div = np.einsum('ijk,ij->ijk', sum_p_top, np.reciprocal(sum_p_bot, out=temp_sum_p_bot, where=sum_p_bot>=1e-300))

        data_term = alpha * div

        regul_term = GMM.__regularization_term(src, out, mu)

        out = np.add(out, data_term, out=np.zeros_like(out), where=(np.logical_not(np.isinf(data_term))))# + regul_term

        out += regul_term

        # Update standard deviation
        sum_p_top_new = np.zeros((src.shape[0], src.shape[1]))
        for c in range(k):
            sub = np.power(np.linalg.norm(out[:, :] - neighbors[:, :, c], axis=2), 2)
            toti = probability_mat[:,:,c] * sub
            sum_p_top_new += toti

        temp_sum_p_bot = np.zeros_like(sum_p_bot)
        std_new = sum_p_top_new * np.reciprocal(sum_p_bot, out=temp_sum_p_bot, where=sum_p_bot>=1e-300)
        var = np.einsum('ij,ijkl->ijkl', std_new, np.tile(np.eye(3),(src.shape[0], src.shape[1], 1, 1)))

        return out, var

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __k_nearest_neighbors(src, ref, k):
        src_num = src.shape[0] * src.shape[1]
        ref_num = ref.shape[0] * ref.shape[1]

        kdt = KDTree(ref.reshape(ref_num, 3), leaf_size=30, metric='euclidean')
        nn_key = kdt.query(src.reshape(src_num, 3), k=k, return_distance=False)
        nn_val = np.zeros((src.shape[0], src.shape[1], k, 3))

        for h in range(src.shape[0]):
            for w in range(src.shape[1]):
                for c in range(k):
                    nn_val[h, w, c] = ref.reshape(ref_num, 3)[nn_key[src.shape[1] * h + w, c]]
        return nn_val

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __regularization_term(src, out, mu):
        l_kern = np.array([[ 0.0, 0.0,-1.0, 0.0, 0,0],
                            [ 0.0,-1.0,-2.0,-1.0, 0,0],
                            [-1.0,-2.0,16.0,-2.0,-1,0],
                            [ 0.0,-1.0,-2.0,-1.0, 0,0],
                            [ 0.0, 0.0,-1.0, 0.0, 0,0]])

        out_padding = np.pad(out, ((2, 2), (2, 2), (0, 0)), 'constant')
        pou_lap = np.zeros((src.shape[0], src.shape[1], 3))
        for wx in range(5):
            for wy in range(5):
                sub = out_padding[wy:src.shape[0]+wy, wx:src.shape[1]+wx, :] - out[:, :]
                pou_lap += l_kern[wy, wx] * sub

        src_lap = cv2.filter2D(src, -1, l_kern)

        regul_term = mu * (pou_lap - src_lap)

        return regul_term