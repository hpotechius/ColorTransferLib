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
from copy import deepcopy
import pyamg
from scipy.sparse import lil_matrix

from ColorTransferLib.ImageProcessing.Image import Image as Img
from ColorTransferLib.Utils.Helper import check_compatibility


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Gradient-Preserving Color Transfer
#   Author: Xuezhong Xiao, Lizhuang Ma
#   Published in: IEEE Computer Graphics and Applications
#   Year of Publication: 2009
#
# Abstract:
#   Color transfer is an image processing technique which can produce a new image combining one source image s contents 
#   with another image s color style. While being able to produce convincing results, however, Reinhard et al. s 
#   pioneering work has two problems-mixing up of colors in different regions and the fidelity problem. Many local color 
#   transfer algorithms have been proposed to resolve the first problem, but the second problem was paid few attentions.
#   In this paper, a novel color transfer algorithm is presented to resolve the fidelity problem of color transfer in 
#   terms of scene details and colors. It s well known that human visual system is more sensitive to local intensity 
#   differences than to intensity itself. We thus consider that preserving the color gradient is necessary for scene 
#   fidelity. We formulate the color transfer problem as an optimization problem and solve it in two steps-histogram 
#   matching and a gradient-preserving optimization. Following the idea of the fidelity in terms of color and gradient, 
#   we also propose a metric for objectively evaluating the performance of example-based color transfer algorithms. The 
#   experimental results show the validity and high fidelity of our algorithm and that it can be used to deal with local 
#   color transfer.
#
# Info:
#   Name: GradientPreservingColorTransfer
#   Identifier: GPC
#   Link: https://doi.org/10.1111/j.1467-8659.2009.01566.x
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class GPC:
    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "GPC",
            "title": "Gradient-Preserving Color Transfer",
            "year": 2009,
            "abstract": "Color transfer is an image processing technique which can produce a new image combining one source image’s contents with another image’s color style. While being able to produce convincing results, however, Reinhard et al.’s pioneering work has two problems—mixing up of colors in different regions and the fidelity problem. Many local color transfer algorithms have been proposed to resolve the first problem, but the second problem was paid few attentions. In this paper, a novel color transfer algorithm is presented to resolve the fidelity problem of color transfer in terms of scene details and colors. It’s well known that human visual system is more sensitive to local intensity differences than to intensity itself. We thus consider that preserving the color gradient is necessary for scene fidelity. We formulate the color transfer problem as an optimization problem and solve it in two steps—histogram matching and a gradient-preserving optimization. Following the idea of the fidelity in terms of color and gradient, we also propose a metric for objectively evaluating the performance of example-based color transfer algorithms. The experimental results show the validity and high fidelity of our algorithm and that it can be used to deal with local color transfer.",
            "types": ["Image", "Mesh"]
        }

        return info
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
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

        Dx = Dx.tocsr()
        Dy = Dy.tocsr()

        return Dx, Dy   
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def solve_for_channel(channel_data_f, channel_data_s, M, N, lambda_val, Dx, Dy):
        size = M * N
        I = lil_matrix((size, size))
        I.setdiag(1)
        I = I.tocsr()

        A = I + lambda_val * (Dx.T @ Dx + Dy.T @ Dy)
        b = channel_data_f + (lambda_val * (Dx.T @ Dx + Dy.T @ Dy) @ channel_data_s)

        ml = pyamg.smoothed_aggregation_solver(A)
        o = ml.solve(b, tol=1e-10)

        return o.reshape((M, N))
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def histogram_matching(source, reference):
        matched = np.empty_like(source)
        for channel in range(source.shape[2]):
            matched[:,:,channel] = GPC.match_single_channel(source[:,:,channel], reference[:,:,channel])
        return matched

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
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
        output = check_compatibility(src, ref, GPC.compatibility)

        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output

        # Preprocessing
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        

        #histogram matching
        matched_img = GPC.histogram_matching(src_img, ref_img)

        # original src size
        #size_src = (src.get_height(), src.get_width(), 3)

        out_img = deepcopy(src)
        out = out_img.get_colors()

        pad = 50

        M, N = src_img.shape[0]+2*pad, src_img.shape[1]+2*pad
        lambda_val = 1.0  # Setzen Sie hier den gewünschten Wert für Lambda ein
        Dx, Dy = GPC.gradient_matrices(M, N)

        o_rgb = np.zeros((M, N, 3))

        # Lösen Sie die Gleichung für jeden Kanal separat
        matched_img = np.pad(matched_img, ((pad,pad),(pad,pad),(0,0)), "reflect")
        src_img = np.pad(src_img, ((pad,pad),(pad,pad),(0,0)), "reflect")
        for channel in range(3):
            o_rgb[:,:,channel] = GPC.solve_for_channel(matched_img[:,:,channel].flatten(), src_img[:,:,channel].flatten(), M, N, lambda_val, Dx, Dy)

        o_rgb = np.clip(o_rgb, 0, 1)
        o_rgb = o_rgb[pad:o_rgb.shape[0]-pad,pad:o_rgb.shape[1]-pad,:]

        out_img.set_colors(o_rgb)

        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output