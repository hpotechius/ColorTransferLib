"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import time
from copy import deepcopy
import pyamg
from scipy.sparse import lil_matrix
import cv2  # neu

from ColorTransferLib.DataTypes.Image import Image as Img
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo
from ColorTransferLib.Utils.ColorSpaces import ColorSpaces  # neu


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
#   Identifier: Xiao09
#   Link: https://doi.org/10.1111/j.1467-8659.2009.01566.x
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Xiao09:
    # ------------------------------------------------------------------------------------------------------------------
    # Checks source and reference compatibility
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        output = {
            "status_code": 0,
            "response": "",
            "object": None,
            "process_time": 0
        }

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField" or ref.get_type() == "GaussianSplatting" or ref.get_type() == "PointCloud":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = Xiao09.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = Xiao09.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = Xiao09.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = Xiao09.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = Xiao09.__apply_mesh(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1
            out_obj = None

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def gradient_matrices(M, N):
        # Creates the gradient matrices Dx and Dy for an image of size MxN
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
            matched[:,:,channel] = Xiao09.match_single_channel(source[:,:,channel], reference[:,:,channel])
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
    def __color_transfer(src_img, ref_img, opt):
        # ensure float32 arrays
        src_arr = np.asarray(src_img, dtype=np.float32)
        ref_arr = np.asarray(ref_img, dtype=np.float32)

        if src_arr.shape[-1] != 3 or ref_arr.shape[-1] != 3:
            raise ValueError(
                f"__color_transfer expects last dimension = 3, "
                f"got src_shape={src_arr.shape}, ref_shape={ref_arr.shape}"
            )

        H, W, _ = src_arr.shape

        # Scale reference to source resolution if necessary
        if ref_arr.shape[0] != H or ref_arr.shape[1] != W:
            ref_arr = cv2.resize(ref_arr, dsize=(W, H), interpolation=cv2.INTER_AREA)

        src_flat = src_arr.reshape(-1, 3)
        ref_flat = ref_arr.reshape(-1, 3)

        # ------------------------------------------------------------------
        # choose working colorspace
        # ------------------------------------------------------------------
        if opt.colorspace == "cielab":
            src_cs_flat = ColorSpaces.rgb_to_cielab(src_flat)
            ref_cs_flat = ColorSpaces.rgb_to_cielab(ref_flat)
        elif opt.colorspace == "lalphabeta":
            src_cs_flat = ColorSpaces.rgb_to_lalphabeta(src_flat)
            ref_cs_flat = ColorSpaces.rgb_to_lalphabeta(ref_flat)
        elif opt.colorspace == "rgb":
            src_cs_flat = src_flat
            ref_cs_flat = ref_flat
        else:
            raise ValueError(f"Unsupported colorspace: {opt.colorspace}")

        src_cs = src_cs_flat.reshape(H, W, 3)
        ref_cs = ref_cs_flat.reshape(H, W, 3)

        # ------------------------------------------------------------------
        # 1) Histogram matching in working color space
        # ------------------------------------------------------------------
        matched_img = Xiao09.histogram_matching(src_cs, ref_cs)

        pad = 50
        M, N = H + 2 * pad, W + 2 * pad

        lambda_val = 1.0
        Dx, Dy = Xiao09.gradient_matrices(M, N)

        o_cs = np.zeros((M, N, 3), dtype=np.float32)

        # 2) Gradient-preserving optimization per channel in working color space
        matched_p = np.pad(matched_img, ((pad, pad), (pad, pad), (0, 0)), "reflect")
        src_p = np.pad(src_cs, ((pad, pad), (pad, pad), (0, 0)), "reflect")

        for channel in range(3):
            o_cs[:, :, channel] = Xiao09.solve_for_channel(
                matched_p[:, :, channel].flatten(),
                src_p[:, :, channel].flatten(),
                M,
                N,
                lambda_val,
                Dx,
                Dy,
            )

        o_cs = o_cs[pad:M - pad, pad:N - pad, :]

        # ------------------------------------------------------------------
        # 3) Back to RGB
        # ------------------------------------------------------------------
        out_flat_cs = o_cs.reshape(-1, 3)
        if opt.colorspace == "cielab":
            out_flat_rgb = ColorSpaces.cielab_to_rgb(out_flat_cs)
        elif opt.colorspace == "lalphabeta":
            out_flat_rgb = ColorSpaces.lalphabeta_to_rgb(out_flat_cs)
        else:  # "rgb"
            out_flat_rgb = out_flat_cs

        out_rgb = out_flat_rgb.reshape(H, W, 3)
        out_rgb = np.clip(out_rgb, 0.0, 1.0).astype(np.float32)

        return out_rgb
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = Xiao09.__color_transfer(src_img, ref_img, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_video(src, ref, opt): 
        # check if type is video
        out_raw_arr = []
        src_raws = src.get_raw()

        for i, src_raw in enumerate(src_raws):
            # Preprocessing
            ref_raw = ref.get_raw()
            out_img = deepcopy(src.get_images()[0])

            out_colors = Xiao09.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_colors(out_colors)
            out_raw_arr.append(out_img)

        outp = Video(imgs=out_raw_arr, fps=src.get_fps())

        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_volumetricvideo(src, ref, opt): 
        out_raw_arr = []
        src_raws = src.get_raw()

        for i, src_raw in enumerate(src_raws):
            # Preprocessing
            ref_raw = ref.get_raw()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = Xiao09.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_colors(out_colors)

            out_raw_arr.append(out_img)
            outp = VolumetricVideo(meshes=out_raw_arr, file_name=src.get_file_name())

        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_lightfield(src, ref, opt):
        src_lightfield_array = src.get_image_array()
        out = deepcopy(src)
        out_lightfield_array = out.get_image_array()

        for row in range(src.get_grid_size()[0]):
            for col in range(src.get_grid_size()[1]):
                src_raw = src_lightfield_array[row][col].get_raw()
                ref_raw = ref.get_raw()

                out_colors = Xiao09.__color_transfer(src_raw, ref_raw, opt)

                out_lightfield_array[row][col].set_colors(out_colors)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = Xiao09.__color_transfer(src_img, ref_img, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

