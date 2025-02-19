"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import math
import time
from copy import deepcopy
import numpy as np

from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo

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
# Info:
#   Name: PdfColorTransfer
#   Identifier: PDF
#   Link: https://doi.org/10.1109/ICCV.2005.166
#
# Implementation Details:
#   m = 1
#   iterations = 20
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PDF:
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

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = PDF.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = PDF.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = PDF.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = PDF.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = PDF.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = PDF.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = PDF.__apply_mesh(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    
    # ------------------------------------------------------------------------------------------------------------------
    # Generate a random 3x3 rotation matrix
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def random_rotation_matrix():
        random_state = np.random.default_rng()
        H = np.eye(3) + random_state.standard_normal((3, 3))
        Q, R = np.linalg.qr(H)
        return Q
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_color, ref_color, opt):
        # Change range from [0.0, 1.0] to [0, 255]
        src_color = src_color.squeeze() * 255.0
        ref_color = ref_color.squeeze() * 255.0

        m = 1.0
        max_range = 442
        stretch = round(math.pow(max_range, 1.0 / m))
        c_range = int(stretch * 2 + 1)

        for t in range(opt.iterations):
            mat_rot = PDF.random_rotation_matrix()
            mat_rot_inv = np.linalg.inv(mat_rot)

            src_rotated = np.einsum('ij,kj->ki', mat_rot, src_color)
            ref_rotated = np.einsum('ij,kj->ki', mat_rot, ref_color)

            # Calculate 1D pdf
            src_marginals = [np.histogram(src_rotated[:, i], bins=c_range, range=(-max_range, max_range), density=True)[0] for i in range(3)]
            ref_marginals = [np.histogram(ref_rotated[:, i], bins=c_range, range=(-max_range, max_range), density=True)[0] for i in range(3)]

            # Calculate cumulative 1D pdf
            src_cum_marginals = [np.cumsum(marg) for marg in src_marginals]
            ref_cum_marginals = [np.cumsum(marg) for marg in ref_marginals]

            lut = []
            for src_marg, ref_marg in zip(src_cum_marginals, ref_cum_marginals):
                lut_channel = np.zeros(c_range)
                for i, elem in enumerate(src_marg):
                    lut_channel[i] = np.abs(ref_marg - elem).argmin()
                lut.append(lut_channel)

            src_rotated_marginals = [(src_rotated[:, i].astype("int64") + stretch) for i in range(3)]
            transferred_rotated = np.stack([lut_channel[marginal] for lut_channel, marginal in zip(lut, src_rotated_marginals)], axis=-1)

            src_color = np.einsum('ij,kj->ki', mat_rot_inv, transferred_rotated - stretch)
            src_color = np.clip(src_color, 0, 255)

        return src_color[:,np.newaxis,:]/255.0
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = PDF.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_video(src, ref, opt): 
        # check if type is video
        out_colors_arr = []
        src_colors = src.get_colors()

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_images()[0])

            out_colors = PDF.__color_transfer(src_color, ref_color, opt)

            out_img.set_colors(out_colors)
            out_colors_arr.append(out_img)

        outp = Video(imgs=out_colors_arr, fps=src.get_fps())

        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_volumetricvideo(src, ref, opt): 
        out_colors_arr = []
        src_colors = src.get_colors()

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = PDF.__color_transfer(src_color, ref_color, opt)

            out_img.set_colors(out_colors)
            out_colors_arr.append(out_img)
            outp = VolumetricVideo(meshes=out_colors_arr, file_name=src.get_file_name())

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
                src_color = src_lightfield_array[row][col].get_colors()
                ref_color = ref.get_colors()

                out_colors = PDF.__color_transfer(src_color, ref_color, opt)

                out_lightfield_array[row][col].set_colors(out_colors)

        return out
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_gaussiansplatting(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = PDF.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_pointcloud(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = PDF.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = PDF.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

