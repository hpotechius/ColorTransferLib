"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import time
from copy import deepcopy

from ColorTransferLib.Utils.ColorSpaces import ColorSpaces
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo


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
#   Identifier: Reinhard01
#   Link: https://doi.org/10.1109/38.946629
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Reinhard01:
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
            out_obj = Reinhard01.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = Reinhard01.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = Reinhard01.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = Reinhard01.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = Reinhard01.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = Reinhard01.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = Reinhard01.__apply_mesh(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorithm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_color, ref_color, opt):
        # remember original shape (e.g. H×W×3, N×1×3, N×3, ...)
        orig_shape = src_color.shape

        # flatten to (N, 3) for global statistics (Reinhard is global)
        src_flat = src_color.reshape(-1, 3)
        ref_flat = ref_color.reshape(-1, 3)

        # [2] Convert RGB to lab (or stay in RGB) in flattened form
        if opt.colorspace == "lalphabeta":
            lab_src = ColorSpaces.rgb_to_lalphabeta(src_flat)
            lab_ref = ColorSpaces.rgb_to_lalphabeta(ref_flat)
        elif opt.colorspace == "cielab":
            lab_src = ColorSpaces.rgb_to_cielab(src_flat)
            lab_ref = ColorSpaces.rgb_to_cielab(ref_flat)
        elif opt.colorspace == "rgb":
            lab_src = src_flat
            lab_ref = ref_flat
        else:
            raise ValueError(f"Unsupported colorspace: {opt.colorspace}")

        # [3] Mean and std over all pixels, channel-wise
        mean_lab_src = np.mean(lab_src, axis=0)
        std_lab_src  = np.std(lab_src, axis=0)
        mean_lab_ref = np.mean(lab_ref, axis=0)
        std_lab_ref  = np.std(lab_ref, axis=0)

        # avoid division by zero
        eps = 1e-6
        std_lab_src_safe = np.where(std_lab_src < eps, eps, std_lab_src)
        ratio = std_lab_ref / std_lab_src_safe

        # [4] Apply global color transfer per pixel
        lab_out = (lab_src - mean_lab_src) * ratio + mean_lab_ref

        # [5] Convert back to RGB if needed
        if opt.colorspace == "lalphabeta":
            out_flat = ColorSpaces.lalphabeta_to_rgb(lab_out)
        elif opt.colorspace == "cielab":
            out_flat = ColorSpaces.cielab_to_rgb(lab_out)
        else:
            out_flat = lab_out

        # [6] Clip to [0,1] and reshape to original shape
        out_flat = np.clip(out_flat, 0.0, 1.0)
        out_colors = out_flat.reshape(orig_shape)

        return out_colors


    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = Reinhard01.__color_transfer(src_color, ref_color, opt)

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
            # print(f"Processing frame {i+1}/{len(src_colors)}")
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_images()[0])

            out_colors = Reinhard01.__color_transfer(src_color, ref_color, opt)

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

            out_colors = Reinhard01.__color_transfer(src_color, ref_color, opt)

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

                out_colors = Reinhard01.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Reinhard01.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Reinhard01.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Reinhard01.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

