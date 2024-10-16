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
from joblib import Parallel, delayed

from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.Helper import check_compatibility
from ColorTransferLib.ImageProcessing.Video import Video
from ColorTransferLib.MeshProcessing.VolumetricVideo import VolumetricVideo


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
class GLO:
    compatibility = {
        "src": ["Image", "Mesh", "PointCloud", "Video", "VolumetricVideo"],
        "ref": ["Image", "Mesh", "PointCloud", "Video", "VolumetricVideo"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Returns basic information of the corresponding publication.
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "GLO",
            "title": "Color Transfer between Images",
            "year": 2001,
            "abstract": "We use a simple statistical analysis to impose one images color characteristics on another. "
                        "We can achieve color correction by choosing an appropriate source image and apply its "
                        "characteristic to another image.",
            "types": ["Image", "Mesh", "PointCloud", "Video", "VolumetricVideo", "LightField"],
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        output = {
            "status_code": 0,
            "response": "",
            "object": None,
            "process_time": 0
        }

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = GLO.apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = GLO.apply_lightfield(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1


        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        #print(output)
        #exit()
        return output


        start_time = time.time()

        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, GLO.compatibility)

        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output
        
        # check if type is video
        out_colors_arr = []
        if src.get_type() == "Video" or src.get_type() == "VolumetricVideo":
            src_colors = src.get_colors()
        else:
            src_colors = [src.get_colors()]

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()

            if src.get_type() == "Video":
                out_img = deepcopy(src.get_images()[0])
            elif src.get_type() == "VolumetricVideo":
                out_img = deepcopy(src.get_meshes()[i])
            else:
                out_img = deepcopy(src)

            out_colors = out_img.get_colors()

            # [2] Convert RGB to lab color space
            if opt.colorspace == "lalphabeta":
                lab_src = ColorSpaces.rgb_to_lab_cpu(src_color)
                lab_ref = ColorSpaces.rgb_to_lab_cpu(ref_color)
            elif opt.colorspace == "rgb":
                lab_src = src_color
                lab_ref = ref_color

            # [3] Get mean, standard deviation and ratio of standard deviations
            mean_lab_src = np.mean(lab_src, axis=(0, 1))
            std_lab_src = np.std(lab_src, axis=(0, 1))
            mean_lab_ref = np.mean(lab_ref, axis=(0, 1))
            std_lab_ref = np.std(lab_ref, axis=(0, 1))

            device_div_std = std_lab_ref / std_lab_src

            # [4] Apply Global Color Transfer
            out_colors[:,:,0] = device_div_std[0] * (lab_src[:,:,0] - mean_lab_src[0]) + mean_lab_ref[0]
            out_colors[:,:,1] = device_div_std[1] * (lab_src[:,:,1] - mean_lab_src[1]) + mean_lab_ref[1]
            out_colors[:,:,2] = device_div_std[2] * (lab_src[:,:,2] - mean_lab_src[2]) + mean_lab_ref[2]

            # [5] Convert lab to RGB color space
            if opt.colorspace == "lalphabeta":
                out_colors = ColorSpaces.lab_to_rgb_cpu(out_colors)

            # [6] Clip color to range [0,1]
            out_colors = np.clip(out_colors, 0, 1)

            out_img.set_colors(out_colors)

            out_colors_arr.append(out_img)

        if src.get_type() == "Video":
            outp = Video(imgs=out_colors_arr)
        elif src.get_type() == "VolumetricVideo":
            outp = VolumetricVideo(meshes=out_colors_arr, file_name=src.get_file_name())
        else:
            outp = out_colors_arr[0]


        output = {
            "status_code": 0,
            "response": "",
            "object": outp,
            "process_time": time.time() - start_time
        }

        return output

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()

        out_img = deepcopy(src)
        out_colors = out_img.get_colors()

        # [2] Convert RGB to lab color space
        if opt.colorspace == "lalphabeta":
            lab_src = ColorSpaces.rgb_to_lab_cpu(src_color)
            lab_ref = ColorSpaces.rgb_to_lab_cpu(ref_color)
        elif opt.colorspace == "rgb":
            lab_src = src_color
            lab_ref = ref_color

        # [3] Get mean, standard deviation and ratio of standard deviations
        mean_lab_src = np.mean(lab_src, axis=(0, 1))
        std_lab_src = np.std(lab_src, axis=(0, 1))
        mean_lab_ref = np.mean(lab_ref, axis=(0, 1))
        std_lab_ref = np.std(lab_ref, axis=(0, 1))

        device_div_std = std_lab_ref / std_lab_src

        # [4] Apply Global Color Transfer
        out_colors[:,:,0] = device_div_std[0] * (lab_src[:,:,0] - mean_lab_src[0]) + mean_lab_ref[0]
        out_colors[:,:,1] = device_div_std[1] * (lab_src[:,:,1] - mean_lab_src[1]) + mean_lab_ref[1]
        out_colors[:,:,2] = device_div_std[2] * (lab_src[:,:,2] - mean_lab_src[2]) + mean_lab_ref[2]

        # [5] Convert lab to RGB color space
        if opt.colorspace == "lalphabeta":
            out_colors = ColorSpaces.lab_to_rgb_cpu(out_colors)

        # [6] Clip color to range [0,1]
        out_colors = np.clip(out_colors, 0, 1)

        out_img.set_colors(out_colors)
        outp = out_img

        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply_lightfield(src, ref, opt):
        src_lightfield_array = src.get_image_array()
        out = deepcopy(src)
        out_lightfield_array = out.get_image_array()

        for row in range(src.get_grid_size()[0]):
            for col in range(src.get_grid_size()[1]):
                src_color = src_lightfield_array[row][col].get_colors()
                ref_color = ref.get_colors()

                out_colors = out_lightfield_array[row][col].get_colors()

                # [2] Convert RGB to lab color space
                if opt.colorspace == "lalphabeta":
                    lab_src = ColorSpaces.rgb_to_lab_cpu(src_color)
                    lab_ref = ColorSpaces.rgb_to_lab_cpu(ref_color)
                elif opt.colorspace == "rgb":
                    lab_src = src_color
                    lab_ref = ref_color

                # [3] Get mean, standard deviation and ratio of standard deviations
                mean_lab_src = np.mean(lab_src, axis=(0, 1))
                std_lab_src = np.std(lab_src, axis=(0, 1))
                mean_lab_ref = np.mean(lab_ref, axis=(0, 1))
                std_lab_ref = np.std(lab_ref, axis=(0, 1))

                device_div_std = std_lab_ref / std_lab_src

                # [4] Apply Global Color Transfer
                out_colors[:,:,0] = device_div_std[0] * (lab_src[:,:,0] - mean_lab_src[0]) + mean_lab_ref[0]
                out_colors[:,:,1] = device_div_std[1] * (lab_src[:,:,1] - mean_lab_src[1]) + mean_lab_ref[1]
                out_colors[:,:,2] = device_div_std[2] * (lab_src[:,:,2] - mean_lab_src[2]) + mean_lab_ref[2]

                # [5] Convert lab to RGB color space
                if opt.colorspace == "lalphabeta":
                    out_colors = ColorSpaces.lab_to_rgb_cpu(out_colors)

                # [6] Clip color to range [0,1]
                out_colors = np.clip(out_colors, 0, 1)

                out_lightfield_array[row][col].set_colors(out_colors)

        return out