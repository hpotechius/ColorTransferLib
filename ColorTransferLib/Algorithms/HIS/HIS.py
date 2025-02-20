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
import torch
from copy import deepcopy
import os

from ColorTransferLib.Algorithms.HIS.models.models import create_model
from ColorTransferLib.Algorithms.HIS.data.data_loader import CreateDataLoader
from ColorTransferLib.Utils.Helper import check_compatibility, init_model_files, get_cache_dir
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Deep Color Transfer using Histogram Analogy
#   Author: Junyong Lee, Hyeongseok Son, Gunhee Lee, Jonghyeop Lee, Sunghyun Cho, Seungyong Lee
#   Published in: The Visual Computer: International Journal of Computer Graphics, Volume 36, Issue 10-12Oct 2020
#   Year of Publication: 2020

# Info:
#   Name: HistogramAnalogy
#   Identifier: HIS
#   Link: https://doi.org/10.1007/s00371-020-01921-6
#   Sources: https://github.com/codeslake/Color_Transfer_Histogram_Analogy
#
# Implementation Details:
#   Restriction of max 700x700px was removed
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class HIS:
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
            out_obj = HIS.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = HIS.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = HIS.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = HIS.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = HIS.__apply_mesh(src, ref, opt)
        else:
            out_obj = None
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_img, ref_img, opt):
        if not torch.cuda.is_available():
            opt.gpu_ids = [-1]

        # Preprocessing
        srcT = src_img
        refT = ref_img

        init_model_files("HIS", ["latest_net_C_A.pth", "latest_net_G_A.pth"])
        opt.checkpoints_dir = os.path.join(get_cache_dir(), "HIS")

        data_loader = CreateDataLoader(opt, srcT, refT)
        dataset = data_loader.load_data()

        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        model = create_model(opt)
        opt.is_psnr = True

        model.set_input(dataset[0])
        model.test()

        visuals = model.get_current_visuals()
        ou = visuals["03_output"]
        ou = np.swapaxes(ou, 0, 1)
        ou = np.swapaxes(ou, 1, 2)

        out = ou.cpu().detach().numpy()

        out = out.astype(np.float32) * 255.0
        # out_img.set_raw(out, normalized=True)
        # output = {
        #     "status_code": 0,
        #     "response": "",
        #     "object": out_img,
        #     "process_time": time.time() - start_time
        # }
        return out
    
       
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = HIS.__color_transfer(src_img, ref_img, opt)
        out_img.set_raw(out_colors)

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

            out_colors = HIS.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_raw(out_colors)
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

            out_colors = HIS.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_raw(out_colors)
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

                out_colors = HIS.__color_transfer(src_raw, ref_raw, opt)

                out_lightfield_array[row][col].set_raw(out_colors)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = HIS.__color_transfer(src_img, ref_img, opt)

        out_img.set_raw(out_colors)
        outp = out_img
        return outp
