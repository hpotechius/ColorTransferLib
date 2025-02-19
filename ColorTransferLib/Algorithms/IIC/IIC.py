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
from joblib import Parallel, delayed

from ColorTransferLib.Utils.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.Helper import check_compatibility
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo
from ColorTransferLib.Utils.Helper import check_compatibility, init_model_files

from .inference_bbox import predict_bbox
from .test_fusion import predict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Instance-aware Image Colorization
#   Author: Jheng-Wei Su, Hung-Kuo Chu, Jia-Bin Huang
#   Published in: CVPR
#   Year of Publication: 2020
#
# Info:
#   Name: Instance-aware Image Colorization
#   Identifier: IIC
#   Link: https://doi.org/10.48550/arXiv.2005.10825
#
# Source: 
#   https://github.com/ericsujw/InstColorization
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class IIC:
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
            out_obj = IIC.__apply_image(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src, ref, opt):
        model_file_paths = init_model_files("IIC", ["coco_finetuned_mask_256_ffs_latest_net_G.pth", "coco_finetuned_mask_256_ffs_latest_net_GComp.pth", "coco_finetuned_mask_256_ffs_latest_net_GF.pth",
        "coco_finetuned_mask_256_latest_net_G.pth",
        "coco_finetuned_mask_256_latest_net_GComp.pth",
        "coco_finetuned_mask_256_latest_net_GF.pth",
        "siggraph_retrained_latest_net_G.pth"])

        src_img = src.get_raw()

        pred_bbox, pred_scores = predict_bbox(src_img)

        opt.A = 2 * opt.ab_max / opt.ab_quant + 1
        opt.B = opt.A

        img_out = predict(src_img, pred_bbox, pred_scores, opt, model_file_paths)

        return img_out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):

        out_img = deepcopy(src)

        out_raw = IIC.__color_transfer(src, ref, opt)

        out_img.set_raw(out_raw)
        outp = out_img
        return outp