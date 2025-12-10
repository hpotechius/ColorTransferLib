"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import time
from copy import deepcopy
import os
import sys


from ColorTransferLib.Utils.Helper import init_model_files
from .third_party.inference_bbox import predict_bbox
from .third_party.test_fusion import predict

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
#   Identifier: Su20
#   Link: https://doi.org/10.48550/arXiv.2005.10825
#
# Source: 
#   https://github.com/ericsujw/InstColorization
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Su20:
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

        # if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField":
        #     output["response"] = "Incompatible reference type."
        #     output["status_code"] = -1
        #     return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = Su20.__apply_image(src, opt)
        else:
            out_obj = None
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src, opt):
        model_file_paths = init_model_files("Su20", ["coco_finetuned_mask_256_ffs_latest_net_G.pth", "coco_finetuned_mask_256_ffs_latest_net_GComp.pth", "coco_finetuned_mask_256_ffs_latest_net_GF.pth",
        "coco_finetuned_mask_256_latest_net_G.pth",
        "coco_finetuned_mask_256_latest_net_GComp.pth",
        "coco_finetuned_mask_256_latest_net_GF.pth",
        "siggraph_retrained_latest_net_G.pth"])

        src_img = src.get_raw()

        # suppress output
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull

        pred_bbox, pred_scores = predict_bbox(src_img)

        opt.A = 2 * opt.ab_max / opt.ab_quant + 1
        opt.B = opt.A

        img_out = predict(src_img, pred_bbox, pred_scores, opt, model_file_paths)

        sys.stdout = old_stdout
        devnull.close()

        return img_out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, opt):

        out_img = deepcopy(src)

        out_raw = Su20.__color_transfer(src, opt)

        out_img.set_raw(out_raw)
        outp = out_img
        return outp