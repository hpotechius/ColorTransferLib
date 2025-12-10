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

from ColorTransferLib.Utils.Helper import init_model_files
from .third_party.inference.inference_colorformer import predict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: ColorFormer: Image Colorization via Color Memory assisted Hybrid-attention Transformer
#   Author: Xiaozhong Ji, Boyuan Jiang, Donghao Luo, Guangpin Tao, Wenqing Chu, Zhifeng Xie, Chengjie Wang, Ying Tai
#   Published in: ECCV
#   Year of Publication: 2022
#
# Info:
#   Name: ColorFormer
#   Identifier: Ji22
#   Link: https://doi.org/10.1007/978-3-031-19787-1_2
#
# Source:
#   https://github.com/jixiaozhong/ColorFormer ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Ji22:
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

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = Ji22.__apply_image(src, opt)
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
        model_file_paths = init_model_files("Ji22", ["color_embed_10000.npy", "semantic_embed_10000.npy", "GLH.pth", "net_g_200000.pth"])

        #print(model_file_paths)
        src_img = src.get_raw()

        out_img = predict(model_file_paths, src_img)

        return out_img

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, opt):

        out_img = deepcopy(src)

        out_raw = Ji22.__color_transfer(src, opt)

        out_img.set_raw(out_raw)
        outp = out_img
        return outp