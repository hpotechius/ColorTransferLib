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

from .inference.inference_colorformer import predict

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: ColorFormer: Image Colorization via Color Memory assisted Hybrid-attention Transformer
#   Author: ...
#   Published in: ECCV
#   Year of Publication: 2022
#
# Info:
#   Name: ColorFormer
#   Identifier: CFM
#   Link: https://doi.org/10.1007/978-3-031-19787-1_2
#
# Source:
#   https://github.com/jixiaozhong/ColorFormer ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CFM:
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
            out_obj = CFM.__apply_image(src, opt)
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
        model_file_paths = init_model_files("CFM", ["color_embed_10000.npy", "semantic_embed_10000.npy", "GLH.pth", "net_g_200000.pth"])

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

        out_raw = CFM.__color_transfer(src, opt)

        out_img.set_raw(out_raw)
        outp = out_img
        return outp