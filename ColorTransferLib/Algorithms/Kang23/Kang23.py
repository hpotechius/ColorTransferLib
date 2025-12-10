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

from ColorTransferLib.Utils.Helper import init_model_files
from .third_party.predict import Predictor

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders
#   Author: Xiaoyang Kang, Tao Yang, Wenqi Ouyang, Peiran Ren, Lingzhi Li, Xuansong Xie
#   Published in: ...
#   Year of Publication: 2023
#
# Info:
#   Name: DDColor
#   Identifier: Kang23
#   Link: ...
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Kang23:
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
            out_obj = Kang23.__apply_image(src, ref, opt)
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
        model_file_paths = init_model_files("Kang23", [opt.model + ".pth"])

        src_img = src.get_raw()

        if "ddcolor_paper_tiny.pth" in model_file_paths:
            model_size = "tiny"
        else:
            model_size = "large"
        pred = Predictor()
        pred.setup(model_file_paths)
        img_out = pred.predict(image=src_img, model_size=model_size)

        out_img = deepcopy(src)

        return img_out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):

        out_img = deepcopy(src)

        out_raw = Kang23.__color_transfer(src, ref, opt)

        out_img.set_raw(out_raw)
        outp = out_img
        return outp