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

from .predict import Predictor

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: ...
#   Author: ...
#   Published in: ...
#   Year of Publication: ...
#
# Info:
#   Name: ...
#   Identifier: DDC
#   Link: ...
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class DDC:
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
            out_obj = DDC.__apply_image(src, ref, opt)
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
        model_file_paths = init_model_files("DDC", [opt.model + ".pth"])

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

        out_raw = DDC.__color_transfer(src, ref, opt)

        out_img.set_raw(out_raw)
        outp = out_img
        return outp