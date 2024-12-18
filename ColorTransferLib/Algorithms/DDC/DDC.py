"""
Copyright 2024 by Herbert Potechius,
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

from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.Helper import check_compatibility
from ColorTransferLib.ImageProcessing.Video import Video
from ColorTransferLib.MeshProcessing.VolumetricVideo import VolumetricVideo
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
    compatibility = {
        "src": ["Image", "Mesh", "PointCloud", "Video", "VolumetricVideo", "LightField", "GaussianSplatting"],
        "ref": ["Image", "Mesh", "PointCloud", "Video", "VolumetricVideo", "LightField", "GaussianSplatting"]
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
            "types": ["Image", "Mesh", "PointCloud", "Video", "VolumetricVideo", "LightField", "GaussianSplatting"],
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):

        model_file_paths = init_model_files("DDC", [opt.model + ".pth"])

        output = {
            "status_code": 0,
            "response": "",
            "object": None,
            "process_time": 0
        }

        start_time = time.time()

        src_img = src.get_raw()

        if src.get_type() == "Image":
            print("Image")
            if "ddcolor_paper_tiny.pth" in model_file_paths:
                model_size = "tiny"
            else:
                model_size = "large"
            pred = Predictor()
            pred.setup(model_file_paths)
            img_out = pred.predict(image=src_img, model_size=model_size)

            out_img = deepcopy(src)
            out_img.set_raw(img_out)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1


        output["process_time"] = time.time() - start_time
        output["object"] = out_img

        return output
