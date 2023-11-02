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
import torch
from copy import deepcopy

from ColorTransferLib.Algorithms.HIS.models.models import create_model
from ColorTransferLib.Algorithms.HIS.data.data_loader import CreateDataLoader
from ColorTransferLib.Utils.Helper import check_compatibility

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Deep Color Transfer using Histogram Analogy
#   Author: Junyong Lee, Hyeongseok Son, Gunhee Lee, Jonghyeop Lee, Sunghyun Cho, Seungyong Lee
#   Published in: The Visual Computer: International Journal of Computer Graphics, Volume 36, Issue 10-12Oct 2020
#   Year of Publication: 2020
#
# Abstract:
#   We propose a novel approach to transferring the color of a reference image to a given source image. Although there
#   can be diverse pairs of source and reference images in terms of content and composition similarity, previous methods
#   are not capable of covering the whole diversity. To resolve this limitation, we propose a deep neural network that
#   leverages color histogram analogy for color transfer. A histogram contains essential color information of an image,
#   and our network utilizes the analogy between the source and reference histograms to modulate the color of the source
#   image with abstract color features of the reference image. In our approach, histogram analogy is exploited basically
#   among the whole images, but it can also be applied to semantically corresponding regions in the case that the source
#   and reference images have similar contents with different compositions. Experimental results show that our approach
#   effectively transfers the reference colors to the source images in a variety of settings. We also demonstrate a few
#   applications of our approach, such as palette-based recolorization, color enhancement, and color editing.
#
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
    identifier = "HIS"
    title = "Deep Color Transfer using Histogram Analogy"
    year = 2020

    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "HistogramAnalogy",
            "title": "Deep Color Transfer using Histogram Analogy",
            "year": 2020,
            "abstract": "We propose a novel approach to transferring the color of a reference image to a given source "
                        "image. Although there can be diverse pairs of source and reference images in terms of content "
                        "and composition similarity, previous methods are not capable of covering the whole diversity. "
                        "To resolve this limitation, we propose a deep neural network that leverages color histogram "
                        "analogy for color transfer. A histogram contains essential color information of an image, and "
                        "our network utilizes the analogy between the source and reference histograms to modulate the "
                        "color of the source image with abstract color features of the reference image. In our "
                        "approach, histogram analogy is exploited basically among the whole images, but it can also be "
                        "applied to semantically corresponding regions in the case that the source and reference "
                        "images have similar contents with different compositions. Experimental results show that our "
                        "approach effectively transfers the reference colors to the source images in a variety of "
                        "settings. We also demonstrate a few applications of our approach, such as palette-based "
                        "recolorization, color enhancement, and color editing.",
            "types": ["Image"]
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, HIS.compatibility)

        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output
        
        if not torch.cuda.is_available():
            opt.gpu_ids = [-1]

        # Preprocessing
        srcT = src.get_raw()
        refT = ref.get_raw()
        out_img = deepcopy(src)

        opt.checkpoints_dir = "Models/HIS"

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

        out = out.astype(np.float32)
        out_img.set_raw(out, normalized=True)
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }
        return output
