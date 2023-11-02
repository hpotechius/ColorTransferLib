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
import os
from copy import deepcopy
from sys import platform

from ColorTransferLib.Utils.Helper import check_compatibility

if platform == "linux" or platform == "linux2":
    # linux
    os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
elif platform == "darwin":
    # OS X
    os.environ["OCTAVE_EXECUTABLE"] = "/opt/homebrew/bin/octave-cli"
elif platform == "win32":
    # Windows...
    pass

from oct2py import octave

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: L2 Divergence for robust colour transfer
#   Author: Mairéad Grogan, Rozenn Dahyot
#   Published in: Computer Vision and Image Understanding
#   Year of Publication: 2019
#
# Abstract:
#   Optimal Transport (OT) is a very popular framework for performing colour transfer in images and videos. We have
#   proposed an alternative framework where the cost function used for inferring a parametric transfer function is
#   defined as the robust L2 divergence between two probability density functions Grogan and Dahyot (2015). In this
#   paper, we show that our approach combines many advantages of state of the art techniques and outperforms many
#   recent algorithms as measured quantitatively with standard quality metrics, and qualitatively using perceptual
#   studies Grogan and Dahyot (2017). Mathematically, our formulation is presented in contrast to the OT cost function
#   that shares similarities with our cost function. Our formulation, however, is more flexible as it allows colour
#   correspondences that may be available to be taken into account and performs well despite potential occurrences of
#   correspondence outlier pairs. Our algorithm is shown to be fast, robust and it easily allows for user interaction
#   providing freedom for artists to fine tune the recoloured images and videos Grogan et al. (2017).
#
# Info:
#   Name: TpsColorTransfer
#   Identifier: TPS
#   Link: https://doi.org/10.1016/j.cviu.2019.02.002
#   Source: https://github.com/groganma/gmm-colour-transfer
#
# Implementation Details:
#   Usage of Octave to run the Matlab-Scripts
#   Clustering is done using KMeans because MVQ does not work in Octave
#   Internal image resizing (mg applyK-Means.m) to 300x350px for clustering
#   Remove largescale and TolCon option in gmmregrbfl2.m because unrecognized
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class TPS:
    identifier = "TpsColorTransfer"
    title = "L2 Divergence for robust colour transfer"
    year = 2019
    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # HOST METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "TpsColorTransfer",
            "title": "L2 Divergence for robust colour transfer",
            "year": 2019,
            "abstract": "Optimal Transport (OT) is a very popular framework for performing colour transfer in images "
                        "and videos. We have proposed an alternative framework where the cost function used for "
                        "inferring a parametric transfer function is defined as the robust L2 divergence between two "
                        "probability density functions Grogan and Dahyot (2015). In this paper, we show that our "
                        "approach combines many advantages of state of the art techniques and outperforms many recent "
                        "algorithms as measured quantitatively with standard quality metrics, and qualitatively using "
                        "perceptual studies Grogan and Dahyot (2017). Mathematically, our formulation is presented in "
                        "contrast to the OT cost function that shares similarities with our cost function. Our "
                        "formulation, however, is more flexible as it allows colour correspondences that may be "
                        "available to be taken into account and performs well despite potential occurrences of "
                        "correspondence outlier pairs. Our algorithm is shown to be fast, robust and it easily allows "
                        "for user interaction providing freedom for artists to fine tune the recoloured images and "
                        "videos Grogan et al. (2017).",
            "types": ["Image"]
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()
        # NOTE: sudo apt-get install liboctave-dev
        # NOTE: pkg install -forge image
        # NOTE: pkg install -forge statistics

        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, TPS.compatibility)

        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output

        # Preprocessing
        # NOTE RGB space needs multiplication with 255
        src_img = src.get_raw() * 255
        ref_img = ref.get_raw() * 255
        out_img = deepcopy(src)

        # mex -g  mex_mgRecolourParallel_1.cpp COMPFLAGS="/openmp $COMPFLAGS"
        octave.addpath(octave.genpath('.'))
        #octave.addpath(octave.genpath('module/Algorithms/TpsColorTransfer/L2RegistrationForCT'))
        octave.eval('pkg load image')
        octave.eval('pkg load statistics')

        outp = octave.ctfunction(ref_img, src_img, opt.cluster_method, opt.cluster_num, opt.colorspace)
        outp = outp.astype(np.float32)
        out_img.set_raw(outp)
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output