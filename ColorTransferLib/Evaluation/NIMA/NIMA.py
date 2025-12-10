"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import copy
import tensorflow as tf

from .third_party.predict import predict
from .third_party.utils.utils import calc_mean_score
from .third_party.handlers.model_builder import Nima

from ColorTransferLib.Utils.Helper import init_model_files

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: NIMA: Neural Image Assessment
#   Author: Hossein Talebi, Peyman Milanfar
#   Published in: IEEE Transactions on Image Processing
#   Year of Publication: 2018
#
# Abstract:
#   Automatically learned quality assessment for images has recently become a hot topic due to its usefulness in a wide 
#   variety of applications, such as evaluating image capture pipelines, storage techniques, and sharing media. Despite 
#   the subjective nature of this problem, most existing methods only predict the mean opinion score provided by data 
#   sets, such as AVA and TID2013. Our approach differs from others in that we predict the distribution of human opinion 
#   scores using a convolutional neural network. Our architecture also has the advantage of being significantly simpler 
#   than other methods with comparable performance. Our proposed approach relies on the success (and retraining) of 
#   proven, state-of-the-art deep object recognition networks. Our resulting network can be used to not only score 
#   images reliably and with high correlation to human perception, but also to assist with adaptation and optimization 
#   of photo editing/enhancement algorithms in a photographic pipeline. All this is done without need for a “golden” 
#   reference image, consequently allowing for single-image, semantic- and perceptually-aware, no-reference quality 
#   assessment.
#
# Info:
#   Name: Neural Image Assessment
#   Shortname: NIMA
#   Identifier: NIMA
#   Link: https://doi.org/10.1109/TIP.2018.2831899
#   Source: https://github.com/idealo/image-quality-assessment
#   Range: [0, 10] with 10 = best quality
#
# Implementation Details:
#   Usage of MobileNet
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class NIMA:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        img = args[2]
        imcopy = copy.deepcopy(img)
        imcopy.resize(224, 224)
        imcopy = imcopy.get_raw()

        base_model_name = "MobileNet"

        model_file_paths = init_model_files("NIMA", [
            "weights_mobilenet_aesthetic_0.07.hdf5"
        ])
        weights_file = model_file_paths["weights_mobilenet_aesthetic_0.07.hdf5"]


        # suppress output
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull

        # build model and load weights
        nima = Nima(base_model_name, weights=None)
        nima.build()
        nima.nima_model.load_weights(weights_file)

        # get predictions
        predictions = predict(nima.nima_model, imcopy)
        nim = calc_mean_score(predictions[0])

        sys.stdout = old_stdout
        devnull.close()


        tf.keras.backend.clear_session()

        return round(nim, 4)