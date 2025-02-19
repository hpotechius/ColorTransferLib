"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import math
import numpy as np

from ColorTransferLib.DataTypes.Image import Image

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Bhattacharyya distance (BA)
# ...
#
# Source: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
#
# Range [0, ??] -> 0 means perfect similarity
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class BA:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = args[1]
        ref = args[2]
        bins=[10,10,10]
        histo1 = src.get_color_statistic_3D(bins=bins, normalized=True)
        histo2 = ref.get_color_statistic_3D(bins=bins, normalized=True)

        num_bins = np.prod(bins)

        histo1_m = np.mean(histo1)
        histo2_m = np.mean(histo2)

        ba_l = 1 / np.sqrt(histo1_m * histo2_m * math.pow(num_bins, 2))
        ba_r = np.sum(np.sqrt(np.multiply(histo1, histo2)))
        ba = math.sqrt(1 - ba_l * ba_r)

        return round(ba, 4)