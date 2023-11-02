"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Corr
# ...
#
# Source: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
#
# Range [-1, 1]
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Corr:
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

        histo1_m = np.mean(histo1)
        histo2_m = np.mean(histo2)

        histo1_shift = histo1 - histo1_m
        histo2_shift = histo2 - histo2_m

        nomi = np.sum(np.multiply(histo1_shift, histo2_shift))
        denom = np.sqrt(np.multiply(np.sum(np.power(histo1_shift, 2)),np.sum(np.power(histo2_shift, 2))))

        corr = nomi / denom

        return round(corr, 4)