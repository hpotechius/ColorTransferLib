"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: -
#   Author: -
#   Published in: - 
#   Year of Publication: -
#
# Abstract:
#   -

# Info:
#   Name: Histogram Intersection
#   Shortname: HI
#   Identifier: HI
#   Link: -
#   Range: [0, 1] with 1 = perfect similarity
#
# Misc:
#   Formula from https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
#
# Implementation Details:
#   usage of $10x10x10$ bins
#   in RGB color space
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class HI:
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
        minimum = np.minimum(histo1, histo2)
        intersection = np.sum(minimum)
        return round(intersection, 4)