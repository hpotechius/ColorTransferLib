"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""


import math
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: ...
#   Author: Anil Kumar Bhattacharyya
#   Published in: ...
#   Year of Publication: ...
#
# Info:
#   Name: Bhattacharyya Distance
#   Identifier: BD
#   Link: ...
#   Range [0, 1] -> 0 means perfect similarity
#
# Implementation Details:
#   from https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class BD:
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

        # ba_l is always 1 because of normalization, but we keep it for completeness
        ba_l = 1 / np.sqrt(histo1_m * histo2_m * math.pow(num_bins, 2))
        ba_r = np.sum(np.sqrt(np.multiply(histo1, histo2)))
        ba = math.sqrt(1 - ba_l * ba_r)

        return round(ba, 4)