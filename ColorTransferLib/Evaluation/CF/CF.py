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
#   Title: Measuring colorfulness in natural images
#   Author: David Hasler, Sabine E. Suesstrunk
#   Published in: ...
#   Year of Publication: 2003
#
# Info:
#   Name: Colorfulness
#   Identifier: CF
#   Link: https://doi.org/10.1117/12.477378
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CF:
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
    def rgb2rgyb(img):
        rg = img[:,:,0] - img[:,:,1]
        yb = 0.5 * (img[:,:,0] + img[:,:,1]) - img[:,:,2]
        return rg, yb

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        img = args[2]
        rg, yb = CF.rgb2rgyb(img.get_raw() * 255)

        mu_rg = np.mean(rg)
        mu_yb = np.mean(yb)

        sig_rg = np.std(rg)
        sig_yb = np.std(yb)

        sig_rgyb = math.sqrt(sig_rg ** 2 + sig_yb ** 2)
        mu_rgyb = math.sqrt(mu_rg ** 2 + mu_yb ** 2)

        M = sig_rgyb + 0.3 * mu_rgyb

        return round(M, 4)