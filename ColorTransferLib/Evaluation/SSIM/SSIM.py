"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from skimage.metrics import structural_similarity as ssim


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Image quality assessment: from error visibility to structural similarity
#   Author: Zhou Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli
#   Published in: IEEE Transactions on Image Processing
#   Year of Publication: 2004
#
# Abstract:
#   Objective methods for assessing perceptual image quality traditionally attempted to quantify the visibility of 
#   errors (differences) between a distorted image and a reference image using a variety of known properties of the 
#   human visual system. Under the assumption that human visual perception is highly adapted for extracting structural 
#   information from a scene, we introduce an alternative complementary framework for quality assessment based on the 
#   degradation of structural information. As a specific example of this concept, we develop a structural similarity 
#   index and demonstrate its promise through a set of intuitive examples, as well as comparison to both subjective 
#   ratings and state-of-the-art objective methods on a database of images compressed with JPEG and JPEG2000. A MATLAB 
#   implementation of the proposed algorithm is available online at http://www.cns.nyu.edu//spl sim/lcv/ssim/.
#
# Info:
#   Name: Structural similarity index measure
#   Identifier: SSIM
#   Link: https://doi.org/10.1109/TIP.2003.819861
#   Range: [-1, 1]
#
# Implementation Details:
#   skimage.metrics is used
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class SSIM:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = args[0]
        ref = args[2]
        mssim = ssim(src.get_raw(), ref.get_raw(), channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

        return round(mssim, 4)