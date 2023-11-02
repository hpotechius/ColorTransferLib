"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import pyiqa
import torch


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: No-Reference Image Quality Assessment in the Spatial Domain
#   Author: Anish Mittal, Anush Krishna Moorthy, Alan Conrad Bovik
#   Published in: IEEE Transactions on Image Processing
#   Year of Publication: 2012
#
# Abstract:
#   We propose a natural scene statistic-based distortion-generic blind/no-reference (NR) image quality assessment 
#   (IQA) model that operates in the spatial domain. The new model, dubbed blind/referenceless image spatial quality 
#   evaluator (BRISQUE) does not compute distortion-specific features, such as ringing, blur, or blocking, but instead 
#   uses scene statistics of locally normalized luminance coefficients to quantify possible losses of “naturalness” in 
#   the image due to the presence of distortions, thereby leading to a holistic measure of quality. The underlying 
#   features used derive from the empirical distribution of locally normalized luminances and products of locally 
#   normalized luminances under a spatial natural scene statistic model. No transformation to another coordinate frame 
#   (DCT, wavelet, etc.) is required, distinguishing it from prior NR IQA approaches. Despite its simplicity, we are 
#   able to show that BRISQUE is statistically better than the full-reference peak signal-to-noise ratio and the 
#   structural similarity index, and is highly competitive with respect to all present-day distortion-generic NR IQA 
#   algorithms. BRISQUE has very low computational complexity, making it well suited for real time applications. 
#   BRISQUE features may be used for distortion-identification as well. To illustrate a new practical application of 
#   BRISQUE, we describe how a nonblind image denoising algorithm can be augmented with BRISQUE in order to perform 
#   blind image denoising. Results show that BRISQUE augmentation leads to performance improvements over 
#   state-of-the-art methods. A software release of BRISQUE is available online: 
#   http://live.ece.utexas.edu/research/quality/BRISQUE_release.zip for public use and evaluation.
#
# Info:
#   Name: Blind/Referenceless Image Spatial Quality Evaluator
#   Shortname: BRISQUE
#   Identifier: BRISQUE
#   Link: https://doi.org/10.1109/TIP.2012.2214050
#   Range: [0, 100] with 100 = perfect quality
#
# Implementation Details:
#   from https://github.com/spmallick/learnopencv
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ... (BRISQUE)
# 
#
# Source: https://github.com/spmallick/learnopencv
#
# Range []
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class BRISQUE:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        out = args[2]
        img = out.get_raw()

        img_ten = torch.from_numpy(img)
        img_ten = torch.swapaxes(img_ten, 1, 2)
        img_ten = torch.swapaxes(img_ten, 0, 1)
        img_ten = img_ten.unsqueeze(0)

        device = torch.device("cpu")
        iqa_metric = pyiqa.create_metric('brisque', device=device)
        score_nr = iqa_metric(img_ten)

        score = float(score_nr.cpu().detach().numpy())

        return round(score, 4)