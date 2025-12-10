"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import torch
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Multi-scale Structural Similarity for Image Quality Assessment
#   Author: Zhou Wang, Eero P. Simoncelli, Alan C. Bovik
#   Published in: IEEE Asilomar Conference on Signals, Systems and Computers
#   Year of Publication: 2003
#
# Abstract:
#   The structural similarity image quality assessment approach is based on the assumption that the human visual system 
#   is highly adapted for extracting structural information from the scene, and therefore a measure of structural 
#   similarity can provide a good approximation to perceived image quality. This paper proposes a novel multi-scale 
#   structural similarity method, which supplies more flexibility than single-scale methods in incorporating the 
#   variations of image resolution and viewing condition. Experimental comparisons demonstrate the effectiveness of the 
#   proposed method.
#
# Info:
#   Name: Multi-scale Structural Similarity
#   Shotname: MS-SSIM
#   Identifier: MSSSIM
#   Link: https://doi.org/10.1109/ACSSC.2003.1292216
#   Range: [0, 1]
#
# Implementation Details:
#   implementation from torchmetrics
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class MSSSIM:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = args[0]
        ref = args[2]
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        ten_src = torch.from_numpy(src.get_raw())
        ten_ref = torch.from_numpy(ref.get_raw())

        ten_src = torch.swapaxes(ten_src, 0, 2)
        ten_src = torch.swapaxes(ten_src, 1, 2)
        ten_src = torch.unsqueeze(ten_src, 0)


        ten_ref = torch.swapaxes(ten_ref, 0, 2)
        ten_ref = torch.swapaxes(ten_ref, 1, 2)
        ten_ref = torch.unsqueeze(ten_ref, 0)

        ms_val = ms_ssim(ten_src, ten_ref)
        ms_val = float(ms_val.numpy())

        return round(ms_val, 4)