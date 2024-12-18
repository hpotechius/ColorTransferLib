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