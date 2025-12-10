"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""


import pyiqa
import torch

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ... (NIQE)
# 
#
# Source: https://github.com/idealo/image-quality-assessment
# https://github.com/chaofengc/Awesome-Image-Quality-Assessment
#
# Range []
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class NIQE:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        img = args[2]
        img = img.get_raw()

        img_ten = torch.from_numpy(img)
        img_ten = torch.swapaxes(img_ten, 1, 2)
        img_ten = torch.swapaxes(img_ten, 0, 1)
        img_ten = img_ten.unsqueeze(0)

        device = "cpu"
        iqa_metric = pyiqa.create_metric('niqe', device=device)
        score_nr = iqa_metric(img_ten)

        score_nr = float(score_nr.cpu().detach().numpy())

        return round(score_nr, 4)