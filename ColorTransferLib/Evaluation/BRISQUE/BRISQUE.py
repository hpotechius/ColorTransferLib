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
import numpy as np
import cv2


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
    def __init__(self, model_path, range_path):
        # Load SVM
        self.svm = cv2.ml.SVM_load(model_path)

        # Load range file
        fr = cv2.FileStorage(range_path, cv2.FILE_STORAGE_READ)
        range_mat = fr.getNode("range").mat()
        fr.release()

        # Extract feature_min and feature_max
        self.feature_min = range_mat[0, :].astype(np.float32)
        self.feature_max = range_mat[1, :].astype(np.float32)
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