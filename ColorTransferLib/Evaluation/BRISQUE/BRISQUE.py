"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import sys
sys.path.insert(0, '/home/potechius/Projects/VSCode/ColorTransferLib/')

from skimage.metrics import structural_similarity as ssim
import cv2
import math
import numpy as np
from scipy import signal
from ColorTransferLib.ImageProcessing.Image import Image
from libsvm import svm
#from .brisquequality import test_measure_BRISQUE
from brisque import BRISQUE as BRI
import pyiqa
import torch

import os
os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
from oct2py import octave, Oct2Py
from multiprocessing import Process, Pool, Manager, Semaphore

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
        out = args[2]
        img = out.get_raw()

        img_ten = torch.from_numpy(img)
        img_ten = torch.swapaxes(img_ten, 1, 2)
        img_ten = torch.swapaxes(img_ten, 0, 1)
        img_ten = img_ten.unsqueeze(0)

        #print(pyiqa.list_models())
        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        iqa_metric = pyiqa.create_metric('brisque', device=device)
        score_nr = iqa_metric(img_ten)

        score = float(score_nr.cpu().detach().numpy())

        #obj = BRI(url=False)
        #score = obj.score(out.get_raw())

        #score = test_measure_BRISQUE(out.get_raw())#"/media/potechius/Active_Disk/Datasets/ACM-MM-Evaluation-Dataset/abstract/256_abstract-02.png")
        return round(score, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
  

    file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "BCC"
    total_tests = 0
    eval_arr = []
    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        #print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        ssim = BRISQUE.apply(out)
        print(ssim)

        eval_arr.append(ssim)

        with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/brisque.txt","a") as file2:
            file2.writelines(str(round(ssim,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



        # calculate mean
    mean = sum(eval_arr) / len(eval_arr)

    # calculate std
    std = 0
    for t in eval_arr:
        std += math.pow(t-mean, 2)
    std /= len(eval_arr)
    std = math.sqrt(std)


    print("Averaged: " + str(round(mean,3)) + " +- " + str(round(std,3)))

    file1.close()



if __name__ == "__main__":
    main()
