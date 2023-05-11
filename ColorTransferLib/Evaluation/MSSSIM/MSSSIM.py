"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from skimage.metrics import structural_similarity as ssim
import cv2
import math
import numpy as np
from scipy import signal
import torch
import sys
sys.path.insert(0, '/home/potechius/Projects/VSCode/ColorTransferLib/')
from ColorTransferLib.ImageProcessing.Image import Image
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
        src = args[0]
        ref = args[2]
        #mssim = ssim(src.get_raw(), ref.get_raw(), channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
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

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    fuu = ["FCM"]
    for ALG in fuu:
        print(ALG)
        file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
        #ALG = "TPS"
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
            ssim = MSSSIM.apply(src, out)
            eval_arr.append(ssim)
            print(ssim)

            with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/msssim.txt","a") as file2:
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
