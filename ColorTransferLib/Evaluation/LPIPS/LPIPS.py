"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import math
import numpy as np
import lpips
import torch
import sys
import os
sys.path.insert(0, '/home/potechius/Projects/VSCode/ColorTransferLib/')
from ColorTransferLib.ImageProcessing.Image import Image

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Learned Perceptual Image Patch Simiality (LPIPS)
# 
#
# Source: https://github.com/richzhang/PerceptualSimilarity
# Paper: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
#
# Range [0, ?] -> Value of 0 means perfect similarity
# 
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class LPIPS:
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
        src_img = src.get_raw()
        ref_img = ref.get_raw()

        # image should be RGB, IMPORTANT: normalized to [-1,1]
        src_img_norm = torch.from_numpy(src_img * 2 - 1)
        ref_img_norm = torch.from_numpy(ref_img * 2 - 1)
        #print(src_img_norm.shape)

        src_img_norm = torch.swapaxes(src_img_norm, 1, 2)
        src_img_norm = torch.swapaxes(src_img_norm, 0, 1)
        src_img_norm = src_img_norm.unsqueeze(0)

        ref_img_norm = torch.swapaxes(ref_img_norm, 1, 2)
        ref_img_norm = torch.swapaxes(ref_img_norm, 0, 1)
        ref_img_norm = ref_img_norm.unsqueeze(0)

        # prevent printing
        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w")

        loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
        lp = loss_fn_alex(src_img_norm, ref_img_norm).detach().numpy().squeeze()

        sys.stdout = old_stdout # reset old stdout
        
        return float(lp)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    #fuu = ["FUZ", "TPS","PDF","MKL","HIS" "NST", "CAM", "DPT", "RHG", "BCC"]
    fuu = ["FCM"]
    for ALG in fuu:
        print(ALG)
        file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
        #ALG = "GLO"
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
            mse = LPIPS.apply(src, out)
            print(mse)
            #exit()
            eval_arr.append(mse)

            with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/lpips.txt","a") as file2:
                file2.writelines(str(round(mse,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



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
