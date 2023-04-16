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
import os
import numpy as np
import json
from ColorTransferLib.ImageProcessing.Image import Image
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
    def apply(img):
        img = img.get_raw()

        img_ten = torch.from_numpy(img)
        img_ten = torch.swapaxes(img_ten, 1, 2)
        img_ten = torch.swapaxes(img_ten, 0, 1)
        img_ten = img_ten.unsqueeze(0)

        #print(pyiqa.list_models())
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        iqa_metric = pyiqa.create_metric('niqe', device=device)
        score_nr = iqa_metric(img_ten)

        score_nr = float(score_nr.cpu().detach().numpy())

        return round(score_nr, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    file1 = open("/media/potechius/Active_Disk/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "GLO"
    total_tests = 0
    eval_arr = []
    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/potechius/Active_Disk/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        out = Image(array=out_img)
        mse = NIQE.apply(out)
        print(mse)
        exit()
        eval_arr.append(mse)

        with open("/media/potechius/Active_Disk/Tests/MetricEvaluation/"+ALG+"/nima.txt","a") as file2:
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
