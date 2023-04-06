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
from ColorTransferLib.ImageProcessing.Image import Image

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Corr
# ...
#
# Source: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
#
# Range [-1, 1]
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Corr:
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
    def apply(src, ref, bins=[10,10,10]):
        histo1 = src.get_color_statistic_3D(bins=bins, normalized=True)
        histo2 = ref.get_color_statistic_3D(bins=bins, normalized=True)

        histo1_m = np.mean(histo1)
        histo2_m = np.mean(histo2)

        histo1_shift = histo1 - histo1_m
        histo2_shift = histo2 - histo2_m

        nomi = np.sum(np.multiply(histo1_shift, histo2_shift))
        denom = np.sqrt(np.multiply(np.sum(np.power(histo1_shift, 2)),np.sum(np.power(histo2_shift, 2))))

        corr = nomi / denom

        return round(corr, 4)
    
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    file1 = open("/media/hpadmin/Active_Disk/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "GLO"
    total_tests = 0
    eval_arr = []
    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/hpadmin/Active_Disk/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        #print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        corr = Corr.apply(ref, out)
        print(corr)
        eval_arr.append(corr)

        with open("/media/hpadmin/Active_Disk/Tests/MetricEvaluation/"+ALG+"/corr.txt","a") as file2:
            file2.writelines(str(round(corr,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



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