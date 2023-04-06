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
# Entropy
# ...
#
# Source: Experimental Tests of Image Fusion for Night Vision
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class EP:
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
        histo1, _, _ = src.get_color_statistic(normalized=True)
        histo2, _, _ = ref.get_color_statistic(normalized=True)
        ep = 0
        for c in range(3):
            ep -= 1/3 * np.sum(histo1[:,c]*np.log2(histo1[:,c]))
        print(ep)
        exit()
        minimum = np.minimum(histo1, histo2)
        intersection = np.sum(minimum)
        #intersection = np.true_divide(np.sum(minima), np.sum(histo2))
        return round(intersection, 4)
    
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
        entropy = EP.apply(ref, out)
        print(entropy)
        eval_arr.append(entropy)

        with open("/media/hpadmin/Active_Disk/Tests/MetricEvaluation/"+ALG+"/ep.txt","a") as file2:
            file2.writelines(str(round(entropy,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



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