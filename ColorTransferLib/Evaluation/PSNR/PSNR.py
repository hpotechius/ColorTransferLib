"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import math
import numpy as np
from ColorTransferLib.ImageProcessing.Image import Image

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Peak Signal-to-Noise Ratio (PSNR)
# ...
#
# Source: ...
#
# Range: []
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PSNR:
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
    def apply(src, ref):
        # check if images are completely equal
        difference = cv2.subtract(src.get_raw(), ref.get_raw())
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            return 9999

        psnrval = psnr(src.get_raw(), ref.get_raw())
        return round(psnrval, 4)

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
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        out = Image(array=out_img)
        psnr = PSNR.apply(src, out)
        print(psnr)
        eval_arr.append(psnr)

        with open("/media/hpadmin/Active_Disk/Tests/MetricEvaluation/"+ALG+"/psnr.txt","a") as file2:
            file2.writelines(str(round(psnr,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



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
