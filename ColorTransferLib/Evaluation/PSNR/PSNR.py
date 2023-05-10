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
    def apply(*args):
        src = args[0]
        ref = args[2]
        src_img = src.get_raw()
        ref_img = ref.get_raw()

        num_pix = src_img.shape[0] * src_img.shape[1]

        mse_c = np.sum(np.power(np.subtract(src_img, ref_img), 2), axis=(0,1)) / num_pix
        mse = np.sum(mse_c) / 3

        # print(mse)
        psnr_v = 10 * math.log10(math.pow(1, 2) / mse)
        # print(psnr_v)

        # check if images are completely equal
        # difference = cv2.subtract(src.get_raw(), ref.get_raw())
        # b, g, r = cv2.split(difference)
        # if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        #     return 9999

        # psnrval = psnr(src.get_raw(), ref.get_raw())
        return round(psnr_v, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    # with open("/media/potechius/Active_Disk/Tests/MetricEvaluation/CAM/psnr.txt","r") as file2:
    #     cc = 0
    #     summ = 0
    #     for line in file2.readlines():
    #         tim = float(line.strip().split(" ")[0])
    #         if math.isinf(tim):
    #             continue
    #         summ += tim
    #         cc += 1
    #     summ /= cc
    #     print(cc)
    #     print(summ)
    # exit()    


    file1 = open("/media/potechius/Active_Disk/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "FCM"
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
        psnr = PSNR.apply(src, out)
        print(psnr)
        eval_arr.append(psnr)

        with open("/media/potechius/Active_Disk/Tests/MetricEvaluation/"+ALG+"/psnr.txt","a") as file2:
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
