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
import phasepack.phasecong as PC

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Structural similarity index measure (SSIM)
# Measuring the perceived quality of an image regarding an original (uncompressed and distortion-free) image.
#
# Source: Image quality assessment: from error visibility to structural similarity
#
# Range [-1, 1]
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class FSIM:
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
    def rgb2yiq(img):
        T_mat = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
        img_yiq = np.dot(img.reshape(-1, 3), T_mat.transpose()).reshape(img.shape)
        return img_yiq

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def phase_congruency(img, c):
        # mult 2 leads to the gabor_scales = [1/6, 1/12, 1/24, 1/48]
        pc2d = PC(img[:, :, c], nscale=4, norient=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pc2d = np.sum(np.asarray(pc2d[4]), axis=0)
        return pc2d

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def gradient_magnitude(img, c):
        src_sharr_x = cv2.Sobel(img[:, :, c], cv2.CV_64F, 1, 0, ksize=3)
        src_sharr_y = cv2.Sobel(img[:, :, c], cv2.CV_64F, 0, 1, ksize=3)
        src_sharr = np.sqrt(np.power(src_sharr_x, 2) + np.power(src_sharr_y, 2))
        return src_sharr
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def similarity_measure(src_pc2d, ref_pc2d, T1):
        SPC = (2 * src_pc2d * ref_pc2d + T1) / (np.power(src_pc2d, 2) + np.power(ref_pc2d, 2) + T1)
        return SPC

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def combine_similarity_measures(SPC, SG):
        alpha = beta = 1.0
        SL = np.power(SPC, alpha) * np.power(SG, beta)
        return SL 
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def fsim_idx(src_pc2d, ref_pc2d, SL, S_C):
        PC_m = np.maximum(src_pc2d, ref_pc2d)
        fsim_val = np.sum(SL * np.power(S_C, 0.03) * PC_m) / np.sum(PC_m)
        return fsim_val 
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref):
        src_img = src.get_raw()
        ref_img = ref.get_raw()

        # 1. RGB -> YIQ
        src_yiq = FSIM.rgb2yiq(src_img)
        ref_yiq = FSIM.rgb2yiq(ref_img)

        # 1. calculate phase congruency (PC)
        src_pc2d_y = FSIM.phase_congruency(src_yiq, 0)
        ref_pc2d_y = FSIM.phase_congruency(ref_yiq, 0)

        # 2. calculate gradient magnitude (GM)
        src_sharr_y = FSIM.gradient_magnitude(src_yiq, 0)
        ref_sharr_y = FSIM.gradient_magnitude(ref_yiq, 0)

        # 3. Similarity measure (PC)
        SPC_y = FSIM.similarity_measure(src_pc2d_y, ref_pc2d_y, 0.85)

        # 4. Similarity measure (GM)
        SG_y = FSIM.similarity_measure(src_sharr_y, ref_sharr_y, 160.0)

        # 5. Combined similarity measure
        SL_y = FSIM.combine_similarity_measures(SPC_y, SG_y)

        # 4. Similarity measure (Chromatic)
        S_i = FSIM.similarity_measure(src_yiq[:,:,1], ref_yiq[:,:,1], 200.0)
        S_q = FSIM.similarity_measure(src_yiq[:,:,2], ref_yiq[:,:,2], 200.0)
        S_C = S_i * S_q

        # 6. FSIM
        fsim_val = FSIM.fsim_idx(src_pc2d_y, ref_pc2d_y, SL_y, S_C)

        return fsim_val

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    # @staticmethod
    # def apply2(src, ref):
    #     mssim = ssim(src.get_raw(), ref.get_raw(), channel_axis=2)
    #     return round(mssim, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    # with open("/media/potechius/Active_Disk/Tests/MetricEvaluation/PDF/fsim.txt","r") as file2:
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
    ALG = "PDF"
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
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        fsim = FSIM.apply(src, out)
        print(fsim)
        eval_arr.append(fsim)

        with open("/media/potechius/Active_Disk/Tests/MetricEvaluation/"+ALG+"/fsim.txt","a") as file2:
            file2.writelines(str(round(fsim,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")

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
