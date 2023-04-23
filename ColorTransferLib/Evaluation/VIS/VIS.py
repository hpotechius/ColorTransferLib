"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from skimage.metrics import structural_similarity as ssim
import os
os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
from oct2py import octave, Oct2Py
import cv2
import math
import numpy as np
from scipy import signal
from ColorTransferLib.ImageProcessing.Image import Image
#import pysaliency

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Structural similarity index measure (VIS)
# ...
#
# Source: VSI: A visual saliency-induced index for perceptual image quality assessment
#
# Range [??, 1]
#
# Note: recompile mex files -> see "gbvs_compile.m"
# change #include <matrix.h> to <Matrix.h>
#
# Sources:
# http://www.animaclock.com/harel/share/gbvs.php
# https://github.com/Pinoshino/gbvs
#
# Good to know:
# https://ajcr.net/Basic-guide-to-einsum/
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class VIS:
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
    def initOcatve():
        # mex -g  mex_mgRecolourParallel_1.cpp COMPFLAGS="/openmp $COMPFLAGS"
        octave.addpath(octave.genpath('.'))
        #octave.addpath(octave.genpath('module/Algorithms/TpsColorTransfer/L2RegistrationForCT'))
        octave.eval('pkg load image')
        octave.eval('pkg load statistics')
        octave.eval("dir")

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def VS(img):
        outp = octave.gbvs((img * 255).astype("uint8"))
        sal_map = outp["master_map_resized"]
        return sal_map
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rgb2lmn(img):
        M = np.array([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]])
        out = np.einsum("ij, klj -> kli", M, img)
        return out
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def G(img):
        kernel_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]]) / 16.0
        kernel_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]) / 16.0

        schaar_x = cv2.filter2D(img, -1, kernel_x)
        schaar_y = cv2.filter2D(img, -1, kernel_y)

        schaar_out = np.sqrt(abs(schaar_x) ** 2 + abs(schaar_y) ** 2)
        return schaar_out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def S_SV(VS1, VS2, C1):
        nomi = 2 * VS1 * VS2 + C1
        denomi = (abs(VS1) ** 2) + (abs(VS2) ** 2) + C1
        SSV = nomi / denomi
        return SSV

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def S_G(G1, G2, C2):
        nomi = 2 * G1 * G2 + C2
        denomi = (abs(G1) ** 2) + (abs(G2) ** 2) + C2
        SG = nomi / denomi
        return SG

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def S_C(M1, M2, N1, N2, C3):
        nomi1 = 2 * M1 * M2 + C3
        denomi1 = (abs(M1) ** 2) + (abs(M2) ** 2) + C3
        X1 = nomi1 / denomi1

        nomi2 = 2 * N1 * N2 + C3
        denomi2 = (abs(N1) ** 2) + (abs(N2) ** 2) + C3
        X2 = nomi2 / denomi2

        SC = X1 * X2
        return SC

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def S(SSV, SG, SC, a, b):
        S_val = SSV * (abs(SG) ** a) * (abs(SC) ** b)
        return S_val

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def VS_m(VS1, VS2):
        VS_m = np.maximum(VS1, VS2)
        return VS_m

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def VIS(S_val, VS_m):
        VSI_val = np.sum(S_val * VS_m) / np.sum(VS_m)
        return VSI_val


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref):
        VIS.initOcatve()
        C1 = 0.00001
        C2 = 0.00001
        C3 = 0.00001
        a = 0.4
        b = 0.02

        src_img = src.get_raw()
        ref_img = ref.get_raw()

        src_lmn = VIS.rgb2lmn(src_img)
        ref_lmn = VIS.rgb2lmn(ref_img)

        L1 = src_lmn[:,:,0]
        L2 = ref_lmn[:,:,0]
        M1 = src_lmn[:,:,1]
        M2 = ref_lmn[:,:,1]
        N1 = src_lmn[:,:,2]
        N2 = ref_lmn[:,:,2]

        VS1 = VIS.VS(src_img)
        VS2 = VIS.VS(ref_img)
        VS_m = VIS.VS_m(VS1, VS2)

        G1 = VIS.G(L1)
        G2 = VIS.G(L2)

        SSV = VIS.S_SV(VS1, VS2, C1)
        SG = VIS.S_G(G1, G2, C2)
        SC = VIS.S_C(M1, M2, N1, N2, C3)
        S_val = VIS.S(SSV, SG, SC, a, b)

        VIS_val = VIS.VIS(S_val, VS_m)

        print(VIS_val)
        #cv2.imwrite("/home/potechius/Downloads/sal2.png",src_schaar*255)

        exit()
        return round(0, 4)

  
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
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        ssim = VIS.apply(src, src)

        eval_arr.append(ssim)

        with open("/media/potechius/Active_Disk/Tests/MetricEvaluation/"+ALG+"/ssim.txt","a") as file2:
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
