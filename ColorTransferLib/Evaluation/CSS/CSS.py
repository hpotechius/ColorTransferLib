"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

from skimage.metrics import structural_similarity as ssim
from torchmetrics import StructuralSimilarityIndexMeasure
import cv2
import math
import numpy as np
from scipy import signal
import sys
sys.path.insert(0, '/home/potechius/Projects/VSCode/ColorTransferLib/')
from ColorTransferLib.ImageProcessing.Image import Image
import time
#import pysaliency
from multiprocessing import Process, Pool, Manager, Semaphore

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Colorfulness Structure Similarity (CSS)
# 
#
# Source: Selective color transfer with multi-source images
#
# Range [..]
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CSS:
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
    def iso2Dgauss(size=11, sig=1.5, normalized=True):
        kernel = np.zeros((size,size))
        shift = size // 2
        for x in range(size): 
            for y in range(size): 
                x_s = x - shift
                y_s = y - shift
                kernel[y, x] = (1.0 / (2 * math.pi * math.pow(sig, 2))) * np.exp( -(math.pow(x_s,2)+math.pow(y_s,2)) / (2*math.pow(sig,2)) )
        
        if normalized:
            kernel /= np.sum(kernel)
        return kernel

    # ------------------------------------------------------------------------------------------------------------------
    # the CSS uses a 8x8 windows intead of a 11x11
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def ssim(src, ref):
        kernel_gaus = CSS.iso2Dgauss()

        src_img = src
        ref_img = ref

        k1 = 0.01
        k2 = 0.03
        L = 1.0
        N = 11
        M = src_img.shape[0] * src_img.shape[1]
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        c3 = c2 / 2.0
        alp = 1.0
        bet = 1.0
        gam = 1.0
        w_r = 1.0/3.0
        w_g = 1.0/3.0
        w_b = 1.0/3.0
        
        mu_src = cv2.filter2D(src_img, -1, kernel_gaus)
        mu_ref = cv2.filter2D(ref_img, -1, kernel_gaus)


        src_pad = np.pad(src_img, ((5, 5), (5, 5), (0, 0)), 'reflect')
        ref_pad = np.pad(ref_img, ((5, 5), (5, 5), (0, 0)), 'reflect')

        sig_src = np.zeros_like(src_img)
        sig_ref = np.zeros_like(ref_img)
        cov = np.zeros_like(src_img)

        # Version 3
        # src_pad_ext.shape = (512, 512, 1, 11, 11, 3)
        src_pad_ext = np.lib.stride_tricks.sliding_window_view(src_pad, (11,11,3)).squeeze()
        ref_pad_ext = np.lib.stride_tricks.sliding_window_view(ref_pad, (11,11,3)).squeeze()

        # mu_src.shape = (512, 512, 3)
        # mu_src_win_ext.shape = (512, 512, 11, 11, 3)
        mu_src_win = np.expand_dims(mu_src, (2,3))
        mu_src_win_ext = np.tile(mu_src_win, (1, 1, 11, 11, 1))
        mu_ref_win = np.expand_dims(mu_ref, (2,3))
        mu_ref_win_ext = np.tile(mu_ref_win, (1, 1, 11, 11, 1))

        kernel_gaus_3d = np.concatenate((np.expand_dims(kernel_gaus, 2), np.expand_dims(kernel_gaus, 2), np.expand_dims(kernel_gaus, 2)), 2)

        kernel_gaus_3d = np.expand_dims(kernel_gaus_3d, (0,1))
        kernel_gaus_3d_ext = np.tile(kernel_gaus_3d, (512, 512, 1, 1, 1))

        src_pad_ext_norm = src_pad_ext - mu_src_win_ext
        ref_pad_ext_norm = ref_pad_ext - mu_ref_win_ext
        sig_src = np.sum(kernel_gaus_3d_ext * src_pad_ext_norm ** 2, axis=(2,3)) ** 0.5
        sig_ref = np.sum(kernel_gaus_3d_ext * ref_pad_ext_norm ** 2, axis=(2,3)) ** 0.5
        cov = np.sum(kernel_gaus_3d_ext * src_pad_ext_norm * ref_pad_ext_norm, axis=(2,3))
 
        l = (2 * mu_src * mu_ref + c1) / (mu_src ** 2 + mu_ref ** 2 + c1)
        c = (2 * sig_src * sig_ref + c2) / (sig_src ** 2 + sig_ref ** 2 + c2)
        s = (cov + c3) / (sig_src * sig_ref + c3)

        ssim_local = l ** alp * c ** bet * s ** gam
        mssim = np.sum(ssim_local, axis=(0,1)) / M
        mssim = mssim[0] * w_r + mssim[1] * w_g + mssim[2] * w_b

        return l, c, s
    
    # ------------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def gssim(src, ref):
        src_img = src
        ref_img = ref

        l, _, _ = CSS.ssim(src_img, ref_img)

        # apply Sobel filter for each channel
        src_edge_mag = np.zeros((512, 512, 0))
        ref_edge_mag = np.zeros((512, 512, 0))
        for c in range(3):
            src_grad_x = cv2.Sobel(src_img[:,:,c], cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            src_grad_y = cv2.Sobel(src_img[:,:,c], cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            src_sobel_mag = np.sqrt(src_grad_x ** 2 + src_grad_y ** 2)
            src_edge_mag = np.concatenate((src_edge_mag, np.expand_dims(src_sobel_mag, 2)), 2)

            ref_grad_x = cv2.Sobel(ref_img[:,:,c], cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            ref_grad_y = cv2.Sobel(ref_img[:,:,c], cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            ref_sobel_mag = np.sqrt(ref_grad_x ** 2 + ref_grad_y ** 2)
            ref_edge_mag = np.concatenate((ref_edge_mag, np.expand_dims(ref_sobel_mag, 2)), 2)

        _, c, s = CSS.ssim(src_edge_mag, ref_edge_mag)

        M = src_img.shape[0] * src_img.shape[1]
        alp = bet = gam = 1.0
        w_r = w_g = w_b = 1.0/3.0
        gssim_local = l ** alp * c ** bet * s ** gam
        mgssim = np.sum(gssim_local, axis=(0,1)) / M
        mgssim = mgssim[0] * w_r + mgssim[1] * w_g + mgssim[2] * w_b
   
        return mgssim
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rgb2rgyb(img):
        rg = img[:,:,0] - img[:,:,1]
        yb = 0.5 * (img[:,:,0] + img[:,:,1]) - img[:,:,2]
        return rg, yb

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def colorfulness(img):
        rg, yb = CSS.rgb2rgyb(img * 255)

        mu_rg = np.mean(rg)
        mu_yb = np.mean(yb)

        sig_rg = np.std(rg)
        sig_yb = np.std(yb)

        sig_rgyb = math.sqrt(sig_rg ** 2 + sig_yb ** 2)
        mu_rgyb = math.sqrt(mu_rg ** 2 + mu_yb ** 2)

        C = sig_rgyb + 0.3 * mu_rgyb

        return C
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, out):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = out.get_raw()

        # calculate colorfulness similarity
        ref_cf = CSS.colorfulness(ref_img)
        out_cf = CSS.colorfulness(out_img)

        cs = np.abs(out_cf - ref_cf)

        # calculate GSSIM
        ss = CSS.gssim(out_img, src_img)

        # calculate colorfulness structure similarity
        A4 = 1.0
        css_val = (ss + A4) / (cs + A4)

        return round(css_val, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "BCC"
    total_tests = 0
    eval_arr = []

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    max_processes = 20
    sem = Semaphore(max_processes)

    def task(i, return_dict, sem):
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
        ssim = CSS.apply(src, ref, out)
        return_dict[i] = (ssim, str(round(ssim,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")
        sem.release()

    for line in file1.readlines():
        total_tests += 1

        sem.acquire() # acquire semaphore before starting a new process
        process = Process(target=task, args=(total_tests, return_dict, sem))
        jobs.append(process)
        process.start()

        print(total_tests)

        # if total_tests == 20:
        #     break

    for i, proc in enumerate(jobs):
        print(i)
        proc.join()

    print("DONE")

    for i, val in enumerate(return_dict.values()):
        print(i)
        eval_arr.append(val[0])
        with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/css.txt","a") as file2:
           file2.writelines(val[1])

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

    exit()





    file1 = open("/media/potechius/Backup_00/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "GLO"
    total_tests = 0
    eval_arr = []
    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        ssim = CSS.apply(src, ref, out)

        eval_arr.append(ssim)

        with open("/media/potechius/Backup_00/Tests/MetricEvaluation/"+ALG+"/css.txt","a") as file2:
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
