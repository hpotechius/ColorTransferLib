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
from ColorTransferLib.ImageProcessing.Image import Image
import time

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Image quality assessment: from error visibility to structural similarity
#   Author: Zhou Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli
#   Published in: IEEE Transactions on Image Processing
#   Year of Publication: 2004
#
# Abstract:
#   Objective methods for assessing perceptual image quality traditionally attempted to quantify the visibility of 
#   errors (differences) between a distorted image and a reference image using a variety of known properties of the 
#   human visual system. Under the assumption that human visual perception is highly adapted for extracting structural 
#   information from a scene, we introduce an alternative complementary framework for quality assessment based on the 
#   degradation of structural information. As a specific example of this concept, we develop a structural similarity 
#   index and demonstrate its promise through a set of intuitive examples, as well as comparison to both subjective 
#   ratings and state-of-the-art objective methods on a database of images compressed with JPEG and JPEG2000. A MATLAB 
#   implementation of the proposed algorithm is available online at http://www.cns.nyu.edu//spl sim/lcv/ssim/.
#
# Info:
#   Name: Structural similarity index measure
#   Identifier: SSIM
#   Link: https://doi.org/10.1109/TIP.2003.819861
#   Range: [-1, 1]
#
# Implementation Details:
#   skimage.metrics is used
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class SSIM:
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
        mssim = ssim(src.get_raw(), ref.get_raw(), channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


        # mssim_rr = ssim(src.get_raw()[:,:,0], ref.get_raw()[:,:,0], gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        # mssim_gg = ssim(src.get_raw()[:,:,1], ref.get_raw()[:,:,1], gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        # mssim_bb = ssim(src.get_raw()[:,:,2], ref.get_raw()[:,:,2], gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        # mssim_rgb = 1/3 * mssim_rr + 1/3 * mssim_gg + 1/3 * mssim_bb

        # print(mssim)
        # print(mssim_rgb)
        # exit()
        return round(mssim, 4)


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
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply2(src, ref):
        # testarr = np.array([[[4,2],[3,2],[4,2],[3,2]],
        #                     [[2,2],[1,2],[2,2],[1,2]],
        #                     [[4,2],[3,2],[4,2],[3,2]],
        #                     [[2,2],[1,2],[2,2],[1,2]]])

        # ou = np.lib.stride_tricks.sliding_window_view(testarr, (3,3,2))
        # print(ou)
        # print(ou.shape)

        # exit()

        kernel_gaus = SSIM.iso2Dgauss()

        src_img = src.get_raw()
        ref_img = ref.get_raw()

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

        # Version 1
        # start_time = time.time()
        # for y in range(src_img.shape[0]):
        #     for x in range(src_img.shape[1]):
        #         for c in range(3):
        #             wind_src = src_pad[y:y+11, x:x+11, c]
        #             wind_ref = ref_pad[y:y+11, x:x+11, c]
        #             sig_src[y, x, c] = np.sum(kernel_gaus * (wind_src - np.full((11, 11), mu_src[y,x,c])) ** 2) ** 0.5
        #             sig_ref[y, x, c] = np.sum(kernel_gaus * (wind_ref - np.full((11, 11), mu_ref[y,x,c])) ** 2) ** 0.5
        #             cov[y, x, c] = np.sum(kernel_gaus * (wind_src - np.full((11, 11), mu_src[y,x,c])) * (wind_ref - np.full((11, 11), mu_ref[y,x,c])))
        # print(time.time()-start_time)

        # Version 2
        # start_time = time.time()
        # kernel_gaus_3d = np.concatenate((np.expand_dims(kernel_gaus, 2), np.expand_dims(kernel_gaus, 2), np.expand_dims(kernel_gaus, 2)), 2)
        # for y in range(src_img.shape[0]):
        #     for x in range(src_img.shape[1]):
        #         wind_src = src_pad[y:y+11, x:x+11, :]
        #         wind_ref = ref_pad[y:y+11, x:x+11, :]

        #         mu_src_win = np.tile(mu_src[y, x], (11, 11, 1))
        #         mu_ref_win = np.tile(mu_ref[y, x], (11, 11, 1))

        #         sig_src[y, x] = np.sum(kernel_gaus_3d * (wind_src - mu_src_win) ** 2, axis=(0,1)) ** 0.5
        #         sig_ref[y, x] = np.sum(kernel_gaus_3d * (wind_ref - mu_ref_win) ** 2, axis=(0,1)) ** 0.5
        #         cov[y, x] = np.sum(kernel_gaus_3d * (wind_src - mu_src_win) * (wind_ref - mu_ref_win), axis=(0,1))
        # print(time.time()-start_time)

        # Version 3
        # src_pad_ext.shape = (512, 512, 1, 11, 11, 3)
        start_time = time.time()
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
        print(time.time()-start_time)
 
        l = (2 * mu_src * mu_ref + c1) / (mu_src ** 2 + mu_ref ** 2 + c1)
        c = (2 * sig_src * sig_ref + c2) / (sig_src ** 2 + sig_ref ** 2 + c2)
        s = (cov + c3) / (sig_src * sig_ref + c3)

        ssim_local = l ** alp * c ** bet * s ** gam
        mssim = np.sum(ssim_local, axis=(0,1)) / M
        mssim = mssim[0] * w_r + mssim[1] * w_g + mssim[2] * w_b

        print(mssim)

        rr = SSIM.ssimGPT(src.get_raw()[:,:,0], ref.get_raw()[:,:,0])
        gg = SSIM.ssimGPT(src.get_raw()[:,:,1], ref.get_raw()[:,:,1])
        bb = SSIM.ssimGPT(src.get_raw()[:,:,2], ref.get_raw()[:,:,2])
        ff = 1/3 * rr + 1/3 * gg + 1/3 * bb
        print(ff)

        ret = ssim(src.get_raw()[:,:,0], ref.get_raw()[:,:,0], data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        get = ssim(src.get_raw()[:,:,1], ref.get_raw()[:,:,1], data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        bet = ssim(src.get_raw()[:,:,2], ref.get_raw()[:,:,2], data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        mssim2 = 1/3 * ret + 1/3 * get + 1/3 * bet
        print(mssim2)

        exit()
        return round(mssim, 4)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def gaussian_kernel(sigma):
        # Größe des Filters
        size = 2 * np.ceil(3 * sigma) + 1
        x, y = np.meshgrid(np.arange(-size / 2 + 0.5, size / 2 + 0.5), np.arange(-size / 2 + 0.5, size / 2 + 0.5))
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return kernel / np.sum(kernel)
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def ssimGPT(img1, img2, k1=0.01, k2=0.03, win_size=11, L=1.0, sigma=1.5):
        # Stufenweite des Histogramms
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2

        # Gaussfilter
        kernel = SSIM.gaussian_kernel(sigma)
        window = kernel / np.sum(kernel)

        # Mittelwert
        mu1 = signal.convolve2d(img1, window, mode='same', boundary='symm')
        mu2 = signal.convolve2d(img2, window, mode='same', boundary='symm')

        # Quadrat der Mittelwerte
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Varianz und Kovarianz
        sigma1_sq = signal.convolve2d(img1 ** 2, window, mode='same', boundary='symm') - mu1_sq
        sigma2_sq = signal.convolve2d(img2 ** 2, window, mode='same', boundary='symm') - mu2_sq
        sigma12 = signal.convolve2d(img1 * img2, window, mode='same', boundary='symm') - mu1_mu2

        # SSIM-Formel
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        # Durchschnittlicher SSIM-Wert
        return np.mean(ssim_map)
    
# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
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
        ref = Image(array=ref_img)
        out = Image(array=out_img)
        ssim = SSIM.apply(src, out)

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
