"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import math
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Novel multi-color transfer algorithms and quality measure
#   Author: Karen Panetta, Long Bao, Sos Agaian
#   Published in: IEEE Transactions on Consumer Electronics
#   Year of Publication: 2016
#
# Abstract:
#   In this paper, two new image multi-color transfer algorithms for still images and image sequences are proposed. 
#   These methods can be used to capture the artistic ambience or "mood" of the source image and transfer that same 
#   color ambience to the target image. The performance and effectiveness of these new algorithms are demonstrated 
#   through simulations and comparisons to other state of the art methods, including Alla's, Reinhard's and Pitie's 
#   methods. These algorithms are straight-forward, automatic, and suitable for various practical recoloring 
#   applications, including coloring, color correction, animation and color restoration for imaging tools and consumer 
#   products. This work is also useful for fast implementation of special effects for the entertainment industry and 
#   reduces manual labor costs for these types of tasks. Another contribution of this paper is the introduction of a 
#   new color transfer quality measure. The new measure is highly consistent with human perception, even compared to 
#   other current color transfer quality measures such as Xiao's measure and Xiang's measure.
#
# Info:
#   Name: Color Transfer Quality Measure
#   Shortname: CTQM
#   Identifier: CTQM
#   Link: https://doi.org/10.1109/TCE.2016.7613196
#   Range: [0, ??]
#
# Implementation Details:
#   nan values can appear -> are ignored for evaluation
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CTQM:
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
    def ssim(src, ref):
        kernel_gaus = CTQM.iso2Dgauss()

        src_img = src
        ref_img = ref

        hi, wi = src_img.shape

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
        
        mu_src = cv2.filter2D(src_img, -1, kernel_gaus)
        mu_ref = cv2.filter2D(ref_img, -1, kernel_gaus)

        src_pad = np.pad(src_img, ((5, 5), (5, 5)), 'reflect')
        ref_pad = np.pad(ref_img, ((5, 5), (5, 5)), 'reflect')

        sig_src = np.zeros_like(src_img)
        sig_ref = np.zeros_like(ref_img)
        cov = np.zeros_like(src_img)

        # Version 3
        # src_pad_ext.shape = (512, 512, 1, 11, 11, 3)
        src_pad_ext = np.lib.stride_tricks.sliding_window_view(src_pad, (11,11)).squeeze()
        ref_pad_ext = np.lib.stride_tricks.sliding_window_view(ref_pad, (11,11)).squeeze()

        # mu_src.shape = (512, 512)
        # mu_src_win_ext.shape = (512, 512, 11, 11)
        mu_src_win = np.expand_dims(mu_src, (2,3))
        mu_src_win_ext = np.tile(mu_src_win, (1, 1, 11, 11))
        mu_ref_win = np.expand_dims(mu_ref, (2,3))
        mu_ref_win_ext = np.tile(mu_ref_win, (1, 1, 11, 11))

        kernel_gaus = np.expand_dims(kernel_gaus, (0,1))
        kernel_gaus_ext = np.tile(kernel_gaus, (hi, wi, 1, 1))

        src_pad_ext_norm = src_pad_ext - mu_src_win_ext
        ref_pad_ext_norm = ref_pad_ext - mu_ref_win_ext
        sig_src = np.sum(kernel_gaus_ext * src_pad_ext_norm ** 2, axis=(2,3)) ** 0.5
        sig_ref = np.sum(kernel_gaus_ext * ref_pad_ext_norm ** 2, axis=(2,3)) ** 0.5
        cov = np.sum(kernel_gaus_ext * src_pad_ext_norm * ref_pad_ext_norm, axis=(2,3))
 
        l = (2 * mu_src * mu_ref + c1) / (mu_src ** 2 + mu_ref ** 2 + c1)
        c = (2 * sig_src * sig_ref + c2) / (sig_src ** 2 + sig_ref ** 2 + c2)
        s = (cov + c3) / (sig_src * sig_ref + c3)

        #ssim_local = l ** alp * c ** bet * s ** gam
        #mssim = np.sum(ssim_local, axis=(0,1)) / M

        return l, c, s
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def gradient_img(img):
        grad_x = cv2.Sobel(img[:,:], cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img[:,:], cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        sobel_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return sobel_mag
    
    # ------------------------------------------------------------------------------------------------------------------
    # R1 regions: preserved edge pixel region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R1_region(ssim_loc, src_edge, ref_edge, T1):
        R1 = ssim_loc[:,:][(src_edge[:,:] > T1) & (ref_edge[:,:] > T1)]
        return R1


    # ------------------------------------------------------------------------------------------------------------------
    # R2 regions: changed edge pixel region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R2_region(ssim_loc, src_edge, ref_edge, T1):
        R2 = ssim_loc[:,:][(src_edge[:,:] > T1) & (ref_edge[:,:] <= T1) | 
                                (ref_edge[:,:] > T1) & (src_edge[:,:] <= T1)]
        return R2

    # ------------------------------------------------------------------------------------------------------------------
    # R3 regions: Smooth region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R3_region(ssim_loc, src_edge, ref_edge, T1, T2):
        R3 = ssim_loc[:,:][(src_edge[:,:] < T2) & (ref_edge[:,:] > T1)]
        return R3

    # ------------------------------------------------------------------------------------------------------------------
    # R4 regions: Rexture region
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def R4_region(ssim_loc, src_edge, ref_edge, T1, T2):
        R4 = ssim_loc[:,:][~((src_edge[:,:] > T1) & (ref_edge[:,:] > T1)) &
                                ~((src_edge[:,:] > T1) & (ref_edge[:,:] <= T1) | (ref_edge[:,:] > T1) & (src_edge[:,:] <= T1)) &
                                ~((src_edge[:,:] < T2) & (ref_edge[:,:] > T1))]
        return R4
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def ivegssim(src, ref):
        src_img = src
        ref_img = ref

        alp = bet = gam = 1.0

        # apply Sobel filter for each channel
        src_edge_mag = CTQM.gradient_img(src_img)
        ref_edge_mag = CTQM.gradient_img(ref_img)

        l, _, _ = CTQM.ssim(src_img, ref_img)
        _, c, s = CTQM.ssim(np.log(src_edge_mag+1), np.log(ref_edge_mag+1))
        ssim_local = l ** alp * c ** bet * s ** gam

        T1 = 0.12 * np.max(src_edge_mag)
        T2 = 0.06 * np.max(src_edge_mag)

        
        R1 = CTQM.R1_region(ssim_local, src_edge_mag, ref_edge_mag, T1)
        R2 = CTQM.R2_region(ssim_local, src_edge_mag, ref_edge_mag, T1)
        R3 = CTQM.R3_region(ssim_local, src_edge_mag, ref_edge_mag, T1, T2)
        R4 = CTQM.R4_region(ssim_local, src_edge_mag, ref_edge_mag, T1, T2)

        # setting of weights
        #           w1      w2      w3      w4
        # if R1={}  0.00    0.50    0.25    0.25
        # if R2={}  0.50    0.00    0.25    0.25
        # else      0.25    0.25    0.25    0.25
        ivssim_val = 0.0
        w3 = w4 = 0.25

        if R1.shape[0] == 0:
            w1 = 0.0
            w2 = 0.5
        elif R2.shape[0] == 0:
            w1 = 0.5
            w2 = 0.0
        else:
            w1 = 0.25
            w2 = 0.25
    
        if R1.shape[0] != 0.0:
            ivssim_val += np.sum(R1) / R1.shape[0] * w1
        if R2.shape[0] != 0.0:
            ivssim_val += np.sum(R2) / R2.shape[0] * w2
        if R3.shape[0] != 0.0:
            ivssim_val += np.sum(R3) / R3.shape[0] * w3
        if R4.shape[0] != 0.0:
            ivssim_val += np.sum(R4) / R4.shape[0] * w4
        
        return ivssim_val
    
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
        rg, yb = CTQM.rgb2rgyb(img * 255)

        mu_rg = np.mean(rg)
        mu_yb = np.mean(yb)

        sig_rg = np.std(rg)
        sig_yb = np.std(yb)

        rg_val = np.log(sig_rg ** 2 / np.abs(mu_rg) ** 0.2)
        yb_val = np.log(sig_yb ** 2 / np.abs(mu_yb) ** 0.2)

        cf = 0.2 * rg_val * yb_val

        return cf
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def colormapsimilarity(ref, out):
        # round a and b channel to integers
        ref = np.around(ref).astype(np.int32)
        out = np.around(out).astype(np.int32)

        # reshape from (512, 512, 2) to (262144, 2)
        ref = ref.reshape(ref.shape[0] * ref.shape[1], ref.shape[2])
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])

        # get normalized 2D histogram of a and b channel
        bins = [125, 125]
        #print(ref.shape)
        ref_histo = np.asarray(np.histogramdd(ref, bins)[0])
        out_histo = np.asarray(np.histogramdd(out, bins)[0])

        ref_histo /= np.sum(ref_histo)
        out_histo /= np.sum(out_histo)

        cms = -np.sum(np.abs(out_histo-ref_histo))

        return cms


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = args[0]
        ref = args[1]
        out = args[2]
        # RGB to CIELab
        src_lab = cv2.cvtColor(src.get_raw(), cv2.COLOR_RGB2LAB)
        ref_lab = cv2.cvtColor(ref.get_raw(), cv2.COLOR_RGB2LAB)
        out_lab = cv2.cvtColor(out.get_raw(), cv2.COLOR_RGB2LAB)

        # Caluclate 4-EGSSIM of L channel
        # multiple with 2.55 to get range [0, 255] instead of [0, 100]
        ivegssim_val = CTQM.ivegssim(out_lab[:,:,0], src_lab[:,:,0])

        # Calculate Colorfulness
        cf_val = CTQM.colorfulness(out.get_raw())

        # Calculate Color Map Similarity
        cms_val = CTQM.colormapsimilarity(ref_lab[:,:,1:3], out_lab[:,:,1:3])

        # Calculate CTQM
        wo = 1.0
        ws = 100.0
        wm = 1.0

        ctqm_val = wo * cf_val + wm * cms_val + ws *ivegssim_val

        return round(ctqm_val, 4)