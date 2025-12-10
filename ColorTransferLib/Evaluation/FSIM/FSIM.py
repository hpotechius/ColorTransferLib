"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import numpy as np
import phasepack.phasecong as PC

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: FSIM: A Feature Similarity Index for Image Quality Assessment
#   Author: Lin Zhang, Lei Zhang, Xuanqin Mou, David Zhang
#   Published in: IEEE Transactions on Image Processing 
#   Year of Publication: 2011
#
# Abstract:
#   Image quality assessment (IQA) aims to use computational models to measure the image quality consistently with 
#   subjective evaluations. The well-known structural similarity index brings IQA from pixel- to structure-based stage. 
#   In this paper, a novel feature similarity (FSIM) index for full reference IQA is proposed based on the fact that 
#   human visual system (HVS) understands an image mainly according to its low-level features. Specifically, the phase 
#   congruency (PC), which is a dimensionless measure of the significance of a local structure, is used as the primary 
#   feature in FSIM. Considering that PC is contrast invariant while the contrast information does affect HVS' 
#   perception of image quality, the image gradient magnitude (GM) is employed as the secondary feature in FSIM. PC and 
#   GM play complementary roles in characterizing the image local quality. After obtaining the local quality map, we 
#   use PC again as a weighting function to derive a single quality score. Extensive experiments performed on six 
#   benchmark IQA databases demonstrate that FSIM can achieve much higher consistency with the subjective evaluations 
#   than state-of-the-art IQA metrics.

# Info:
#   Name: Feature Similarity Index
#   Shortname: FSIM
#   Identifier: FSIM
#   Link: https://doi.org/10.1109/TIP.2011.2109730
#   Range: [0, 1]
#
# Implementation Details:
#   For calculating the phase congruency, the 'phasepack'-library from https://github.com/alimuldal/phasepack was used, 
#   which uses a butterworth filter instead of a gaussian.
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class FSIM:
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
    def apply(*args):
        src = args[0]
        ref = args[2]
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

        return round(fsim_val, 4)