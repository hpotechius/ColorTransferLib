"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import os
from sys import platform

# if platform == "linux" or platform == "linux2":
#     # linux
#     os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
# elif platform == "darwin":
#     # OS X
#     os.environ["OCTAVE_EXECUTABLE"] = "/opt/homebrew/bin/octave-cli"
# elif platform == "win32":
#     # Windows...
#     pass

# from oct2py import octave
import cv2
import numpy as np
# from .third_party.gbvs import compute_gbvs_saliency
from .third_party.saliency_models import gbvs as gbvs_module

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Structural similarity index measure (VSI)
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
# https://github.com/shreelock/gbvs
#
# Good to know:
# https://ajcr.net/Basic-guide-to-einsum/
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class VSI:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def initOcatve():
        # gbvs_install_path = os.path.join(os.getcwd(), "ColorTransferLib/Evaluation/VSI/gbvs/gbvs_install.m")
        
        # current_path = os.getcwd()
        # gbvs_path = os.path.join(current_path, "ColorTransferLib/Evaluation/VSI/gbvs")


        # octave.addpath(octave.genpath(gbvs_path))
        # octave.eval(f"cd '{gbvs_path}'")
        # octave.eval("warning('off','Octave:shadowed-function')")
        # octave.eval('pkg load image')
        # octave.eval('pkg load statistics')

        # octave.eval(f"source('{gbvs_install_path}')")
        # octave.eval("dir")

        # mex -g  mex_mgRecolourParallel_1.cpp COMPFLAGS="/openmp $COMPFLAGS"
        # Necessary to run "gbvs_install" once on a new system

        pass

        # octave.addpath(octave.genpath('.'))
        # octave.eval("warning('off','Octave:shadowed-function')")
        # octave.eval('pkg load image')
        # octave.eval('pkg load statistics')

    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def VS(img):
        # outp = gbvs.compute_saliency((img * 255).astype("uint8"))
        # return outp

        outp = gbvs_module.compute_saliency(img)
        return outp

    
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
    def VSI(S_val, VS_m):
        VSI_val = np.sum(S_val * VS_m) / np.sum(VS_m)
        return VSI_val

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(*args):
        src = args[0]
        ref = args[2]
        
        # VSI.initOcatve()
        C1 = 0.00001
        C2 = 0.00001
        C3 = 0.00001
        a = 0.4
        b = 0.02

        src_img = src.get_raw()
        ref_img = ref.get_raw()

        src_lmn = VSI.rgb2lmn(src_img)
        ref_lmn = VSI.rgb2lmn(ref_img)

        L1 = src_lmn[:,:,0]
        L2 = ref_lmn[:,:,0]
        M1 = src_lmn[:,:,1]
        M2 = ref_lmn[:,:,1]
        N1 = src_lmn[:,:,2]
        N2 = ref_lmn[:,:,2]

        VS1 = VSI.VS(src_img)
        VS2 = VSI.VS(ref_img)
        VS_m = VSI.VS_m(VS1, VS2)

        G1 = VSI.G(L1)
        G2 = VSI.G(L2)

        SSV = VSI.S_SV(VS1, VS2, C1)
        SG = VSI.S_G(G1, G2, C2)
        SC = VSI.S_C(M1, M2, N1, N2, C3)
        S_val = VSI.S(SSV, SG, SC, a, b)

        VSI_val = VSI.VSI(S_val, VS_m)

        return round(VSI_val, 4)