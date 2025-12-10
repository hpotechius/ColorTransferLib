"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import time
import os
from copy import deepcopy
from sys import platform
from contextlib import contextmanager
import sys
from io import StringIO

# NOTE:
# oct2py expects a *host* octave executable (e.g. /usr/bin/octave-cli).
# The Flatpak package org.octave.Octave does NOT expose octave-cli on the host,
# so you cannot just point OCTAVE_EXECUTABLE to a flatpak path.
# To use the MATLAB/Octave backend, install Octave natively (e.g. `sudo apt install octave`)
# and set OCTAVE_EXECUTABLE to the corresponding binary (usually `/usr/bin/octave-cli`).

# if platform == "linux" or platform == "linux2":
#     # linux (requires native Octave installation, not flatpak)
#     os.environ["OCTAVE_EXECUTABLE"] = "/usr/bin/octave-cli"
# elif platform == "darwin":
#     # OS X (e.g. via Homebrew)
#     os.environ["OCTAVE_EXECUTABLE"] = "/opt/homebrew/bin/octave-cli"
# elif platform == "win32":
#     # Windows: configure OCTAVE_EXECUTABLE to point to octave-cli.exe if installed
#     pass

# from oct2py import octave

from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo
from .l2e_tps_color_transfer import L2EColorTransferPaper

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: L2 Divergence for robust colour transfer
#   Author: Mairéad Grogan, Rozenn Dahyot
#   Published in: Computer Vision and Image Understanding
#   Year of Publication: 2019

# Info:
#   Name: TpsColorTransfer
#   Identifier: Grogan19
#   Link: https://doi.org/10.1016/j.cviu.2019.02.002
#   Source: https://github.com/groganma/gmm-colour-transfer
#
# Implementation Details:
#   Usage of Octave to run the Matlab-Scripts
#   Clustering is done using KMeans because MVQ does not work in Octave
#   Internal image resizing (mg applyK-Means.m) to 300x350px for clustering
#   Remove largescale and TolCon option in gmmregrbfl2.m because unrecognized
# 
# Note:
#   The Octave Forge package repository is no longer actively maintained. 
#   Please find Octave Packages at https://packages.octave.org. 
#   pkg install "https://downloads.sourceforge.net/project/octave/Octave%20Forge%20Packages/Individual%20Package%20Releases/image-2.14.0.tar.gz"
#   pkg install "https://github.com/gnu-octave/statistics/archive/refs/tags/release-1.7.0.tar.gz"
#   ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Grogan19:
    # ------------------------------------------------------------------------------------------------------------------
    # Checks source and reference compatibility
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        output = {
            "status_code": 0,
            "response": "",
            "object": None,
            "process_time": 0
        }

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField" or ref.get_type() == "GaussianSplatting" or ref.get_type() == "PointCloud":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = Grogan19.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = Grogan19.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = Grogan19.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = Grogan19.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = Grogan19.__apply_mesh(src, ref, opt)
        else:
            out_obj = None
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
 
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    # @staticmethod
    # def __color_transfer_matlab(src_img, ref_img, opt):
    #     # NOTE: sudo add-apt-repository ppa:ubuntuhandbook1/octave
    #     # NOTE: sudo apt update
    #     # NOTE: sudo apt install octave

    #     # NOTE: pkg install -forge image
    #     # NOTE: pkg install -forge statistics
    #     # NOTE: mex -g ./mex_GaussTransform.c ./GaussTransform.c 
    #     # NOTE: mex -g ./mex_mgRecolourParallel_1.cpp

    #     # Preprocessing
    #     # NOTE RGB space needs multiplication with 255
    #     src_img = src_img * 255
    #     ref_img = ref_img * 255

    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     octave.addpath(octave.genpath(os.path.join(current_dir, 'third_party')))

    #     octave.eval('warning("off", "all")')
    #     octave.eval('pkg load image')
    #     octave.eval('pkg load statistics')
    #     outp = octave.ctfunction(ref_img, src_img, opt.cluster_method, opt.cluster_num, opt.colorspace)

    #     outp = outp.astype(np.float32)

    #     return outp
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_img, ref_img, opt):
        model = L2EColorTransferPaper(
            K=opt.cluster_num,
            grid_size=opt.grid_size,
            lambda_w=1e-3,
            bandwidth_alpha=0.3,
            anneal_factor=0.5,
            anneal_stages=3,
            device=None,
        )

        model.fit_from_numpy(src_img, ref_img, n_iters=opt.iterations, lr=opt.learning_rate, verbose=True)

        out_colors = model.transfer_numpy(src_img)

        out_colors = out_colors.astype(np.float32)

        return out_colors
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        # out_colors = Grogan19.__color_transfer_matlab(src_img, ref_img, opt)
        out_colors = Grogan19.__color_transfer(src_img, ref_img, opt)

        out_img.set_raw(out_colors)

        outp = out_img
        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_video(src, ref, opt): 
        # check if type is video
        out_raw_arr = []
        src_raws = src.get_raw()

        for i, src_raw in enumerate(src_raws):
            # Preprocessing
            ref_raw = ref.get_raw()
            out_img = deepcopy(src.get_images()[0])

            out_colors = Grogan19.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_raw(out_colors)
            out_raw_arr.append(out_img)

        outp = Video(imgs=out_raw_arr, fps=src.get_fps())

        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_volumetricvideo(src, ref, opt): 
        out_raw_arr = []
        src_raws = src.get_raw()

        for i, src_raw in enumerate(src_raws):
            # Preprocessing
            ref_raw = ref.get_raw()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = Grogan19.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_raw(out_colors)
            out_raw_arr.append(out_img)
            outp = VolumetricVideo(meshes=out_raw_arr, file_name=src.get_file_name())

        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_lightfield(src, ref, opt):
        src_lightfield_array = src.get_image_array()
        out = deepcopy(src)
        out_lightfield_array = out.get_image_array()

        for row in range(src.get_grid_size()[0]):
            for col in range(src.get_grid_size()[1]):
                src_raw = src_lightfield_array[row][col].get_raw()
                ref_raw = ref.get_raw()

                out_colors = Grogan19.__color_transfer(src_raw, ref_raw, opt)

                out_lightfield_array[row][col].set_raw(out_colors)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = Grogan19.__color_transfer(src_img, ref_img, opt)

        out_img.set_raw(out_colors)
        outp = out_img
        return outp

