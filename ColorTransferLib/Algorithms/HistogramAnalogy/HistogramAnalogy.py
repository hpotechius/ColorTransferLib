"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from numba import cuda
import math
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Algorithms.HistogramAnalogy.models.models import create_model
from ColorTransferLib.Algorithms.HistogramAnalogy.data.data_loader import CreateDataLoader
import cv2
import torch
from PIL import Image
import torchvision.utils as vutils
import sys

from ColorTransferLib.Utils.BaseOptions import BaseOptions
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Deep Color Transfer using Histogram Analogy
#   Author: Junyong Lee, Hyeongseok Son, Gunhee Lee, Jonghyeop Lee, Sunghyun Cho, Seungyong Lee
#   Published in: The Visual Computer: International Journal of Computer Graphics, Volume 36, Issue 10-12Oct 2020
#   Year of Publication: 2020
#
# Abstract:
#   We propose a novel approach to transferring the color of a reference image to a given source image. Although there
#   can be diverse pairs of source and reference images in terms of content and composition similarity, previous methods
#   are not capable of covering the whole diversity. To resolve this limitation, we propose a deep neural network that
#   leverages color histogram analogy for color transfer. A histogram contains essential color information of an image,
#   and our network utilizes the analogy between the source and reference histograms to modulate the color of the source
#   image with abstract color features of the reference image. In our approach, histogram analogy is exploited basically
#   among the whole images, but it can also be applied to semantically corresponding regions in the case that the source
#   and reference images have similar contents with different compositions. Experimental results show that our approach
#   effectively transfers the reference colors to the source images in a variety of settings. We also demonstrate a few
#   applications of our approach, such as palette-based recolorization, color enhancement, and color editing.
#
# Link: https://doi.org/10.1007/s00371-020-01921-6
#
# Sources:
#   https://github.com/codeslake/Color_Transfer_Histogram_Analogy
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class HistogramAnalogy:
    identifier = "HistogramAnalogy"
    title = "Deep Color Transfer using Histogram Analogy"
    year = 2020

    compatibility = {
        "src": ["Image"],
        "ref": ["Image"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # HOST METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "HistogramAnalogy",
            "title": "Deep Color Transfer using Histogram Analogy",
            "year": 2020,
            "abstract": "We propose a novel approach to transferring the color of a reference image to a given source "
                        "image. Although there can be diverse pairs of source and reference images in terms of content "
                        "and composition similarity, previous methods are not capable of covering the whole diversity. "
                        "To resolve this limitation, we propose a deep neural network that leverages color histogram "
                        "analogy for color transfer. A histogram contains essential color information of an image, and "
                        "our network utilizes the analogy between the source and reference histograms to modulate the "
                        "color of the source image with abstract color features of the reference image. In our "
                        "approach, histogram analogy is exploited basically among the whole images, but it can also be "
                        "applied to semantically corresponding regions in the case that the source and reference "
                        "images have similar contents with different compositions. Experimental results show that our "
                        "approach effectively transfers the reference colors to the source images in a variety of "
                        "settings. We also demonstrate a few applications of our approach, such as palette-based "
                        "recolorization, color enhancement, and color editing."
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, HistogramAnalogy.compatibility)

        # Preprocessing
        srcT = src.get_raw()
        refT = ref.get_raw()
        out_img = deepcopy(src)

        opt.checkpoints_dir = "Models/HistogramAnalogy"

        data_loader = CreateDataLoader(opt, srcT, refT)
        dataset = data_loader.load_data()

        # set gpu ids
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        model = create_model(opt)
        opt.is_psnr = True

        model.set_input(dataset[0])
        model.test()

        visuals = model.get_current_visuals()
        ou = visuals["03_output"]
        ou = np.swapaxes(ou, 0, 1)
        ou = np.swapaxes(ou, 1, 2)

        out = ou.cpu().detach().numpy()

        out = out.astype(np.float32)
        out_img.set_raw(out, normalized=True)
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img
        }
        return output

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Options Class
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    class Options:
        def __init__(self):
            # Base Options
            self.dataroot = "test"  # path to images (should have subfolders trainA, trainB, valA, valB, etc)'
            self.batchSize = 1  # input batch size
            self.loadSize = 286  # scale images to this size
            self.fineSize = 256  # then crop to this size
            self.ngf = 64  # num. of gen filters in first conv layer
            self.ndf = 64  # num. of discrim filters in first conv layer
            self.which_model_netD = "basic"  # selects model to use for netD
            self.which_model_netG = "resnet_9blocks"  # selects model to use for netG
            self.n_layers_D = 3  # only used if which_model_netD==n_layers
            self.gpu_ids = [0]  # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
            self.name = "experiment_name"  # name of the experiment. It decides where to store samples and models
            self.model = "cycle_gan"  # chooses which model to use. cycle_gan, pix2pix, test
            self.which_direction = "AtoB"  # AtoB or BtoA
            self.nThreads = 1  # num. threads for loading data
            self.checkpoints_dir = "checkpoints"  # models are saved here
            self.network = "iccv_submitted"  # iccv_submitted
            self.network_H = "basic"
            self.norm = "instance"  # instance normalization or batch normalization
            self.serial_batches = True  # if true, takes images in order to make batches, otherwise takes them randomly
            self.display_winsize = 256  # display window size
            self.display_id = 1  # window id of the web display
            self.display_env = "main"  # Environment name of the web display
            self.display_port = 6005  # visdom port of the web display
            self.no_dropout = True  # no dropout for the generator
            self.max_dataset_size = float("inf")  # Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.
            self.resize_or_crop = "resize_and_crop"  # scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]
            self.no_flip = True  # if specified, do not flip the images for data augmentation
            self.init_type = "normal"  # network initialization [normal|xavier|kaiming|orthogonal]
            self.img_type = "lab"  # Environment name of the web display
            self.pair_ratio = 0.0  # Ratio of Pair data
            self.mode = "gsgt"  # gsgt, gsrt, rsrt
            self.test_dir = "1"  # 1,2,3,4,5
            self.is_psnr = False  # 1,2,3,4,5
            self.is_SR = False  # 1,2,3,4,5

            # Test Options
            self.ntest = float("inf")  # num of test examples
            self.results_dir = "results"  # saves results here.
            self.aspect_ratio = 1.0  # aspect ratio of result images
            self.phase = "test"  # train, val, test, etc
            self.which_epoch = "latest"  # which epoch to load? set to latest to use latest cached model
            self.how_many = 600  # how many test images to run
            self.video_folder = "bear"  # folder name ..
            self.ab_bin = 64  # ab_bin
            self.l_bin = 8  # l_bin
            self.isTrain = False
