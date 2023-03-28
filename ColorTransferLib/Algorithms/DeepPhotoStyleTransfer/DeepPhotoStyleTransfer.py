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
import argparse
from PIL import Image
import numpy as np
import os
import time
from ColorTransferLib.Algorithms.DeepPhotoStyleTransfer.photo_style import stylize
import cv2
from ColorTransferLib.Utils.BaseOptions import BaseOptions
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Deep Photo Style Transfer
#   Author: Fujun Luan, Sylvain Paris, Eli Shechtman, Kavita Bala
#   Published in: ...
#   Year of Publication: 2017
#
# Abstract:
#   This paper introduces a deep-learning approach to photographic style transfer that handles a large variety of image
#   content while faithfully transferring the reference style. Our approach builds upon the recent work on painterly
#   transfer that separates style from the content of an image by considering different layers of a neural network.
#   However, as is, this approach is not suitable for photorealistic style transfer. Even when both the input and
#   reference images are photographs, the output still exhibits distortions reminiscent of a painting. Our contribution
#   is to constrain the transformation from the input to the output to be locally affine in colorspace, and to express
#   this constraint as a custom fully differentiable energy term. We show that this approach successfully suppresses
#   distortion and yields satisfying photorealistic style transfers in a broad variety of scenarios, including transfer
#   of the time of day, weather, season, and artistic edits.
#
# Link: https://doi.org/10.48550/arXiv.1703.07511
#
# Source: https://github.com/LouieYang/deep-photo-styletransfer-tf
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class DeepPhotoStyleTransfer:
    identifier = "DeepPhotoStyleTransfer"
    title = "Color Transfer between Images"
    year = 2017

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
            "identifier": "DeepPhotoStyleTransfer",
            "title": "Color Transfer between Images",
            "year": 2017,
            "abstract": "This paper introduces a deep-learning approach to photographic style transfer that handles a "
                        "large variety of image content while faithfully transferring the reference style. Our "
                        "approach builds upon the recent work on painterly transfer that separates style from the "
                        "content of an image by considering different layers of a neural network. However, as is, this "
                        "approach is not suitable for photorealistic style transfer. Even when both the input and "
                        "reference images are photographs, the output still exhibits distortions reminiscent of a "
                        "painting. Our contribution is to constrain the transformation from the input to the output to "
                        "be locally affine in colorspace, and to express this constraint as a custom fully "
                        "differentiable energy term. We show that this approach successfully suppresses distortion and "
                        "yields satisfying photorealistic style transfers in a broad variety of scenarios, including "
                        "transfer of the time of day, weather, season, and artistic edits."
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, DeepPhotoStyleTransfer.compatibility)

        # Preprocessing
        src_img = src.get_raw() * 255.0
        ref_img = ref.get_raw() * 255.0
        out_img = deepcopy(src)


        #src_shape = src_img.shape
        #src_img = cv2.resize(src_img, (512, 512), interpolation=cv2.INTER_AREA)

        if opt.style_option == 0:
            best_image_bgr = stylize(opt, False, src_img, ref_img)
            #result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            #result.save(opt.output_image)
            out = np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0))
        elif opt.style_option == 1:
            best_image_bgr = stylize(opt, True, src_img, ref_img)
            """
            if not args.apply_smooth:
                result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
                result.save(args.output_image)
            else:
                # Pycuda runtime incompatible with Tensorflow
                from smooth_local_affine import smooth_local_affine
                content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
                # RGB to BGR
                content_input = content_input[:, :, ::-1]
                # H * W * C to C * H * W
                content_input = content_input.transpose((2, 0, 1))
                input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

                _, H, W = np.shape(input_)

                output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
                best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
                result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
                result.save(args.output_image)
            """
            #result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            #result.save(opt.output_image)
            out = np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0))
        elif opt.style_option == 2:
            opt.max_iter = 2 * opt.max_iter
            tmp_image_bgr = stylize(opt, False, src_img, ref_img)
            result = Image.fromarray(np.uint8(np.clip(tmp_image_bgr[:, :, ::-1], 0, 255.0)))
            opt.init_image_path = os.path.join(opt.serial, "tmp_result.png")
            result.save(opt.init_image_path)

            best_image_bgr = stylize(opt, True, src_img, ref_img)
            """
            if not args.apply_smooth:
                result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
                result.save(args.output_image)
            else:
                from smooth_local_affine import smooth_local_affine
                content_input = np.array(Image.open(args.content_image_path).convert("RGB"), dtype=np.float32)
                # RGB to BGR
                content_input = content_input[:, :, ::-1]
                # H * W * C to C * H * W
                content_input = content_input.transpose((2, 0, 1))
                input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.

                _, H, W = np.shape(input_)

                output_ = np.ascontiguousarray(best_image_bgr.transpose((2, 0, 1)), dtype=np.float32) / 255.
                best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, args.f_radius, args.f_edge).transpose(1, 2, 0)
                result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
                result.save(args.output_image)
            """

            #result = Image.fromarray(np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0)))
            #result.save(opt.output_image)
            out = np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0))
        #out = cv2.resize(out, (src_shape[1], src_shape[0]), interpolation=cv2.INTER_AREA)

        out_img.set_raw(out.astype(np.float32))
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Options Class
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    class Options:
        def __init__(self):
            # Input Options
            self.content_image_path = "data/images/2020_Lee_Example-18_Source.png"  # Path to the content image
            self.style_image_path = "data/images/2020_Lee_Example-18_Reference.png"  # Path to the style image
            self.content_seg_path = "data/images/2020_Lee_Example-18_Source_seg.png"  # Path to the style segmentation
            self.style_seg_path = "data/images/2020_Lee_Example-18_Reference_seg.png"  # Path to the style segmentation
            self.init_image_path = ""  # Path to init image
            self.output_image = "best_stylized.png"  # Path to output the stylized image
            self.serial = "./"  # Path to save the serial out_iter_X.png
            self.vgg19_path = "Models/DeepPhotoStyleTransfer/vgg19.npy"  # Path to save the serial out_iter_X.png

            # Training Optimizer Options
            self.max_iter = 100  # maximum image iteration
            self.learning_rate = 1.0  # learning rate for adam optimizer
            self.print_iter = 1  # print loss per iterations

            # Note the result might not be smooth enough since not applying smooth for temp result
            self.save_iter = 100  # save temporary result per iterations
            self.lbfgs = True  # True=lbfgs, False=Adam

            # Weight Options
            self.content_weight = 5e0  # weight of content loss
            self.style_weight = 1e2  # weight of style loss
            self.tv_weight = 1e-3  # weight of total variational loss
            self.affine_weight = 1e4  # weight of affine loss

            # Style Options
            self.style_option = 0  # 0=non-Matting, 1=only Matting, 2=first non-Matting, then Matting
            self.apply_smooth = False  # if apply local affine smooth

            # Smoothing Argument
            self.f_radius = 15  # smooth argument
            self.f_edge = 1e-1  # smooth argument
