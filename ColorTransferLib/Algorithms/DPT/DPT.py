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
from ColorTransferLib.Algorithms.DPT.photo_style import stylize
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
# Info:
#   Name: Deep Photo Style Transfer
#   Identifier: DPT
#   Link: https://doi.org/10.48550/arXiv.1703.07511
#   Source: https://github.com/LouieYang/deep-photo-styletransfer-tf
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class DPT:
    compatibility = {
        "src": ["Image"],
        "ref": ["Image", "Mesh"]
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
            "identifier": "DPT",
            "title": "Deep Photo Style Transfer",
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
                        "transfer of the time of day, weather, season, and artistic edits.",
            "types": ["Image"]
        }

        return info
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, DPT.compatibility)

        # Preprocessing
        src_img = src.get_raw() * 255.0
        ref_img = ref.get_raw() * 255.0
        out_img = deepcopy(src)

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
            #result.save(opt.init_image_path)

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

            out = np.uint8(np.clip(best_image_bgr[:, :, ::-1], 0, 255.0))

        out_img.set_raw(out.astype(np.float32))
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output