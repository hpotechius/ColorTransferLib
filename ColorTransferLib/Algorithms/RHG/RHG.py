"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import cv2
import torch
import os
import time
from copy import deepcopy

from .utils.face_preprocessing import face_extraction
from .rehistoGAN import train_from_folder
from ColorTransferLib.Utils.Helper import check_compatibility, get_cache_dir
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms
#   Author: Mahmoud Afifi, Marcus A. Brubaker, Michael S. Brown
#   Published in: CVPR
#   Year of Publication: 2021
#
# Info:
#   Name: ReHistoGAN
#   Identifier: RHG
#   Link: https://doi.org/10.1109/CVPR46437.2021.00785
#   Sources: https://github.com/mahmoudnafifi/HistoGAN
#
# Implementation Details:
#   model: Universal rehistoGAN v0, internal down and upsampling
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class RHG:
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

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = RHG.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = RHG.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = RHG.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = RHG.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = RHG.__apply_mesh(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1
            out_obj = None

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(img_src, img_ref, opt):
        opt.models_dir = os.path.join(get_cache_dir(), "RHG")
        opt.histGAN_models_dir = os.path.join(opt.models_dir, "RHG")

        src_orig_wh = img_src.shape[:2]

        if torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu)

        if opt.generate and opt.face_extraction:
            if opt.input_image is None:
                raise Exception('No input image is given')
            extension = os.path.splitext(opt.input_image)[1]
            if (extension == str.lower(extension) == '.jpg' or str.lower(extension) == '.png'):
                face_extraction(opt.input_image)
                input_image = f'./temp-faces/{os.path.split(opt.input_image)[-1]}'
            else:
                raise Exception('File extension is not supported!')
        else:
            input_image = opt.input_image

        input_image = img_src
        opt.target_hist = img_ref

        out_temp = train_from_folder(
            results_dir=opt.results_dir,
            models_dir=opt.models_dir,
            name=opt.name,
            new=opt.new,
            load_from=opt.load_from,
            load_histogan_weights=opt.load_histoGAN_weights,
            image_size=opt.image_size,
            network_capacity=opt.network_capacity,
            transparent=opt.transparent,
            batch_size=opt.batch_size,
            gradient_accumulate_every=opt.gradient_accumulate_every,
            num_train_steps=opt.num_train_steps,
            learning_rate=opt.learning_rate,
            num_workers=opt.num_workers,
            save_every=opt.save_every,
            generate=opt.generate,
            trunc_psi=opt.trunc_psi,
            fp16=opt.fp16,
            fq_layers=opt.fq_layers,
            fq_dict_size=opt.fq_dict_size,
            attn_layers=opt.attn_layers,
            hist_method=opt.hist_method,
            hist_resizing=opt.hist_resizing,
            hist_sigma=opt.hist_sigma,
            hist_bin=opt.hist_bin,
            hist_insz=opt.hist_insz,
            target_hist=opt.target_hist,
            alpha=opt.alpha,
            beta=opt.beta,
            gamma=opt.gamma,
            skip_conn_to_GAN=opt.skip_conn_to_GAN,
            fixed_gan_weights=opt.fixed_gan_weights,
            sampling=opt.sampling,
            rec_loss=opt.rec_loss,
            initialize_gan=opt.initialize_gan,
            variance_loss=opt.variance_loss,
            input_image=input_image,
            internal_hist=opt.internal_hist,
            histoGAN_model_name=opt.histoGAN_model_name,
            target_number=opt.target_number,
            change_hyperparameters=opt.change_hyperparameters,
            change_hyperparameters_after=opt.change_hyperparameters_after,
            upsampling_output=opt.upsampling_output,
            upsampling_method=opt.upsampling_method,
            swapping_levels=opt.swapping_levels,
            pyramid_levels=opt.pyramid_levels,
            level_blending=opt.level_blending,
            post_recoloring=opt.post_recoloring
        )

        out_temp = out_temp.squeeze().cpu().detach().numpy().astype("float32")
        out_temp = np.swapaxes(out_temp,0,1)
        out_temp = np.swapaxes(out_temp,1,2)

        new_size = (src_orig_wh[1], src_orig_wh[0])
        out_temp = cv2.resize(out_temp, new_size, interpolation = cv2.INTER_AREA)

        return out_temp


    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = RHG.__color_transfer(src_img, ref_img, opt)

        out_img.set_colors(out_colors)
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

            out_colors = RHG.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_colors(out_colors)
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
            ref_raw = ref.get_cget_rawolors()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = RHG.__color_transfer(src_raw, ref_raw, opt)

            out_img.set_colors(out_colors)
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

                out_colors = RHG.__color_transfer(src_raw, ref_raw, opt)

                out_lightfield_array[row][col].set_colors(out_colors)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_img = src.get_raw()
        ref_img = ref.get_raw()
        out_img = deepcopy(src)

        out_colors = RHG.__color_transfer(src_img, ref_img, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

