"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import cv2
import torch
import os
from .utils.face_preprocessing import face_extraction
from .rehistoGAN import train_from_folder
from ColorTransferLib.ImageProcessing.Image import Image as Img

from ColorTransferLib.Utils.BaseOptions import BaseOptions
from ColorTransferLib.Utils.Helper import check_compatibility

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms
#   Author: ...
#   Published in: ...
#   Year of Publication: 2021
#
# Abstract:
#   In this paper, we present HistoGAN, a color histogram-based method for controlling GAN-generated images' colors. 
#   We focus on color histograms as they provide an intuitive way to describe image color while remaining decoupled 
#   from domain-specific semantics. Specifically, we introduce an effective modification of the recent StyleGAN 
#   architecture to control the colors of GAN-generated images specified by a target color histogram feature. We then 
#   describe how to expand HistoGAN to recolor real images. For image recoloring, we jointly train an encoder network 
#   along with HistoGAN. The recoloring model, ReHistoGAN, is an unsupervised approach trained to encourage the network 
#   to keep the original image's content while changing the colors based on the given target histogram. We show that 
#   this histogram-based approach offers a better way to control GAN-generated and real images' colors while producing 
#   more compelling results compared to existing alternative strategies.
#
# Link: ...
#
# Sources:
#   https://github.com/mahmoudnafifi/HistoGAN
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class HistoGAN:
    identifier = "HistoGAN"
    title = "HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms"
    year = 2021
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
            "identifier": "HistoGAN",
            "title": "HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms",
            "year": 2021,
            "abstract": "In this paper, we present HistoGAN, a color histogram-based method for controlling "
                        "GAN-generated images colors. We focus on color histograms as they provide an intuitive way "
                        "to describe image color while remaining decoupled from domain-specific semantics. "
                        "Specifically, we introduce an effective modification of the recent StyleGAN architecture to "
                        "control the colors of GAN-generated images specified by a target color histogram feature. We "
                        "then describe how to expand HistoGAN to recolor real images. For image recoloring, we jointly "
                        "train an encoder network along with HistoGAN. The recoloring model, ReHistoGAN, is an "
                        "unsupervised approach trained to encourage the network to keep the original images content "
                        "while changing the colors based on the given target histogram. We show that this "
                        "histogram-based approach offers a better way to control GAN-generated and real images colors "
                        "while producing more compelling results compared to existing alternative strategies."
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, HistoGAN.compatibility)

        if output["status_code"] == -1:
            return output

        # START PROCESSING
        img_src = src.get_raw()
        img_ref = ref.get_raw()

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

        output = train_from_folder(
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

        output = output.squeeze().cpu().detach().numpy().astype("float32")
        output = np.swapaxes(output,0,1)
        output = np.swapaxes(output,1,2)

        img_out = Img(array=cv2.cvtColor(output, cv2.COLOR_BGR2RGB), normalized=True)


        output = {
            "status_code": 0,
            "response": "",
            "object": img_out
        }
        
        return output

    