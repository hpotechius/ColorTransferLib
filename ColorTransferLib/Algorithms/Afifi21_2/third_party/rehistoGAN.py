"""
 If you find this code useful, please cite our paper:

 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN:
 Controlling Colors of GAN-Generated and Real Images via Color Histograms."
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
"""

from tqdm import tqdm
from .ReHistoGAN.rehistoGAN import recoloringTrainer
from .histoGAN.histoGAN import Trainer, NanException
from datetime import datetime
import torch
import argparse
from retry.api import retry_call
import os
from PIL import Image
from torchvision import transforms
import torchvision
import numpy as np
import copy
from .utils.face_preprocessing import face_extraction
from .histogram_classes.RGBuvHistBlock import RGBuvHistBlock
import cv2
import gc



# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class expand_greyscale(object):
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def __call__(self, tensor):
        return tensor.expand(self.num_channels, -1, -1)

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def hist_interpolation(hists):
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    ratios = torch.abs(torch.rand(hists.shape[0])).to(device=device)
    ratios = ratios / torch.sum(ratios)
    out_hist = hists[0, :, :, :, :] * ratios[0]
    for i in range(hists.shape[0] - 1):
        out_hist = out_hist + hists[i + 1, :, :, :, :] * ratios[i + 1]
    return out_hist

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def process_image(model, name, input_image, target_hist, image_size=256,
                  upsampling_output=False,
                  upsampling_method='pyramid',
                  swapping_levels=1,
                  pyramid_levels=5,
                  level_blending=False,
                  post_recoloring=False,
                  sampling=True,
                  target_number=1, results_dir='./results_ReHistoGAN/',
                  hist_insz=150, hist_bin=64,
                  hist_method='inverse-quadratic', hist_resizing='sampling',
                  hist_sigma=0.02):
    
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    img = input_image# Image.open(input_image)
    original_img = input_image#np.array(img) / 255
    img = Image.fromarray((img*255).astype("uint8"))

    if upsampling_output:
        width, height = img.size
        if width > image_size or height > image_size:
            resizing_mode = 'upscaling'
        elif width < image_size or height < image_size:
            resizing_mode = 'downscaling'
        else:
            resizing_mode = 'none'
    else:
        resizing_mode = None
        width = None
        height = None

    if width != image_size or height != image_size:
        img = img.resize((image_size, image_size))

    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

    postfix = round(np.random.rand() * 1000)
    transform = transforms.Compose([
        transforms.Lambda(convert_transparent_to_rgb),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(expand_greyscale(3))
    ])

    img = torch.unsqueeze(transform(img), dim=0).to(device=device)
    histblock = RGBuvHistBlock(insz=hist_insz, h=hist_bin, resizing=hist_resizing, method=hist_method, sigma=hist_sigma, device=device)
    transform = transforms.Compose([transforms.ToTensor()])

    img_hist = Image.fromarray((target_hist*255).astype("uint8"))#Image.open(target_hist)
    img_hist = torch.unsqueeze(transform(img_hist), dim=0).to(device=device)
    with torch.no_grad():
        h = histblock(img_hist)
        samples_name = "out"#('output-' + f'{os.path.basename(os.path.splitext(target_hist)[0])}' f'-{timestamp}-{postfix}')
        output = model.evaluate(samples_name, image_batch=img,
                        hist_batch=h,
                        resizing=resizing_mode,
                        resizing_method=upsampling_method,
                        swapping_levels=swapping_levels,
                        pyramid_levels=pyramid_levels,
                        level_blending=level_blending,
                        original_size=[width, height],
                        original_image=original_img,
                        input_image_name=input_image,
                        save_input=False,
                        post_recoloring=post_recoloring)
        
        return output

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def train_from_folder(
        results_dir='./results_ReHistoGAN/',
        models_dir='./Models/HistoGAN',
        name='test',
        new=False,
        load_from=-1,
        image_size=128,
        network_capacity=16,
        transparent=False,
        load_histogan_weights=True,
        batch_size=2,
        sampling=True,
        gradient_accumulate_every=8,
        num_train_steps=200000,
        learning_rate=2e-4,
        num_workers=None,
        save_every=10000,
        generate=False,
        trunc_psi=0.75,
        fp16=False,
        skip_conn_to_GAN=False,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        hist_method='inverse-quadratic',
        hist_resizing='sampling',
        hist_sigma=0.02,
        hist_bin=64,
        hist_insz=150,
        rec_loss='laplacian',
        alpha=32,
        beta=1.5,
        gamma=4,
        fixed_gan_weights=False,
        initialize_gan=False,
        variance_loss=False,
        target_hist=None,
        internal_hist=False,
        histoGAN_model_name=None,
        input_image=None,
        target_number=None,
        change_hyperparameters=False,
        change_hyperparameters_after=100000,
        upsampling_output=False,
        upsampling_method='pyramid',
        swapping_levels=1,
        pyramid_levels=6,
        level_blending=False,
        post_recoloring=False):

    model = recoloringTrainer(
        name,
        results_dir,
        models_dir,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        image_size=image_size,
        network_capacity=network_capacity,
        transparent=transparent,
        lr=learning_rate,
        num_workers=num_workers,
        save_every=save_every,
        trunc_psi=trunc_psi,
        fp16=fp16,
        fq_layers=fq_layers,
        fq_dict_size=fq_dict_size,
        attn_layers=attn_layers,
        hist_insz=hist_insz,
        hist_bin=hist_bin,
        hist_sigma=hist_sigma,
        hist_resizing=hist_resizing,
        hist_method=hist_method,
        rec_loss=rec_loss,
        fixed_gan_weights=fixed_gan_weights,
        skip_conn_to_GAN=skip_conn_to_GAN,
        initialize_gan=initialize_gan,
        variance_loss=variance_loss,
        internal_hist=internal_hist,
        change_hyperparameters=change_hyperparameters,
        change_hyperparameters_after=change_hyperparameters_after
    )
    model.load(name)

    # extension = os.path.splitext(input_image)[1]
    # if (extension == str.lower(extension) == '.jpg' or str.lower(extension) == '.png'):
    output = process_image(model, name, input_image, target_hist, image_size=256,
                    upsampling_output=upsampling_output,
                    upsampling_method=upsampling_method,
                    swapping_levels=swapping_levels,
                    pyramid_levels=pyramid_levels,
                    level_blending=level_blending,
                    post_recoloring=post_recoloring,
                    sampling=sampling,
                    target_number=target_number, results_dir=results_dir,
                    hist_insz=hist_insz, hist_bin=hist_bin,
                    hist_method=hist_method, hist_resizing=hist_resizing,
                    hist_sigma=hist_sigma)
    #print(output.shape)

    #del model
    #torch.cuda.empty_cache()

    return output
