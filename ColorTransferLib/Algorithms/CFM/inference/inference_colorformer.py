import argparse
import cv2
import numpy as np
import os
import torch
import tqdm

from ColorTransferLib.Algorithms.CFM.basicsr.archs.colorformer_arch import ColorFormer as models
from ColorTransferLib.Algorithms.CFM.basicsr.utils.img_util import tensor_lab2rgb
from ColorTransferLib.Algorithms.CFM.basicsr.data.val_dataset import ValDataset
from ColorTransferLib.Algorithms.CFM.basicsr.data.transforms import rgb2lab
from queue import Queue
import _thread

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

write_buffer = Queue(maxsize=500)


def clear_write_buffer(args, write_buffer):
    while True:
        item = write_buffer.get()
        for name in item.keys():
            cls, filename = os.path.split(name)
            if cls:
                os.makedirs(os.path.join(args.output, cls), exist_ok=True)
            cv2.imwrite(os.path.join(args.output, name), item[name])


def predict(model_file_paths, src_img):
    # set up model
    model = models(
        'GLHTransformer',
        pretrained_path=model_file_paths["GLH.pth"],
        input_size=[256, 256],
        num_output_channels=2,
        last_norm='Spectral',
        do_normalize=False,
        color_centers_path=model_file_paths["color_embed_10000.npy"], 
        semantic_centers_path=model_file_paths["semantic_embed_10000.npy"])
    model.load_state_dict(torch.load(model_file_paths["net_g_200000.pth"])['params'], strict=True)
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        original_shape = src_img.shape

        src_img = cv2.resize(src_img, dsize=(256, 256))
        # -------------------- get gray lq, to tensor -------------------- #
        # convert to gray
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_img = src_img.astype(np.float32)
        img_l, _ = rgb2lab(src_img)
        img_l = torch.from_numpy(np.transpose(img_l, (2, 0, 1))).float().unsqueeze(0)
        tensor_lab = torch.cat([img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)], dim=1)
        tensor_rgb = tensor_lab2rgb(tensor_lab)

        src_img_tensor = tensor_rgb[0].unsqueeze(0).to(device)
        src_img_l_tensor = img_l[0].unsqueeze(0).to(device)

        outs = model(src_img_tensor)

        outs = torch.cat([src_img_l_tensor, outs], dim=1)
        outs = tensor_lab2rgb(outs)
        outs = outs.cpu()

        output = outs.data.float().clamp_(0, 1).numpy()
        output = output.squeeze()
        output = np.transpose(output, (1, 2, 0))
        output_img = (output * 255.0).round().astype(np.uint8)
        output_img = cv2.resize(output_img, (original_shape[1], original_shape[0]))
  
        return output_img

    