import argparse
import cv2
import numpy as np
import os
import torch
import tqdm
from numpy import random
from basicsr.archs.colorformer_arch_util import Hook
from basicsr.archs.GLHTransformer import GLHTransformer
import math
import torch.nn as nn


from basicsr.data.transforms import rgb2lab
from basicsr.utils.img_util import tensor_lab2rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

    def __init__(self, hook_names, **kwargs):
        super().__init__()
        self.arch = GLHTransformer()
        self.arch.load_state_dict(torch.load('pretrain/GLH.pth', map_location="cpu"), strict=False)
        self.hook_names = hook_names
        self.hooks = self.setup_hooks()

    def setup_hooks(self):
        hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
        return hooks

    def forward(self, x):
        return self.arch(x)

def process_frame(input_img, model, args):
    img = cv2.resize(input_img, dsize=(args.input_size, args.input_size))
    # -------------------- get gray lq, to tensor -------------------- #
    # convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_l, _ = rgb2lab(img)

    img_l = torch.from_numpy(np.transpose(img_l, (2, 0, 1))).float()
    img_l = img_l.unsqueeze(0).to(device)

    tensor_lab = torch.cat([img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)], dim=1)

    tensor_rgb = tensor_lab2rgb(tensor_lab)

    # inference
    with torch.no_grad():
        model(tensor_rgb)
        output = model.hooks[-1].feature
        bs, L, num_feat = output.shape

        output = output.view(bs, int(math.sqrt(L)), int(math.sqrt(L)), num_feat)
    img_rz = cv2.resize(img, (int(math.sqrt(L)), int(math.sqrt(L))))
    _, img_ab = rgb2lab(img_rz)

    return output.data.squeeze(0).float().cpu().numpy(), img_ab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_txt',
        type=str,
        default='',
        help='input test image folder or video path')
    parser.add_argument(
        '--output',
        type=str,
        default='./memory_build/semantic_color/',
        help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=256, help='input size')
    args = parser.parse_args()

    # set up model
    model = Encoder(['norm3'])
    model.eval()
    model = model.to(device)

    print('Processing images folder...')
    os.makedirs(args.output, exist_ok=True)
    # image_list = sorted(glob.glob(os.path.join(args.input, '*')))
    full_img_names = []
    with open(args.input_txt, 'r') as f:
        full_img_names.extend([line.strip() for line in f.readlines()])
    random.seed(0)
    image_list = sorted(random.choice(full_img_names, 10000))
    semantic_array = []
    color_array = []
    for path in tqdm.tqdm(image_list):
        # read image
        input_img = cv2.imread(path, cv2.IMREAD_COLOR)
        semantic, color = process_frame(input_img, model, args)
        semantic = semantic.reshape(-1, semantic.shape[-1])
        color = color.reshape(-1, color.shape[-1])
        for array_s, array_c in zip(semantic, color):
            semantic_array.append(array_s)
            color_array.append(array_c)
    semantic_array = np.array(semantic_array)
    color_array = np.array(color_array)
    np.save(os.path.join(args.output, 'semantic_array_10000'), semantic_array)
    np.save(os.path.join(args.output, 'color_array_10000'), color_array)



if __name__ == '__main__':
    main()
