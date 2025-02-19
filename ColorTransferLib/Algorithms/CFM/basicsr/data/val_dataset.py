from unicodedata import name
import torch
import os
import cv2
import numpy as np
from ColorTransferLib.Algorithms.CFM.basicsr.data.transforms import rgb2lab
from ColorTransferLib.Algorithms.CFM.basicsr.utils.img_util import tensor_lab2rgb


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, input_size, full=False) -> None:
        super().__init__()
        if full:
            img_list = []
            for cls in os.listdir(img_root):
                for name in os.listdir(os.path.join(img_root, cls)):
                    img_list.append(os.path.join(cls, name))
            self.img_list = img_list
        else:
            self.img_list = os.listdir(img_root)
        self.img_root = img_root
        self.input_size = input_size

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img = cv2.imread(os.path.join(self.img_root, name))
        img = cv2.resize(img, dsize=(self.input_size, self.input_size))
        # -------------------- get gray lq, to tensor -------------------- #
        # convert to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img_l, _ = rgb2lab(img)
        img_l = torch.from_numpy(np.transpose(img_l, (2, 0, 1))).float().unsqueeze(0)
        tensor_lab = torch.cat([img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)], dim=1)

        tensor_rgb = tensor_lab2rgb(tensor_lab)

        return tensor_rgb[0], img_l[0], name

    def __len__(self):
        return len(self.img_list)
