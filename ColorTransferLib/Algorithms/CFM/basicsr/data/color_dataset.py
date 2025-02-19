import cv2
import random
import time
import numpy as np
import torch
from torch.utils import data as data

from ColorTransferLib.Algorithms.CFM.basicsr.data.transforms import rgb2lab
from ColorTransferLib.Algorithms.CFM.basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from ColorTransferLib.Algorithms.CFM.basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LabSegmentDataset(data.Dataset):
    """
    Dataset used for Segmentation
    """

    def __init__(self, opt):
        super(LabSegmentDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")

        meta_info_file = self.opt['meta_info_file']
        if meta_info_file is not None:
            if not isinstance(meta_info_file, list):
                meta_info_file = [meta_info_file]
            self.paths = []
            for meta_info in meta_info_file:
                with open(meta_info, 'r') as fin:
                    self.paths.extend([line.strip() for line in fin])
        else:
            import lmdb
            env = lmdb.open(self.gt_folder)
            with env.begin() as txn:
                myList = [ key.decode('ascii') for key, value in txn.cursor() if value is not None]
            self.paths = myList
        self.min_ab, self.max_ab = -128, 128
        self.interval_ab = 4
        self.ab_palette = [i for i in range(self.min_ab, self.max_ab + self.interval_ab, self.interval_ab)]
        # print(self.ab_palette)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        gt_size = self.opt['gt_size']
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = cv2.resize(img_gt, (gt_size, gt_size))
        # -------------------- get gray lq, to tensor -------------------- #
        # convert to gray
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_l, img_ab = rgb2lab(img_gt)

        target_a, target_b = self.ab2int(img_ab)

        # numpy to tensor
        img_l, img_ab = img2tensor([img_l, img_ab], bgr2rgb=False, float32=False)
        target_a, target_b = torch.LongTensor(target_a), torch.LongTensor(target_b)
        return_d = {
            'lq': img_l,
            'gt': img_ab,
            'target_a': target_a,
            'target_b': target_b,
            'lq_path': gt_path,
            'gt_path': gt_path
        }
        return return_d

    def ab2int(self, img_ab):
        img_a, img_b = img_ab[:, :, 0], img_ab[:, :, 1]
        int_a = (img_a - self.min_ab) / self.interval_ab
        int_b = (img_b - self.min_ab) / self.interval_ab

        return np.round(int_a), np.round(int_b)

    def __len__(self):
        return len(self.paths)