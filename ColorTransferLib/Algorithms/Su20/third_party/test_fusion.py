import os
from .models import create_model

import torch
from tqdm import tqdm
import cv2

from .fusion_dataset import Fusion_Testing_Dataset
from .util import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import multiprocessing
multiprocessing.set_start_method('spawn', True)

torch.backends.cudnn.benchmark = True

def predict(src_img, pred_bbox, pred_scores, opt, model_file_paths):
    original_shape = src_img.shape

    # opt.batch_size = 1
    dataset = Fusion_Testing_Dataset(opt, src_img, pred_bbox, pred_scores)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    # dataset_size = len(dataset)
    # print('#Testing images = %d' % dataset_size)

    model = create_model(opt)
    # model.setup_to_test('coco_finetuned_mask_256')
    model.setup_to_test(model_file_paths, 'coco_finetuned_mask_256_ffs')
    count_empty = 0

    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        # if os.path.isfile(join(save_img_path, data_raw['file_id'][0] + '.png')) is True:
        #     continue
        data_raw['full_img'][0] = data_raw['full_img'][0].cuda()
        if data_raw['empty_box'][0] == 0:
            data_raw['cropped_img'][0] = data_raw['cropped_img'][0].cuda()
            box_info = data_raw['box_info'][0]
            box_info_2x = data_raw['box_info_2x'][0]
            box_info_4x = data_raw['box_info_4x'][0]
            box_info_8x = data_raw['box_info_8x'][0]
            cropped_data = util.get_colorization_data(data_raw['cropped_img'], opt, ab_thresh=0, p=opt.sample_p)
            full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
            model.set_input(cropped_data)
            model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
            model.forward()
        else:
            count_empty += 1
            full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
            model.set_forward_without_box(full_img_data)
        # model.save_current_imgs(join(save_img_path, data_raw['file_id'][0] + '.png'))
        out_imgs = model.get_current_imgs()
        out_imgs = cv2.resize(out_imgs, (original_shape[1], original_shape[0]))
        return out_imgs
    print('{0} images without bounding boxes'.format(count_empty))
