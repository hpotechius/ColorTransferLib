import skimage.measure
import os
import cv2
import numpy as np
import glob

psnrs = []


gt_root = ""
pred_root = ""

for img_name in os.listdir(gt_root):

    img1 = cv2.imread(os.path.join(gt_root, img_name))
    file_name = glob.glob(os.path.join(pred_root, os.path.splitext(img_name)[0]+"*"))[0]

    img2 = cv2.imread(os.path.join(file_name))
    psnr = skimage.measure.compare_psnr(img1, img2)
    if psnr==np.inf:
        continue
    psnrs.append(psnr)
    print(len(psnrs), np.mean(psnrs))