import sys
import os
import cv2
from os.path import basename
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from imblearn.over_sampling import SMOTE
from subprocess import call


#compute similarity between two salienycy maps
def saliency_similarity(input_sal1, input_sal2, target_sal1, target_sal2):
    img1 = cv2.imread(input_sal1, 0)
    img2 = cv2.imread(input_sal2, 0)
    img3 = cv2.imread(target_sal1, 0)
    img4 = cv2.imread(target_sal2, 0)

    img1 = img1 / 2.0 + img2 / 2.0
    img2 = img3 / 2.0 + img4 / 2.0

    map1 = []
    map2 = []

    map1 = img1.reshape(1, img1.size)
    map2 = img2.reshape(1, img2.size)

    map1 = np.array(map1)
    map2 = np.array(map2)

    map1 = (map1 - np.amin(map1)) / float((np.amax(map1) - np.amin(map1)))
    map1 = map1 / np.sum(map1)

    map2 = (map2 - np.amin(map2)) / float((np.amax(map2) - np.amin(map2)))
    map2 = map2 / np.sum(map2)

    diff = []
    for i in range(0, img1.size):
        if (map1[0][i] < map2[0][i]):
            diff.append(map1[0][i])
        else:
            diff.append(map2[0][i])

    score = np.sum(diff)
    with open("For_fit.txt", 'a') as for_fit:
        for_fit.write(str(score)+"\n")

    return score


def style_score(inp, tar, res):
    ssim_data = []
    bhatt_lum_data = []
    bhatt_color_data = []
    saliency_data = []
    bhatt_saliency_data = []
    brightness_data = []
    lightness_data = []
    chroma_data = []
    colorfulness_data = []
    saturation_data = []
    style_data = []

    #open the file with the objective metrics for each images, used in our user study
    with open("metrics_summary.txt", "rb") as metrics:
        for line in metrics:
            metric = line.split()
            ssim_data.append(metric[0])
            bhatt_lum_data.append(metric[1])
            bhatt_color_data.append(metric[2])
            saliency_data.append(metric[3])
            brightness_data.append(metric[4])
            lightness_data.append(metric[5])
            chroma_data.append(metric[6])
            colorfulness_data.append(metric[7])
            saturation_data.append(metric[8])
            style_data.append(metric[9])

    ssim_data = np.array(ssim_data).astype(np.float)
    bhatt_lum_data = np.array(bhatt_lum_data).astype(np.float)
    bhatt_color_data = np.array(bhatt_color_data).astype(np.float)
    saliency_data = np.array(saliency_data).astype(np.float)
    bhatt_saliency_data = np.array(bhatt_saliency_data).astype(np.float)
    brightness_data = np.array(brightness_data).astype(np.float)
    lightness_data = np.array(lightness_data).astype(np.float)
    chroma_data = np.array(chroma_data).astype(np.float)
    colorfulness_data = np.array(colorfulness_data).astype(np.float)
    saturation_data = np.array(saturation_data).astype(np.float)

    x = np.vstack((ssim_data, bhatt_lum_data, bhatt_color_data, saliency_data, brightness_data, lightness_data, chroma_data, colorfulness_data, saturation_data)).T
    y = np.array(style_data).astype(np.float)
    n = np.max(x.shape)

    #oversampling
    sm = SMOTE(kind='regular', k_neighbors = 2)
    for i in range(0, 9):
        sm.fit(x, y)
        x, y = sm.sample(x, y)

    #random forests regression
    rf = RandomForestRegressor(n_estimators=200, oob_score=True, max_features='sqrt')

    rf.fit(x, y)

    #change the given path with the path to Matlab
    path = "/home/hhristov/Matlab/"
    current_dir = "%CD%"

    ssim = call(["./ssim", inp, res, '2'])
    color_bhatt = call(["./bhattacharya", tar, res])
    call(["./run_computeSaliency.sh", path, inp, current_dir])
    call(["./run_computeSaliency.sh", path, res, current_dir])

    inp_name, extension = os.path.splitext(basename(inp))
    res_name, extension = os.path.splitext(basename(res))

    inp_name_GBVS = current_dir+inp_name+"SaliencyMap_GBVS.jpg"
    inp_name_RARE = current_dir+inp_name+"SaliencyMap_RARE.jpg"
    res_name_GBVS = current_dir+res_name+"SaliencyMap_GBVS.jpg"
    res_name_RARE = current_dir+res_name+"SaliencyMap_RARE.jpg"

    saliency_similarity(inp_name_GBVS, inp_name_RARE, res_name_GBVS, res_name_RARE)

    perceptual_bhatt = call(["./bhatt", tar, res])

    data_to_fit = []
    with open("For_fit.txt", 'r') as for_fit:
        for line in for_fit:
            score = line.replace("\n", "")
            score = score.replace("\r", "")

            data_to_fit.append(score)
    data_to_fit = np.array(data_to_fit)

    data_to_fit = data_to_fit.reshape(1, -1)
    final_score = rf.predict(data_to_fit)
    return float(final_score[0])


if __name__ == '__main__':
    inp = sys.argv[1]
    tar = sys.argv[2]
    res = sys.argv[3]

    score = style_score(inp, tar, res)
    print "Perceptual score = ", score
