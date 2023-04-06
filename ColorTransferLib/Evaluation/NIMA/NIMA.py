"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import cv2
import math
import os
import numpy as np
import json
from ColorTransferLib.ImageProcessing.Image import Image
from predict import image_file_to_json, predict, image_dir_to_json
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ... (NIMA)
# 
#
# Source: https://github.com/idealo/image-quality-assessment
#
# Range []
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class NIMA:
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(img):
        img.resize(224, 224)
        img = img.get_raw()

        img_format = "png"
        base_model_name = "MobileNet"
        weights_file = "/home/hpadmin/Projects/image-quality-assessment/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"

        # build model and load weights
        nima = Nima(base_model_name, weights=None)
        nima.build()
        nima.nima_model.load_weights(weights_file)

        # get predictions
        predictions = predict(nima.nima_model, img)
        nim = calc_mean_score(predictions[0])

        return round(nim, 4)

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------ 
def main():
    file1 = open("/media/hpadmin/Active_Disk/Tests/MetricEvaluation/testset_evaluation_512.txt")
    ALG = "GLO"
    total_tests = 0
    eval_arr = []
    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        s_p, r_p = line.strip().split(" ")
        outfile_name = "/media/hpadmin/Active_Disk/Tests/MetricEvaluation/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
        print(outfile_name)
        img_tri = cv2.imread(outfile_name)
        src_img = img_tri[:,:512,:]
        ref_img = img_tri[:,512:1024,:]
        out_img = img_tri[:,1024:,:]

        src = Image(array=src_img)
        out = Image(array=out_img)
        mse = NIMA.apply(out)
        print(mse)
        eval_arr.append(mse)

        with open("/media/hpadmin/Active_Disk/Tests/MetricEvaluation/"+ALG+"/nima.txt","a") as file2:
            file2.writelines(str(round(mse,3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")



        # calculate mean
    mean = sum(eval_arr) / len(eval_arr)

    # calculate std
    std = 0
    for t in eval_arr:
        std += math.pow(t-mean, 2)
    std /= len(eval_arr)
    std = math.sqrt(std)


    print("Averaged: " + str(round(mean,3)) + " +- " + str(round(std,3)))

    file1.close()



if __name__ == "__main__":
    main()
