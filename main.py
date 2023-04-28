"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""


import time
import numpy as np
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR

from ColorTransferLib.MeshProcessing.PLYLoader import PLYLoader
from ColorTransferLib.MeshProcessing.Mesh import Mesh
from ColorTransferLib.ImageProcessing.Image import Image
from ColorTransferLib.ColorTransfer import ColorTransfer

from ColorTransferLib.Utils.Math import get_random_3x3rotation_matrix
from ColorTransferLib.Evaluation.SSIM.SSIM import SSIM
from ColorTransferLib.Evaluation.PSNR.PSNR import PSNR
from ColorTransferLib.Evaluation.HI.HI import HI
from ColorTransferLib.Evaluation.PerceptualMetric.PerceptualMetric import PerceptualMetric
import cv2
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy
import gdown
import zipfile36 as zipfile
import random

# ------------------------------------------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------------------------------------------
def img2img_test(src_path, ref_path, ct_approach):
    src = Image(file_path=src_path)

    #kernel = np.ones((5,5),np.float32)/25
    #src.set_raw(cv2.filter2D(src.get_raw(), -1, kernel), normalized=True)

    ref = Image(file_path=ref_path)
    ct = ColorTransfer(src, ref, ct_approach)
    output = ct.apply()
    return output

# download Models folder
# if not os.path.exists("Models") and not os.path.exists("data"):
#     print("Download DATA.zip ...")
#     url = "https://drive.google.com/file/d/1ShJpPFJ9vCu5Vb7FJk7aSnn7ywFiT0GJ/view?usp=share_link"
#     output_path = 'DATA.zip'
#     gdown.download(url, output_path, quiet=False, fuzzy=True)
#     # Extract DATA.zip
#     print("Extract DATA.zip ...")
#     with zipfile.ZipFile("DATA.zip","r") as zip_ref:
#         zip_ref.extractall()
#     # Delete DATA.zip
#     print("Delete DATA.zip ...")
#     os.remove("DATA.zip")

# exit()

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Possible values for ct_approach:
# ["GlobalColorTransfer",       support for [images, point clouds]
#  "PdfColorTransfer",          support for [images, point clouds]
#  "MongeKLColorTransfer",      support for [images, point clouds]
#  "NeuralStyleTransfer",       support for [images]
#  "HistogramAnalogy",          support for [images]
#  "CamsTransfer",              support for [images]
#  "DeepPhotoStyleTransfer",    support for [images]
#  "TpsColorTransfer",          support for [images]
#  "FuzzyColorTransfer",        support for [images]
#  "GmmEmColorTransfer",        support for [images]
#  "Eb3dColorTransfer",         support for [point clouds]
#  "PSNetStyleTransfer",        support for [point clouds]
#  "SemanticColorTransfer",     support for [images]
#  ]
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #src_img = "/home/hpadmin/Downloads/psource.png"
    #ref_img = "/home/hpadmin/Downloads/preference.png"

    #pm = PerceptualMetric()
    #pm.apply(src, ref)
    #exit()

    #src = Image(file_path=src_img)
    #ref = Image(file_path=ref_img)
    #cstat = HistogramIntersection.apply(src, ref)
    #print(cstat)
    #exit()

    appr = ["GlobalColorTransfer",
            "FuzzyColorTransfer",
            "TpsColorTransfer",
            "GmmEmColorTransfer",
            "PdfColorTransfer",
            "MongeKLColorTransfer",
            "HistogramAnalogy",
            "NeuralStyleTransfer",
            "CamsTransfer",
            "DeepPhotoStyleTransfer",
            "Eb3dColorTransfer", # no image support?
            "PSNetStyleTransfer",
            "HistoGAN",
            "BasicColorCategoryTransfer",
            "FuzzyColorCategoryTransfer"]

    ct_approach = "PdfColorTransfer"
    ct_input = "img-img"





    src_img = '/home/potechius/Downloads/source.png'
    ref_img = '/home/potechius/Downloads/reference.png'
    output = img2img_test(src_img, ref_img, ct_approach)
    cv2.imwrite("/home/potechius/Downloads/pdf.png", cv2.cvtColor(output["object"].get_raw(), cv2.COLOR_BGR2RGB)*255)
    exit()

    # files_256 = []
    # files_512 = []
    # files_1024 = []
    # files_2048 = []
    # files_4096 = []
    # files_8192 = []
    # for path, subdirs, files in os.walk("/media/hpadmin/Active_Disk/Datasets/ACM-MM-Evaluation-Dataset"):
    #     for name in files:
    #         if name.split("_")[0] == "256":
    #             files_256.append(path + "/" + name)
    #         elif name.split("_")[0] == "512":
    #             files_512.append(path + "/" + name)
    #         elif name.split("_")[0] == "1024":
    #             files_1024.append(path + "/" + name)
    #         elif name.split("_")[0] == "2048":
    #             files_2048.append(path + "/" + name)
    #         elif name.split("_")[0] == "4096":
    #             files_4096.append(path + "/" + name)
    #         elif name.split("_")[0] == "8192":
    #             files_8192.append(path + "/" + name)

    # file1 = open("/home/hpadmin/Downloads/testset_evaluation_512.txt","w")
    # eval_list = []
    # counter = 1
    # while True:
    #     print(counter)
    #     src_img = random.choice(files_512).replace('/media/hpadmin/Active_Disk/Datasets/ACM-MM-Evaluation-Dataset/', '')
    #     ref_img = random.choice(files_512).replace('/media/hpadmin/Active_Disk/Datasets/ACM-MM-Evaluation-Dataset/', '')
    #     if src_img + ref_img in eval_list:
    #         continue
    #     eval_list.append(src_img + ref_img)
    #     file1.writelines(src_img + " " + ref_img + "\n")

    #     if counter == 1000:
    #         break
    #     counter += 1

    # exit()

    times_arr = []
    total_tests = 0

    size = "512"
    ALG = "DPT"
    #file1 = open("/media/hpadmin/Active_Disk/Tests/Process_Time_Evaluation/testset_"+size+".txt")
    file1 = open("/media/NAS/Datasets/PAPER_METRIC/testset_evaluation_512.txt")
    #for i in range(total_tests):
    for line in file1.readlines():
        total_tests += 1
        print(total_tests)
        # src_img = random.choice(files_2048)
        # ref_img = random.choice(files_2048)
        s_p, r_p = line.strip().split(" ")
        src_img = '/media/NAS/Datasets/PAPER_METRIC/ACM-MM-Evaluation-Dataset/' + s_p
        ref_img = '/media/NAS/Datasets/PAPER_METRIC/ACM-MM-Evaluation-Dataset/' + r_p
        #src_img = '/media/hpadmin/Active_Disk/Datasets/ACM-MM-Evaluation-Dataset/interior/256_interior-07_dithering-4.png'
        #ref_img = '/media/hpadmin/Active_Disk/Datasets/ACM-MM-Evaluation-Dataset/abstract/256_abstract-03_dithering-4.png'
        #src_img = "/home/potechius/Downloads/ACM-MM-Evaluation-Dataset/abstract/4096_abstract-02.png"
        #ref_img = "/home/potechius/Downloads/ACM-MM-Evaluation-Dataset/abstract/4096_abstract-02.png"
        #src_img = "/home/potechius/Pictures/Screenshots/src.png"
        #ref_img = "/home/potechius/Pictures/Screenshots/ref.png"

        ref_pc = "data/pointclouds/athen_postprocessed_simp.ply"
        src_pc = "data/pointclouds/Wappentier_blue.ply"


        if ct_input == "img-img":
            src = Image(file_path=src_img)
            ref = Image(file_path=ref_img)
        elif ct_input == "pc-pc":
            loader = PLYLoader(src_pc)
            src = loader.get_mesh()
            loader2 = PLYLoader(ref_pc)
            ref = loader2.get_mesh()
        elif ct_input == "img-pc":
            src = Image(file_path=src_img)
            loader2 = PLYLoader(ref_pc)
            ref = loader2.get_mesh()
        elif ct_input == "pc-img":
            loader = PLYLoader(src_pc)
            src = loader.get_mesh()
            ref = Image(file_path=ref_img)
        else:
            print("Unsupported types or type combination")
            exit(1)

        ct = ColorTransfer(src, ref, ct_approach)
        output = ct.apply()

        times_arr.append(output["process_time"])
        print("TOTAL: " + str(output["process_time"]))

        if output["status_code"] == -1:
            print("\033[91mTypeError\033[0m: " + output["response"])
            exit()

        if ct_input == "img-img" or ct_input == "img-pc":
            #file_name = "/media/hpadmin/Active_Disk/Tests/Process_Time_Evaluation/"+ALG+"/"+ALG+"-"+size+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
            file_name = "/media/NAS/Datasets/PAPER_METRIC/"+ALG+"/"+s_p.split("/")[1].split(".")[0] +"__to__"+r_p.split("/")[1].split(".")[0]+".png"
            print(file_name)
            ou = np.concatenate((src.get_raw(), ref.get_raw(), output["object"].get_raw()), axis=1) 
            cv2.imwrite(file_name, cv2.cvtColor(ou, cv2.COLOR_BGR2RGB)*255)

            # with open("/media/hpadmin/Active_Disk/Tests/Process_Time_Evaluation/"+ALG+"/process_time_"+size+".txt","a") as file2:
            #     file2.writelines(str(round(output["process_time"],3)) + " " + s_p.split(".")[0] + " " + r_p.split(".")[0] + "\n")

            #output["object"].write(file_name)
            #output.show()
        elif ct_input == "pc-pc" or ct_input == "pc-img":
            out_loader = PLYLoader(mesh=output["object"])
            out_loader.write("/home/potechius/Downloads/"+ct_approach+".ply")

        #if total_tests == 1:
        #   exit()

    # calculate mean
    mean = sum(times_arr) / len(times_arr)

    # calculate std
    std = 0
    for t in times_arr:
        std += math.pow(t-mean, 2)
    std /= len(times_arr)
    std = math.sqrt(std)


    print("Averaged: " + str(round(mean,3)) + " +- " + str(round(std,3)))
    file1.close()
    #file2.close()

