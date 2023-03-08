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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR

from ColorTransferLib.MeshProcessing.PLYLoader import PLYLoader
from ColorTransferLib.MeshProcessing.Mesh import Mesh
from ColorTransferLib.ImageProcessing.Image import Image
from ColorTransferLib.ColorTransfer import ColorTransfer

from ColorTransferLib.Utils.Math import get_random_3x3rotation_matrix
from ColorTransferLib.Evaluation.SSIM.SSIM import SSIM
from ColorTransferLib.Evaluation.PSNR.PSNR import PSNR
from ColorTransferLib.Evaluation.HistogramIntersection.HistogramIntersection import HistogramIntersection
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
            "PdfColorTransfer",
            "NeuralStyleTransfer",
            "HistogramAnalogy",
            "CamsTransfer",
            "DeepPhotoStyleTransfer",
            "TpsColorTransfer",
            "FuzzyColorTransfer",
            "GmmEmColorTransfer",
            "Eb3dColorTransfer", # no image support?
            "PSNetStyleTransfer",
            "MongeKLColorTransfer",
            "HistoGAN",
            "BasicColorCategoryTransfer"]

    ct_approach = "TpsColorTransfer"
    ct_input = "img-img"

    files_256 = []
    files_512 = []
    files_1024 = []
    files_2048 = []
    files_4096 = []
    files_8192 = []
    for path, subdirs, files in os.walk("/home/potechius/Downloads/ACM-MM-Evaluation-Dataset"):
        for name in files:
            if name.split("_")[0] == "256":
                files_256.append(path + "/" + name)
            elif name.split("_")[0] == "512":
                files_512.append(path + "/" + name)
            elif name.split("_")[0] == "1024":
                files_1024.append(path + "/" + name)
            elif name.split("_")[0] == "2048":
                files_2048.append(path + "/" + name)
            elif name.split("_")[0] == "4096":
                files_4096.append(path + "/" + name)
            elif name.split("_")[0] == "8192":
                files_8192.append(path + "/" + name)
        #    print(os.path.join(path, name))

    times = 0
    total_tests = 1
    for i in range(total_tests):
        # src_img = random.choice(files_2048)
        # ref_img = random.choice(files_2048)
        src_img = "/home/potechius/Downloads/ACM-MM-Evaluation-Dataset/city/2048_city-06.png"
        ref_img = "/home/potechius/Downloads/ACM-MM-Evaluation-Dataset/city/2048_city-04.png"
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

        start_time = time.time()
        ct = ColorTransfer(src, ref, ct_approach)
        output = ct.apply()        
        end_time = time.time() - start_time
        print("TOTAL: " + str(end_time))
        times += end_time

        if output["status_code"] == -1:
            print("\033[91mTypeError\033[0m: " + output["response"])
            exit()



        if ct_input == "img-img" or ct_input == "img-pc":
            output["object"].write("/home/potechius/Downloads/"+ct_approach+"_"+str(i).zfill(3)+".png")
            #output.show()
        elif ct_input == "pc-pc" or ct_input == "pc-img":
            out_loader = PLYLoader(mesh=output["object"])
            out_loader.write("/home/potechius/Downloads/"+ct_approach+".ply")

    print("Averaged: " + str(times / total_tests))

