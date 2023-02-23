"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from numba import cuda
import math
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.BaseOptions import BaseOptions
from ColorTransferLib.ImageProcessing.Image import Image as Img
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility
import csv
from scipy.spatial import Voronoi
from sklearn.neighbors import KNeighborsClassifier
import cv2
import open3d as o3d
from pyhull import qconvex, qdelaunay, qvoronoi
from pyhull.voronoi import VoronoiTess

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: A framework for transfer colors based on the basic color categories
#   Author: Youngha Chang, Suguru Saito, Masayuki Nakajima
#   Published in: Proceedings Computer Graphics International
#   Year of Publication: 2003
#
# Abstract:
#   Usually, paintings are more appealing than photographic images. This is because paintings have styles. This style 
#   can be distinguished by looking at elements such as motif, color, shape deformation and brush texture. We focus on 
#   the effect of "color" element and devise a method for transforming the color of an input photograph according to a 
#   reference painting. To do this, we consider basic color category concepts in the color transformation process. By 
#   doing so, we achieve large but natural color transformations of an image.
#
# Link: ...
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class BasicColorCategoryTransfer:
    compatibility = {
        "src": ["Image", "Mesh"],
        "ref": ["Image", "Mesh"]
    }
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # HOST METHODS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "BasicColorCategoryTransfer",
            "title": "A Framework for Transfer Colors Based on the Basic Color Categories",
            "year": 2003,
            "abstract": "Usually, paintings are more appealing than photographic images. This is because paintings have "
                        "styles. This style can be distinguished by looking at elements such as motif, color, shape "
                        "deformation and brush texture. We focus on the effect of color element and devise a method "
                        "for transforming the color of an input photograph according to a reference painting. To do "
                        "this, we consider basic color category concepts in the color transformation process. By doing "
                        "so, we achieve large but natural color transformations of an image."
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, BasicColorCategoryTransfer.compatibility)

        # Preprocessing
        src_color = src.get_colors()
        src_color = cv2.cvtColor(src_color, cv2.COLOR_BGR2Lab)
        ref_color = ref.get_colors()
        ref_color = cv2.cvtColor(ref_color, cv2.COLOR_BGR2Lab)
        out_img = deepcopy(src)
        


        color_terms = np.array(["Red", "Yellow", "Green", "Blue", "Black", "White", "Grey", "Orange", "Brown", "Pink", "Purple"])
        color_mapping = []
        with open("Models/BasicColorCategoryTransfer/colormapping.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    #color_terms[row[3]].append([float(row[0]), float(row[1]), float(row[2])])
                    color_mapping.append([float(row[0]), float(row[1]), float(row[2]), float(np.where(color_terms == row[3])[0][0])])
                line_count += 1

        color_mapping = np.asarray(color_mapping)
        # sort labels
        #label_sort = color_mapping[:,3].argsort()
        # sort compelte array based on label sort
        #color_mapping = color_mapping[label_sort]

        #red_list = color_mapping[color_mapping[3] == 1]
        #print(red_list)
        #print(color_mapping.shape)

        # convert BGR ro CIEL*a*b*
        colors = np.expand_dims(color_mapping[:,:3], 0) # add dimension to generate array with width and height: (500, 3) -> (1, 500, 3)
        colors = colors.astype("float32")
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2Lab)
        colors = colors[0] # remove added dimension
        labels = color_mapping[:,3]

        qv = np.asarray(qvoronoi("o", colors))
        print(colors.shape)
        #qv =  VoronoiTess(colors)
        print(qv.shape)
        #print(np.asarray(qv.vertices).shape)

        exit()

        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(colors, labels)
        #print(vor.point_region)

        # colors are of size (number of colors, 1, dimension)
        src_preds = neigh.predict(src_color[:,0,:])
        print(src_preds.shape)
        print(src.get_colors()[0,0,:] * 255)
        print(src_preds[0])
        #ref_preds = neigh.predict(ref_color[:,0,:])

        color_cats_src = {
            "Red": [],
            "Yellow": [],
            "Green": [],
            "Blue": [],
            "Black": [],
            "White": [],
            "Grey": [],
            "Orange": [],
            "Brown": [],
            "Pink": [],
            "Purple": []
        }


        for pred, color in zip(src_preds, src_color[:,0,:]):
            color_cats_src[color_terms[pred]].append(color)

        # get convex hull for each category
        exit()


        # # [1] Copy source and reference to GPU and create output
        # device_src = cuda.to_device(src_color)
        # device_ref = cuda.to_device(ref_color)
        # device_out = cuda.device_array(src_color.shape)

        # # [2] Convert RGB to lab color space
        # if opt.colorspace == "lalphabeta":
        #     lab_src = ColorSpaces.rgb_to_lab(device_src)
        #     lab_ref = ColorSpaces.rgb_to_lab(device_ref)
        # elif opt.colorspace == "rgb":
        #     lab_src = device_src
        #     lab_ref = device_ref

        # # [3] Get mean, standard deviation and ratio of standard deviations
        # mean_lab_src = cuda.to_device(np.mean(lab_src, axis=(0, 1)))
        # std_lab_src = np.std(lab_src, axis=(0, 1))
        # mean_lab_ref = cuda.to_device(np.mean(lab_ref, axis=(0, 1)))
        # std_lab_ref = np.std(lab_ref, axis=(0, 1))

        # device_div_std = cuda.to_device(std_lab_ref / std_lab_src)

        # # [4] Apply Global Color Transfer on GPU
        # threadsperblock = (32, 32)
        # blockspergrid_x = int(math.ceil(device_out.shape[0] / threadsperblock[0]))
        # blockspergrid_y = int(math.ceil(device_out.shape[1] / threadsperblock[1]))
        # blockspergrid = (blockspergrid_x, blockspergrid_y)
        # GlobalColorTransfer.__kernel_apply[blockspergrid, threadsperblock](lab_src,
        #                                                                    device_out,
        #                                                                    mean_lab_src,
        #                                                                    mean_lab_ref,
        #                                                                    device_div_std)

        # # [5] Convert lab to RGB color space
        # if opt.colorspace == "lalphabeta":
        #     device_out = ColorSpaces.lab_to_rgb(device_out)

        # # [6] Copy output from GPU to RAM
        # out_colors = device_out.copy_to_host()
        # out_colors = np.clip(out_colors, 0, 1)

        # out_img.set_colors(out_colors)

        # output = {
        #     "status_code": 0,
        #     "response": "",
        #     "object": out_img
        # }

        return output
