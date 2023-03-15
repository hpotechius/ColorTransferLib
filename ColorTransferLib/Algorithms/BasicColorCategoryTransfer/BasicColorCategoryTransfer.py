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
import cv2
import open3d as o3d
from pyhull.convex_hull import ConvexHull
import time
import faiss
from .FaissKNeighbors import FaissKNeighbors

# RayCasting: http://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html

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

    color_samples = {
        "Red": np.array([1.0,0.0,0.0]),
        "Yellow":np.array([1.0,1.0,0.0]),
        "Green": np.array([0.0,1.0,0.0]),
        "Blue": np.array([0.0,0.0,1.0]),
        "Black": np.array([0.0,0.0,0.0]),
        "White": np.array([1.0,1.0,1.0]),
        "Grey": np.array([0.5,0.5,0.5]),
        "Orange": np.array([1.0,0.5,0.0]),
        "Brown": np.array([0.4,0.2,0.1]),
        "Pink": np.array([0.85,0.5,0.75]),
        "Purple": np.array([0.4,0.01,0.77]),
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
        start_time = time.time()
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, BasicColorCategoryTransfer.compatibility)

        # Preprocessing
        src_color = src.get_colors()
        src_color = cv2.cvtColor(src_color, cv2.COLOR_RGB2Lab)
        ref_color = ref.get_colors()
        ref_color = cv2.cvtColor(ref_color, cv2.COLOR_RGB2Lab)
        out_img = deepcopy(src)
        print("Number of points: " + str(src_color.shape))

        # Read Color Dataset
        time_stamp = time.time()
        color_terms = np.array(["Red", "Yellow", "Green", "Blue", "Black", "White", "Grey", "Orange", "Brown", "Pink", "Purple"])
        color_mapping = []
        with open("Models/BasicColorCategoryTransfer/colormapping.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    color_mapping.append([float(row[0]), float(row[1]), float(row[2]), float(np.where(color_terms == row[3])[0][0])])
                line_count += 1

        color_mapping = np.asarray(color_mapping)
        colors = color_mapping[:,:3] / 255
        colors = np.expand_dims(colors, axis=1).astype("float32")
        colors = cv2.cvtColor(colors, cv2.COLOR_RGB2Lab)
        colors = np.squeeze(colors)
        labels = color_mapping[:,3].astype("int64")

        # Train Classifier
        time_stamp = time.time()
        neigh = FaissKNeighbors(k=1)
        neigh.fit(colors, labels)
        print("Time - Classifier:", round(time.time() - time_stamp, 4))

        time_stamp = time.time()
        # predic src color label
        src_preds = neigh.predict(src_color[:,0,:]) # colors are of size (number of colors, 1, dimension)
        # predic ref color label
        ref_preds = neigh.predict(ref_color[:,0,:])
        print("Time - Prediction:", round(time.time() - time_stamp, 4))

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


        color_cats_src_ids = {
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

        color_cats_ref = {
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

        # save categorise image
        # for i, (pred, color) in enumerate(zip(ref_preds, ref_color[:,0,:])):
        #     ref_color[i,0,:] = color_samples[color_terms[int(pred)]]
        # out_img.set_colors(ref_color)

        # output = {2048
        #     "status_code": 0,
        #     "response": "",
        #     "object": out_img
        # }

        # return output

        time_stamp = time.time()
        for i, (pred, color) in enumerate(zip(src_preds, src_color[:,0,:])):
            color_cats_src[color_terms[int(pred)]].append(color)
            color_cats_src_ids[color_terms[int(pred)]].append(i)

        for pred, color in zip(ref_preds, ref_color[:,0,:]):
            color_cats_ref[color_terms[int(pred)]].append(color)
        print("Time - Sorting:", round(time.time() - time_stamp, 4))


        output_colors = np.empty([0, 3])
        output_ids = np.empty([0, 1])
        for color_cat in color_cats_src.keys():
            print(color_cat)
            output_ids = np.concatenate((output_ids, np.asarray(color_cats_src_ids[color_cat])[:, np.newaxis]))
            # Create Convex Hulls
            # Check if color categories are not empty
            if len(color_cats_src[color_cat]) >= 4 and len(color_cats_ref[color_cat]) < 4:
                output_colors = np.concatenate((output_colors, np.asarray(color_cats_src[color_cat])))
                continue
            elif len(color_cats_src[color_cat]) == 0:
                continue
            elif len(color_cats_src[color_cat]) < 4:
                output_colors = np.concatenate((output_colors, np.asarray(color_cats_src[color_cat])))
                continue


            if BCC.__check_identity(np.asarray(color_cats_src[color_cat])) or BCC.__check_identity(np.asarray(color_cats_ref[color_cat])):
                output_colors = np.concatenate((output_colors, np.asarray(color_cats_src[color_cat])))
                continue
            #exit()

            #time_stamp = time.time()
            #print(len(color_cats_src[color_cat]))
            mesh_src = BasicColorCategoryTransfer.__calc_convex_hull(color_cats_src[color_cat])
            mesh_ref = BasicColorCategoryTransfer.__calc_convex_hull(color_cats_ref[color_cat])
            #print("Time - Convex Hull:", round(time.time() - time_stamp, 4))

            #BCC.__check_coplanarity(np.asarray(color_cats_src_ids[color_cat]))

            mass_center_src = BasicColorCategoryTransfer.__calc_gravitational_center(mesh_src)
            mass_center_ref = BasicColorCategoryTransfer.__calc_gravitational_center(mesh_ref)

            # BasicColorCategoryTransfer.__write_convex_hull_mesh(colors=color_cats_src[color_cat], 
            #                                                     shape=np.asarray(color_cats_src[color_cat]).shape, 
            #                                                     path="/home/potechius/Downloads/convex_hull_"+color_cat+".ply", 
            #                                                     color=BasicColorCategoryTransfer.color_samples[color_cat],
            #                                                     color_space="LAB")
            
            # BasicColorCategoryTransfer.__write_convex_hull_mesh(colors=color_cats_ref[color_cat], 
            #                                                     shape=np.asarray(color_cats_ref[color_cat]).shape, 
            #                                                     path="/home/potechius/Downloads/convex_hull_"+color_cat+".ply", 
            #                                                     color=BasicColorCategoryTransfer.color_samples[color_cat],
            #                                                     color_space="LAB")


            # calculate intersection between convex hull and ray consisting of the center of mass and the given pixel color
            inter_src = BasicColorCategoryTransfer.__calc_line_mesh_intersection(mesh_src, color_cats_src[color_cat] - mass_center_src, mass_center_src)
            inter_ref = BasicColorCategoryTransfer.__calc_line_mesh_intersection(mesh_ref, color_cats_src[color_cat] - mass_center_src, mass_center_ref)

            # Color Transfer
            output_colors = BasicColorCategoryTransfer.__transfer_colors(output_colors=output_colors, 
                                                                         colors=color_cats_src[color_cat], 
                                                                         mass_center_src=mass_center_src, 
                                                                         mass_center_ref=mass_center_ref, 
                                                                         dist_src=inter_src['t_hit'], 
                                                                         dist_ref=inter_ref['t_hit'])
  
        sort = np.argsort(output_ids, axis=0)
        sorted_colors = output_colors[sort]

        output_colors = cv2.cvtColor(sorted_colors.astype("float32"), cv2.COLOR_Lab2RGB)
        output_colors = np.clip(output_colors, 0, 1)


        out_img.set_colors(output_colors)

        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output

    # ------------------------------------------------------------------------------------------------------------------
    # checks if the given data does not lie on a plane -> this would lead to a convex hull with volume = 0
    # ------------------------------------------------------------------------------------------------------------------  
    def __check_coplanarity(data):
        print(data.shape)
        for val in data:
            print(val)
            exit()

    # ------------------------------------------------------------------------------------------------------------------
    # checks if the given data contains at least four different values for creating a convex hull with volume > 0
    # returns true if the data does not contain at least four different values
    # ------------------------------------------------------------------------------------------------------------------  
    def __check_identity(data):
        unique_val = np.unique(data, axis=0)
        if unique_val.shape[0] < 4:
            return True
        else:
            return False

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    def __calc_gravitational_center(mesh):
        # calculate gravitational center of convex hull
        # (1) get geometrical center
        coord_center = mesh.get_center()
        #meshw = meshw.translate(-coord_center)
        # (2) iterate over triangles and calculate tetrahaedon mass and center using the coordinate center of the whole mesh
        vol_center = 0
        vertices = np.asarray(mesh.vertices)
        mesh_volume = 0
        for tri in mesh.triangles:
            # calculate center
            pos0 = vertices[tri[0]]
            pos1 = vertices[tri[1]]
            pos2 = vertices[tri[2]]
            pos3 = coord_center
            geo_center = np.sum([pos0, pos1, pos2, pos3], axis=0) / 4
            # calculate volume using the formula:
            # V = |(a-b) * ((b-d) x (c-d))| / 6
            vol = np.abs(np.dot((pos0 - pos3), np.cross((pos1 - pos3), (pos2-pos3)))) / 6
            vol_center += vol * geo_center
            mesh_volume += vol
        # (3) calculate mesh center based on:
        # mass_center = sum(tetra_volumes*tetra_centers)/sum(volumes)
        mass_center = vol_center / mesh_volume
        return mass_center
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    def __calc_convex_hull(points):
        chull_red_src = ConvexHull(points)
        chull_red_src_p = np.expand_dims(chull_red_src.points, axis=1).astype("float32")
        #chull_red_src_p = cv2.cvtColor(chull_red_src_p, cv2.COLOR_Lab2RGB)
        chull_red_src_p = np.squeeze(chull_red_src_p)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(chull_red_src_p),
                                          triangles=o3d.utility.Vector3iVector(chull_red_src.vertices))
        return mesh
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------    
    def __calc_line_mesh_intersection(mesh, directions, mass_center):
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_id = scene.add_triangles(mesh)

        # Note: directions have to be normalized in order to get the correct ray cast distance
        norms = np.linalg.norm(directions, axis=1)[:, np.newaxis]
        norms_ext = np.concatenate((norms, norms, norms), axis= 1)
        norm_directions = directions / norms_ext

        rays_src = np.concatenate((np.full(np.asarray(directions).shape, mass_center), norm_directions), axis=1)

        rays_src_tensor = o3d.core.Tensor(rays_src, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_src_tensor)
        return ans
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------ 
    def __write_convex_hull_mesh(colors, shape, path, color, color_space="LAB"):
        if color_space == "RGB":
            ex = np.asarray(colors)[:, np.newaxis]
            cex = cv2.cvtColor(ex, cv2.COLOR_Lab2RGB)
            mesh = BasicColorCategoryTransfer.__calc_convex_hull(cex.squeeze())
        else:
            mesh = BasicColorCategoryTransfer.__calc_convex_hull(colors)

        colors = np.full(shape, color)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_triangle_mesh(filename=path, 
                                   mesh=mesh, 
                                   write_ascii=True,
                                   write_vertex_normals=False,
                                   write_vertex_colors=True,
                                   write_triangle_uvs=False)
        
    # ------------------------------------------------------------------------------------------------------------------
    #coplanarity
    # ------------------------------------------------------------------------------------------------------------------         
    def __transfer_colors(output_colors, colors, mass_center_src, mass_center_ref, dist_src, dist_ref):
        point_dir = colors - mass_center_src 
        point_dist = np.linalg.norm(point_dir, axis=1)
        intersection_dist_src = dist_src.numpy()
        relative_point_dist = (point_dist / intersection_dist_src)[:, np.newaxis]

        point_dist_ext = point_dist[:, np.newaxis]
        norm_point_dir = point_dir / np.concatenate((point_dist_ext, point_dist_ext, point_dist_ext), axis= 1)
        intersection_dist_ref = dist_ref.numpy()[:, np.newaxis]

        shift = norm_point_dir * np.concatenate((intersection_dist_ref, intersection_dist_ref, intersection_dist_ref), axis= 1) * np.concatenate((relative_point_dist, relative_point_dist, relative_point_dist), axis= 1)
        out = shift + mass_center_ref
        output_colors = np.concatenate((output_colors, out))
        return output_colors
    
BCC = BasicColorCategoryTransfer