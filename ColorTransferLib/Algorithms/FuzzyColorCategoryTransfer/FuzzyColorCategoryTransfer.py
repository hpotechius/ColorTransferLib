"""
Copyright 2022 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from numba import cuda
import math
import time
import cv2
import csv
import itertools
import open3d as o3d
from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.BaseOptions import BaseOptions
from ColorTransferLib.ImageProcessing.Image import Image as Img
from copy import deepcopy
from ColorTransferLib.Utils.Helper import check_compatibility
from .FaissKNeighbors import FaissKNeighbors
from pyhull.convex_hull import ConvexHull
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Color Transfer between Images
#   Author: Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley
#   Published in: IEEE Computer Graphics and Applications
#   Year of Publication: 2001
#
# Abstract:
#   We use a simple statistical analysis to impose one image's color characteristics on another. We can achieve color
#   correction by choosing an appropriate source image and apply its characteristic to another image.
#
# Link: https://doi.org/10.1109/38.946629
#
# Source:
#   - https://arnabfly.github.io/arnab_blog/fknn/
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class FuzzyColorCategoryTransfer:
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
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "FuzzyColorCategoryTransfer",
            "title": "...",
            "year": 2023,
            "abstract": "..."
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        start_time = time.time()

        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, FCCT.compatibility)

        # Preprocessing
        src_color = src.get_colors()
        print(src_color[0] * 255)
        src_color = cv2.cvtColor(src_color, cv2.COLOR_RGB2Lab)
        src_num = src_color.shape[0]
        ref_color = ref.get_colors()
        ref_color = cv2.cvtColor(ref_color, cv2.COLOR_RGB2Lab)
        ref_num = ref_color.shape[0]
        out_img = deepcopy(src)

        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(ref_color[:,0,:])
        # pc.colors = o3d.utility.Vector3dVector(ref.get_colors()[:,0,:])
        # o3d.io.write_point_cloud("/home/potechius/Downloads/ref_points.ply", pc)
        # exit()

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

        neigh = FaissKNeighbors(k=100)
        neigh.fit(colors, labels)

        src_preds, src_votes, src_distances = neigh.predict(src_color[:,0,:]) # colors are of size (number of colors, 1, dimension)
        ref_preds, ref_votes, ref_distances = neigh.predict(ref_color[:,0,:])

        src_membership = FCCT.__calc_membership(src_votes, src_distances, src_num, 2)
        ref_membership = FCCT.__calc_membership(ref_votes, ref_distances, ref_num, 2)

        # sort colors by their categories with membership
        color_cats_src, color_cats_src_ids, color_cats_src_mem = FCCT.__sort_by_category(src_preds, src_color, src_membership)
        color_cats_ref, color_cats_ref_ids, color_cats_ref_mem = FCCT.__sort_by_category(ref_preds, ref_color, ref_membership)

        volumes_src = []
        volumes_ref = []
        for c in color_terms:
            # Calculate convex hull for each class in src and ref
            c_hull_src, validity_src = FCCT.__calc_convex_hull(np.asarray(color_cats_src[c]))
            c_hull_ref, validity_ref = FCCT.__calc_convex_hull(np.asarray(color_cats_ref[c]))

            if validity_src:
                # Calculate barycenter and volume for each convex hull
                b_center_src, vol_src = FCCT.__calc_bary_center_volume(c_hull_src)
                # Resampling of the convex hulls as uniformly distributed point cloud
                pc_src = c_hull_src.sample_points_uniformly(number_of_points=1000)
                pc_src.colors = o3d.utility.Vector3dVector(np.full((1000,3), FCCT.color_samples[c]))
                # apply PCA to get eigenvectors and -values
                pca_src = PCA(n_components = 3)
                pca_src.fit_transform(np.asarray(pc_src.points))
                eigenvectors_src = pca_src.components_
                eigenvalues_src = pca_src.explained_variance_

                # pc = o3d.geometry.PointCloud()
                # pc.points = o3d.utility.Vector3dVector(color_cats_src[c])
                # pc.colors = o3d.utility.Vector3dVector(np.full((np.asarray(color_cats_src[c]).shape[0],3), FCCT.color_samples[c]))
                # o3d.io.write_point_cloud("/home/potechius/Downloads/src_points_"+c+".ply", pc)
                #exit()
            else:
                b_center_src = (0.0,0.0,0.0)
                eigenvectors_src = [(0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0)]
                eigenvalues_src = [0.0, 0.0, 0.0]
                vol_src = 0.0

            if validity_ref:
                # Calculate barycenter and volume for each convex hull
                b_center_ref, vol_ref = FCCT.__calc_bary_center_volume(c_hull_ref)
                pc_ref = c_hull_ref.sample_points_uniformly(number_of_points=1000)
                pc_ref.colors = o3d.utility.Vector3dVector(np.full((1000,3), FCCT.color_samples[c]))
                # apply PCA to get eigenvectors and -values
                pca_ref = PCA(n_components = 3)
                pca_ref.fit_transform(np.asarray(pc_ref.points))
                eigenvectors_ref = pca_ref.components_
                eigenvalues_ref = pca_ref.explained_variance_

                # pc = o3d.geometry.PointCloud()
                # pc.points = o3d.utility.Vector3dVector(color_cats_ref[c])
                # pc.colors = o3d.utility.Vector3dVector(np.full((np.asarray(color_cats_ref[c]).shape[0],3), FCCT.color_samples[c]))
                # o3d.io.write_point_cloud("/home/potechius/Downloads/ref_points_"+c+".ply", pc)

                # o3d.io.write_point_cloud("/home/potechius/Downloads/c_hull_resample_"+c+".ply", pc_ref)
                # FCCT.__write_convex_hull_mesh(colors=color_cats_ref[c], 
                #                               shape=np.asarray(color_cats_ref[c]).shape, 
                #                               path="/home/potechius/Downloads/convex_hull_ref_"+c+".ply", 
                #                               color=FCCT.color_samples[c],
                #                               color_space="LAB")
            else:
                b_center_ref = (0.0,0.0,0.0)
                eigenvectors_ref = [(0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0)]
                eigenvalues_ref = [0.0, 0.0, 0.0]
                vol_ref = 0.0

            # pc = c_hull_src.sample_points_uniformly(number_of_points=1000)
            # colors = np.full((1000,3), FCCT.color_samples[c])
            #pc.translate(-b_center_src)
            # pc.rotate(pca.components_, center=(0, 0, 0))
            # T = np.zeros((4, 4), float)
            # np.fill_diagonal(T, np.concatenate((1./eigenvalues, [1])))
            # pc.transform(T)
            # o3d.io.write_point_cloud("/home/potechius/Downloads/c_hull_resample_"+c+".ply", pc)

            #print(c)
            volumes_src.append((c, vol_src, b_center_src, eigenvectors_src, eigenvalues_src))
            volumes_ref.append((c, vol_ref, b_center_ref, eigenvectors_ref, eigenvalues_ref))

        # Get class pairs between src and ref based on the volume
        sorted_volumes_src = sorted(volumes_src, key=lambda x: x[1])
        sorted_volumes_ref = sorted(volumes_ref, key=lambda x: x[1])
        class_pairs = [[s, r] for s, r in zip(sorted_volumes_src,sorted_volumes_ref)]


        # Calculate translation between class pairs
        # Calculate rotation based on Eigenvectors
        color_translation = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_rotation = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_scaling = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        src_centers = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for cp in class_pairs:
            src_info, ref_info = cp
            src_class, src_vol, src_center, src_eigenvec, src_eigenval = src_info
            ref_class, ref_vol, ref_center, ref_eigenvec, ref_eigenval = ref_info
            print(src_class + " - " + ref_class)
            # check if volume is greater than 0
            if src_vol == 0.0 or ref_vol == 0.0:
                translation = np.array([0.0, 0.0, 0.0])
                rotation = np.zeros((3, 3), float)
                np.fill_diagonal(rotation, 1)
                scaling = np.array([1.0, 1.0, 1.0])
            else:
                translation = ref_center - src_center
                rotation_src_inv = src_eigenvec
                rotation_ref = np.transpose(ref_eigenvec)
                #rotation = rotation_ref.dot(rotation_src_inv)
                rotation = rotation_ref.dot(rotation_src_inv)
                scaling = ref_eigenval / src_eigenval


            color_translation[src_class] = translation
            color_rotation[src_class] = rotation
            color_scaling[src_class] = scaling
            src_centers[src_class] = np.asarray(src_center)

            if src_vol != 0.0 and ref_vol != 0.0:
                rotated_temp = np.add(np.asarray(color_cats_src[src_class]), -src_center)
                rotated_temp = src_eigenvec.dot(rotated_temp.T).T
                scaled_temp = rotated_temp * (1./src_eigenval)
                scaled_temp = scaled_temp * ref_eigenval
                rotated_temp = rotation_ref.dot(scaled_temp.T).T
                scaled_temp = np.add(rotated_temp, src_center)
                trans_temp = np.add(scaled_temp, translation)
                # FCCT.__write_convex_hull_mesh(colors=trans_temp, 
                #                     shape=np.asarray(color_cats_src[src_class]).shape, 
                #                     path="/home/potechius/Downloads/convex_hull_src_trans_rot_scal_"+src_class+".ply", 
                #                     color=FCCT.color_samples[src_class],
                #                     color_space="LAB")

        # exit()

        # Calculate isotropic scaling based on volume
        # TODO

        # Calculate anisotropic scaling based on Eigenvectors and Eigenvalues
        # TODO

        # [0] Color transfer points based on translation         
        #sorted_colors = FCCT.__apply_DT(color_cats_src, color_cats_src_ids, color_translation, color_terms)

        # [1] Color transfer points based on translation and membership        
        #sorted_colors = FCCT.__apply_DTM(src_color, color_terms, src_membership, color_translation)

        # [2] Color transfer points based on translation + rotation and membership
        sorted_colors = FCCT.__apply_DTMR(color_cats_src, color_cats_src_ids, color_cats_src_mem, color_translation, color_rotation, src_centers, color_terms)

        # [3] Color transfer points based on translation + rotation + isotropic scaling and membership
        # TODO

        # [4] Color transfer points based on translation + rotation + anisotropic scaling and membership
        #sorted_colors = FCCT.__apply_DTMRS(color_cats_src, color_cats_src_ids, color_cats_src_mem, color_translation, color_rotation, color_scaling, src_centers)

        # [5] Color transfer points based on translation + rotation + isotropic scaling + non-linear adaptation and membership
        # TODO

        output_colors = cv2.cvtColor(sorted_colors.astype("float32"), cv2.COLOR_Lab2RGB)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(sorted_colors[:,0,:])
        pc.colors = o3d.utility.Vector3dVector(output_colors[:,0,:])
        o3d.io.write_point_cloud("/home/potechius/Downloads/out_points.ply", pc)
        exit()




        # Adapt luminance
        #output_colors = np.concatenate((src_color[:,:,0][:,:,np.newaxis], sorted_colors[:,:,1:]), axis=2)
        #output_colors = cv2.cvtColor(output_colors.astype("float32"), cv2.COLOR_Lab2RGB)





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
    #
    # ------------------------------------------------------------------------------------------------------------------ 
    def __write_convex_hull_mesh(colors, shape, path, color, color_space="LAB"):
        if color_space == "RGB":
            ex = np.asarray(colors)[:, np.newaxis]
            cex = cv2.cvtColor(ex, cv2.COLOR_Lab2RGB)
            mesh, validity = FCCT.__calc_convex_hull(cex.squeeze())
        else:
            mesh, validity = FCCT.__calc_convex_hull(colors)

        if validity:
            colors = np.full(shape, color)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_triangle_mesh(filename=path, 
                                    mesh=mesh, 
                                    write_ascii=True,
                                    write_vertex_normals=False,
                                    write_vertex_colors=True,
                                    write_triangle_uvs=False)
        
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    def __calc_bary_center_volume(mesh):
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
        return mass_center, mesh_volume
    
    # ------------------------------------------------------------------------------------------------------------------
    # D - calculate transform direction
    # T - Translation
    # ------------------------------------------------------------------------------------------------------------------   
    @staticmethod
    def __apply_DT(color_cats_src, color_cats_src_ids, color_translation, color_terms):
        output_ids = np.empty([0, 1])
        output_colors = np.empty([0, 3])
        for idx, c in enumerate(color_terms):
            if np.asarray(color_cats_src[c]).shape[0] != 0:
                # save sorted ids for reversion of sorting 
                output_ids = np.concatenate((output_ids, np.asarray(color_cats_src_ids[c])[:, np.newaxis]))
                translated = np.add(np.asarray(color_cats_src[c]), color_translation[c])
                output_colors = np.concatenate((output_colors, translated))
        sort = np.argsort(output_ids, axis=0)
        sorted_colors = output_colors[sort]
        return sorted_colors
    # ------------------------------------------------------------------------------------------------------------------
    # D - calculate transform direction
    # T - Translation
    # M - Membership
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_DTM(src_color, color_terms, src_membership, color_translation):   
        output_colors = src_color[:,0,:]
        for idx, c in enumerate(color_terms):
            translated = np.concatenate((src_membership[:,idx][:,np.newaxis],src_membership[:,idx][:,np.newaxis],src_membership[:,idx][:,np.newaxis]), axis=1) * color_translation[c]
            output_colors = np.add(output_colors, translated)
        sorted_colors = output_colors[:, np.newaxis, :]
        return sorted_colors
    
    # ------------------------------------------------------------------------------------------------------------------
    # D - calculate transform direction
    # T - Translation
    # M - Membership
    # R - Rotation
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_DTMR(color_cats_src, color_cats_src_ids, color_cats_src_mem, color_translation, color_rotation, src_centers, color_terms):   
        output_ids = np.empty([0, 1])
        output_colors = np.empty([0, 3])
        for idx, c in enumerate(color_terms):
            # save sorted ids for reversion of sorting 
            output_ids = np.concatenate((output_ids, np.asarray(color_cats_src_ids[c])[:, np.newaxis]))

            rotated_temp = np.add(np.asarray(color_cats_src[c]), -src_centers[c])
            rotated_temp = color_rotation[c].dot(rotated_temp.T).T
            rotated_temp = np.add(rotated_temp, src_centers[c])

            output_colors_sub = rotated_temp
            for ci, cz in enumerate(color_terms):
                weighted_translation = np.asarray(color_translation[cz]) * np.concatenate((np.asarray(color_cats_src_mem[c])[:,ci][:,np.newaxis],np.asarray(color_cats_src_mem[c])[:,ci][:,np.newaxis],np.asarray(color_cats_src_mem[c])[:,ci][:,np.newaxis]), axis=1)
                output_colors_sub = np.add(output_colors_sub, weighted_translation)
            # print(np.asarray(color_cats_src[c]).shape)
            # print(np.asarray(color_cats_src_mem[c]).shape)
            # print(np.asarray(color_cats_src[c])[0])
            # print(np.asarray(color_cats_src_mem[c])[0])
            # print(weighted_translation[0])
            # exit()
            #translated = np.add(np.asarray(color_cats_src[c]), weighted_translation)

            output_colors = np.concatenate((output_colors, output_colors_sub))
        sort = np.argsort(output_ids, axis=0)
        sorted_colors = output_colors[sort]
        return sorted_colors


        # output_colors = src_color[:,0,:]
        # for idx, c in enumerate(color_terms):
        #     rotated_temp = np.add(output_colors, -src_centers[c])
        #     rotated_temp = color_rotation[c].dot(rotated_temp.T).T
        #     rotated_temp = np.add(rotated_temp, src_centers[c])
        
        #     translated = np.concatenate((src_membership[:,idx][:,np.newaxis],src_membership[:,idx][:,np.newaxis],src_membership[:,idx][:,np.newaxis]), axis=1) * color_translation[c]
        #     # print(color_translation[c].shape)
        #     # print(color_rotation[c].shape)
        #     # print(translated.shape)
        #     # exit()
        #     output_colors = np.add(output_colors, translated)
        # sorted_colors = output_colors[:, np.newaxis, :]
        # return sorted_colors
    
    # ------------------------------------------------------------------------------------------------------------------
    # D - calculate transform direction
    # T - Translation
    # M - Membership
    # R - Rotation
    # S - Scaling
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_DTMRS(src_color, color_terms, src_membership, color_translation, color_rotation, color_scaling, src_centers):   
        output_colors = src_color[:,0,:]
        for idx, c in enumerate(color_terms):
            rotated_temp = np.add(output_colors, -src_centers[c])
            rotated_temp = color_rotation[c].dot(rotated_temp.T).T
            scaled_temp = scaled_temp * color_scaling[c]
            rotated_temp = np.add(output_colors, src_centers[c])
        
            translated = np.concatenate((src_membership[:,idx][:,np.newaxis],src_membership[:,idx][:,np.newaxis],src_membership[:,idx][:,np.newaxis]), axis=1) * color_translation[c]
            # print(color_translation[c].shape)
            # print(color_rotation[c].shape)
            # print(translated.shape)
            # exit()
            output_colors = np.add(output_colors, translated)
        sorted_colors = output_colors[:, np.newaxis, :]
        return sorted_colors    

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    @staticmethod   
    def __sort_by_category(predictions, colors, membership):  
        color_terms = np.array(["Red", "Yellow", "Green", "Blue", "Black", "White", "Grey", "Orange", "Brown", "Pink", "Purple"])
        color_cats = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_mem = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_ids = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}

        for i, (pred, color, mem) in enumerate(zip(predictions, colors[:,0,:], membership)):
            color_cats[color_terms[int(pred)]].append(color)
            color_cats_mem[color_terms[int(pred)]].append(mem)
            color_cats_ids[color_terms[int(pred)]].append(i)
        return color_cats, color_cats_ids, color_cats_mem

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    @staticmethod   
    def __calc_convex_hull(points):
        # Check if array has enough points to create convex hull
        if len(points) <= 4:
            return None, False

        chull_red_src = ConvexHull(points)
        chull_red_src_p = np.expand_dims(chull_red_src.points, axis=1).astype("float32")
        chull_red_src_p = np.squeeze(chull_red_src_p)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(chull_red_src_p),
                                         triangles=o3d.utility.Vector3iVector(chull_red_src.vertices))
        return mesh, True
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod   
    def __calc_membership(votes, distances, num, m):
        epsilon = 1e-5 # prevents division by 0
        dd = 1.0 / (np.power(distances, 2.0/(m-1.0)) + epsilon)
        denominator = np.sum(dd, axis=1)

        class_num = 11
        membership = np.empty((num, 0))
        for c in range(class_num):
            class_votes = (votes == c).astype(int)
            numerator = np.sum(class_votes * dd, axis=1)
            mem_class = numerator / denominator
            membership = np.concatenate((membership, mem_class[:,np.newaxis]), axis=1)
        return membership

        # neigh = FaissKNeighbors(k=1)
        # neigh.fit(colors, labels)

        # with open('/home/potechius/Downloads/LUT.txt', 'w') as f:
        #     arrays = [np.fromiter(range(256), dtype=int), np.fromiter(range(256), dtype=int), np.fromiter(range(256), dtype=int)]
        #     f.write("red green blue label\n")
        #     for res in itertools.product(*arrays):
        #         print(res)
        #         test_c = cv2.cvtColor(np.asarray(res)[np.newaxis, np.newaxis, :].astype("float32"), cv2.COLOR_RGB2Lab)
        #         src_preds = neigh.predict(test_c[:,0,:])
        #         f.write(str(res[0]) + " " + str(res[1]) + " " + str(res[2]) + " " + str(src_preds[0]) + "\n")
        #         #break

        # exit()

FCCT = FuzzyColorCategoryTransfer