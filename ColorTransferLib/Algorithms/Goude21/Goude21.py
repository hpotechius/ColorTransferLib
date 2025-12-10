"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import fractional_matrix_power
from copy import deepcopy
import time

from ColorTransferLib.Utils.ColorSpaces import ColorSpaces

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: Example-Based Colour Transfer for 3D Point Clouds
#   Author: Ific Goudé, Rémi Cozot, Olivier Le Meur, Kadi Bouatouch
#   Published in: Computer Graphics Forum, Volume 40
#   Year of Publication: 2021
#
# Abstract:
#   Example-based colour transfer between images, which has raised a lot of interest in the past decades, consists of
#   transferring the colour of an image to another one. Many methods based on colour distributions have been proposed,
#   and more recently, the efficiency of neural networks has been demonstrated again for colour transfer problems. In
#   this paper, we propose a new pipeline with methods adapted from the image domain to automatically transfer the
#   colour from a target point cloud to an input point cloud. These colour transfer methods are based on colour
#   distributions and account for the geometry of the point clouds to produce a coherent result. The proposed methods
#   rely on simple statistical analysis, are effective, and succeed in transferring the colour style from one point
#   cloud to another. The qualitative results of the colour transfers are evaluated and compared with existing methods.
#
# Link: https://doi.org/10.1111/cgf.14388
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Goude21:
    # ------------------------------------------------------------------------------------------------------------------
    # Checks source and reference compatibility
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        output = {
            "status_code": 0,
            "response": "",
            "object": None,
            "process_time": 0
        }

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "PointCloud" and ref.get_type() == "PointCloud":
            out_obj = Goude21.__apply_pointcloud(src, ref, opt)
            output["object"] = out_obj
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1


        output["process_time"] = time.time() - start_time


        return output
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def IGD(src, ref, pca_enabled):
        # Convert colors from RGB to lalphabeta color space
        src_color_lab = ColorSpaces.rgb_to_lalphabeta(src.get_colors()).reshape(src.get_num_vertices(), 3)
        ref_color_lab = ColorSpaces.rgb_to_lalphabeta(ref.get_colors()).reshape(ref.get_num_vertices(), 3)

        # convert 3D normals to PCA adjusted 6D normals
        norma_src = src.get_vertex_normals().reshape(src.get_num_vertices(), 3)
        norma_ref = ref.get_vertex_normals().reshape(ref.get_num_vertices(), 3)

        if pca_enabled:
            pca = PCA(n_components=3)
            pca.fit(norma_src)
            src_pca_nx = np.einsum('ij, ij->i', norma_src, np.full((src.get_num_vertices(), 3), pca.components_[0]))
            src_pca_ny = np.einsum('ij, ij->i', norma_src, np.full((src.get_num_vertices(), 3), pca.components_[1]))
            src_pca_nz = np.einsum('ij, ij->i', norma_src, np.full((src.get_num_vertices(), 3), pca.components_[2]))
            norma_src = np.stack((src_pca_nx, src_pca_ny, src_pca_nz), axis=1)

            pca_ref = PCA(n_components=3)
            pca_ref.fit(norma_ref)
            ref_pca_nx = np.einsum('ij, ij->i', norma_ref, np.full((ref.get_num_vertices(), 3), pca_ref.components_[0]))
            ref_pca_ny = np.einsum('ij, ij->i', norma_ref, np.full((ref.get_num_vertices(), 3), pca_ref.components_[1]))
            ref_pca_nz = np.einsum('ij, ij->i', norma_ref, np.full((ref.get_num_vertices(), 3), pca_ref.components_[2]))
            norma_ref = np.stack((ref_pca_nx, ref_pca_ny, ref_pca_nz), axis=1)

        src_normal_6d = np.concatenate(
            (np.where(norma_src > 0.0, 0.0, np.abs(norma_src)), np.where(norma_src < 0.0, 0.0, np.abs(norma_src))),
            axis=1)
        ref_normal_6d = np.concatenate(
            (np.where(norma_ref > 0.0, 0.0, np.abs(norma_ref)), np.where(norma_ref < 0.0, 0.0, np.abs(norma_ref))),
            axis=1)

        # contains color and normal information per vertex -> [[cx, cy, cz, nx, ny, nz], ...]
        src_in_raw = np.concatenate((src_color_lab, src_normal_6d), axis=1).reshape(src.get_num_vertices(), 9)
        ref_in_raw = np.concatenate((ref_color_lab, ref_normal_6d), axis=1).reshape(ref.get_num_vertices(), 9)

        src_sum_normal = np.sum(src_in_raw[:, 3:], axis=0)
        ref_sum_normal = np.sum(ref_in_raw[:, 3:], axis=0)

        # calculate means and standard deviations
        src_mean_nx = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 3:4], src_in_raw[:, :3]), axis=0) / src_sum_normal[0]
        src_mean_ny = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 4:5], src_in_raw[:, :3]), axis=0) / src_sum_normal[1]
        src_mean_nz = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 5:6], src_in_raw[:, :3]), axis=0) / src_sum_normal[2]
        src_mean_px = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 6:7], src_in_raw[:, :3]), axis=0) / src_sum_normal[3]
        src_mean_py = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 7:8], src_in_raw[:, :3]), axis=0) / src_sum_normal[4]
        src_mean_pz = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 8:9], src_in_raw[:, :3]), axis=0) / src_sum_normal[5]

        src_std_nx = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 3:4], np.power(src_in_raw[:, :3] - src_mean_nx, 2)),
                            axis=0) / src_sum_normal[0]
        src_std_ny = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 4:5], np.power(src_in_raw[:, :3] - src_mean_ny, 2)),
                            axis=0) / src_sum_normal[1]
        src_std_nz = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 5:6], np.power(src_in_raw[:, :3] - src_mean_nz, 2)),
                            axis=0) / src_sum_normal[2]
        src_std_px = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 6:7], np.power(src_in_raw[:, :3] - src_mean_px, 2)),
                            axis=0) / src_sum_normal[3]
        src_std_py = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 7:8], np.power(src_in_raw[:, :3] - src_mean_py, 2)),
                            axis=0) / src_sum_normal[4]
        src_std_pz = np.sum(np.einsum('ik, ij->ij', src_in_raw[:, 8:9], np.power(src_in_raw[:, :3] - src_mean_pz, 2)),
                            axis=0) / src_sum_normal[5]

        ref_mean_nx = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 3:4], ref_in_raw[:, :3]), axis=0) / ref_sum_normal[0]
        ref_mean_ny = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 4:5], ref_in_raw[:, :3]), axis=0) / ref_sum_normal[1]
        ref_mean_nz = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 5:6], ref_in_raw[:, :3]), axis=0) / ref_sum_normal[2]
        ref_mean_px = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 6:7], ref_in_raw[:, :3]), axis=0) / ref_sum_normal[3]
        ref_mean_py = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 7:8], ref_in_raw[:, :3]), axis=0) / ref_sum_normal[4]
        ref_mean_pz = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 8:9], ref_in_raw[:, :3]), axis=0) / ref_sum_normal[5]

        ref_std_nx = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 3:4], np.power(ref_in_raw[:, :3] - ref_mean_nx, 2)),
                            axis=0) / ref_sum_normal[0]
        ref_std_ny = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 4:5], np.power(ref_in_raw[:, :3] - ref_mean_ny, 2)),
                            axis=0) / ref_sum_normal[1]
        ref_std_nz = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 5:6], np.power(ref_in_raw[:, :3] - ref_mean_nz, 2)),
                            axis=0) / ref_sum_normal[2]
        ref_std_px = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 6:7], np.power(ref_in_raw[:, :3] - ref_mean_px, 2)),
                            axis=0) / ref_sum_normal[3]
        ref_std_py = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 7:8], np.power(ref_in_raw[:, :3] - ref_mean_py, 2)),
                            axis=0) / ref_sum_normal[4]
        ref_std_pz = np.sum(np.einsum('ik, ij->ij', ref_in_raw[:, 8:9], np.power(ref_in_raw[:, :3] - ref_mean_pz, 2)),
                            axis=0) / ref_sum_normal[5]


        # Standardabweichungen: numerisch absichern (keine 0-Division)
        eps = 1e-6
        src_std_nx = np.where(np.abs(src_std_nx) < eps, eps, src_std_nx)
        src_std_ny = np.where(np.abs(src_std_ny) < eps, eps, src_std_ny)
        src_std_nz = np.where(np.abs(src_std_nz) < eps, eps, src_std_nz)
        src_std_px = np.where(np.abs(src_std_px) < eps, eps, src_std_px)
        src_std_py = np.where(np.abs(src_std_py) < eps, eps, src_std_py)
        src_std_pz = np.where(np.abs(src_std_pz) < eps, eps, src_std_pz)

        src_std_nx = np.sqrt(src_std_nx)
        src_std_ny = np.sqrt(src_std_ny)
        src_std_nz = np.sqrt(src_std_nz)
        src_std_px = np.sqrt(src_std_px)
        src_std_py = np.sqrt(src_std_py)
        src_std_pz = np.sqrt(src_std_pz)

        ref_std_nx = np.sqrt(ref_std_nx)
        ref_std_ny = np.sqrt(ref_std_ny)
        ref_std_nz = np.sqrt(ref_std_nz)
        ref_std_px = np.sqrt(ref_std_px)
        ref_std_py = np.sqrt(ref_std_py)
        ref_std_pz = np.sqrt(ref_std_pz)

        src_out_nx = (src_in_raw[:, :3] - src_mean_nx) * np.full((src.get_num_vertices(), 3),
                                                                 (ref_std_nx * np.reciprocal(src_std_nx))) + ref_mean_nx
        src_out_ny = (src_in_raw[:, :3] - src_mean_ny) * np.full((src.get_num_vertices(), 3),
                                                                 (ref_std_ny * np.reciprocal(src_std_ny))) + ref_mean_ny
        src_out_nz = (src_in_raw[:, :3] - src_mean_nz) * np.full((src.get_num_vertices(), 3),
                                                                 (ref_std_nz * np.reciprocal(src_std_nz))) + ref_mean_nz
        src_out_px = (src_in_raw[:, :3] - src_mean_px) * np.full((src.get_num_vertices(), 3),
                                                                 (ref_std_px * np.reciprocal(src_std_px))) + ref_mean_px
        src_out_py = (src_in_raw[:, :3] - src_mean_py) * np.full((src.get_num_vertices(), 3),
                                                                 (ref_std_py * np.reciprocal(src_std_py))) + ref_mean_py
        src_out_pz = (src_in_raw[:, :3] - src_mean_pz) * np.full((src.get_num_vertices(), 3),
                                                                 (ref_std_pz * np.reciprocal(src_std_pz))) + ref_mean_pz

        src_out_l_temp = np.concatenate((src_out_nx[:, 0:1], src_out_ny[:, 0:1], src_out_nz[:, 0:1], src_out_px[:, 0:1],
                                         src_out_py[:, 0:1], src_out_pz[:, 0:1]), axis=1)
        src_out_a_temp = np.concatenate((src_out_nx[:, 1:2], src_out_ny[:, 1:2], src_out_nz[:, 1:2], src_out_px[:, 1:2],
                                         src_out_py[:, 1:2], src_out_pz[:, 1:2]), axis=1)
        src_out_b_temp = np.concatenate((src_out_nx[:, 2:3], src_out_ny[:, 2:3], src_out_nz[:, 2:3], src_out_px[:, 2:3],
                                         src_out_py[:, 2:3], src_out_pz[:, 2:3]), axis=1)

        # 6D-Normalgewichte pro Punkt normieren (Summe = 1) → Identität, wenn src==ref
        w = src_normal_6d
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_norm = np.where(w_sum > eps, w / w_sum, 0.0)

        src_out_l = np.einsum('ij, ij-> i', w_norm, src_out_l_temp)
        src_out_a = np.einsum('ij, ij-> i', w_norm, src_out_a_temp)
        src_out_b = np.einsum('ij, ij-> i', w_norm, src_out_b_temp)

        out = np.concatenate((src_out_l.reshape(src.get_num_vertices(), 1),
                              src_out_a.reshape(src.get_num_vertices(), 1),
                              src_out_b.reshape(src.get_num_vertices(), 1)), axis=1)

        # [8] Convert to RGB (lαβ → RGB)
        lab_new = out.reshape(src.get_num_vertices(), 1, 3)
        lab_new = ColorSpaces.lalphabeta_to_rgb(lab_new)
        lab_new = np.clip(lab_new, 0.0, 1.0)

        return lab_new

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def MGD(src, ref, pca_enabled):
        # Convert colors from RGB to lalphabeta color space
        src_color_lab = ColorSpaces.rgb_to_lalphabeta(src.get_colors()).reshape(src.get_num_vertices(), 3)
        ref_color_lab = ColorSpaces.rgb_to_lalphabeta(ref.get_colors()).reshape(ref.get_num_vertices(), 3)

        norma_src = src.get_vertex_normals().reshape(src.get_num_vertices(), 3)
        norma_ref = ref.get_vertex_normals().reshape(ref.get_num_vertices(), 3)

        if pca_enabled:
            pca = PCA(n_components=3)
            pca.fit(norma_src)
            src_pca_nx = np.einsum('ij, ij->i', norma_src, np.full((src.get_num_vertices(), 3), pca.components_[0]))
            src_pca_ny = np.einsum('ij, ij->i', norma_src, np.full((src.get_num_vertices(), 3), pca.components_[1]))
            src_pca_nz = np.einsum('ij, ij->i', norma_src, np.full((src.get_num_vertices(), 3), pca.components_[2]))
            norma_src = np.stack((src_pca_nx, src_pca_ny, src_pca_nz), axis=1)

            pca_ref = PCA(n_components=3)
            pca_ref.fit(norma_ref)
            ref_pca_nx = np.einsum('ij, ij->i', norma_ref, np.full((ref.get_num_vertices(), 3), pca_ref.components_[0]))
            ref_pca_ny = np.einsum('ij, ij->i', norma_ref, np.full((ref.get_num_vertices(), 3), pca_ref.components_[1]))
            ref_pca_nz = np.einsum('ij, ij->i', norma_ref, np.full((ref.get_num_vertices(), 3), pca_ref.components_[2]))
            norma_ref = np.stack((ref_pca_nx, ref_pca_ny, ref_pca_nz), axis=1)

        src_normal_6d = np.concatenate((np.where(norma_src > 0.0, 0.0, np.abs(norma_src)),
                                        np.where(norma_src < 0.0, 0.0, np.abs(norma_src))), axis=1)
        ref_normal_6d = np.concatenate((np.where(norma_ref > 0.0, 0.0, np.abs(norma_ref)),
                                        np.where(norma_ref < 0.0, 0.0, np.abs(norma_ref))),axis=1)

        # contains color and normal information per vertex -> [[cx, cy, cz, nx, ny, nz], ...]
        src_in_raw = np.concatenate((src_color_lab, src_normal_6d), axis=1).reshape(src.get_num_vertices(), 9)
        ref_in_raw = np.concatenate((ref_color_lab, ref_normal_6d), axis=1).reshape(ref.get_num_vertices(), 9)

        src_mean = np.mean(src_in_raw, axis=0)
        ref_mean = np.mean(ref_in_raw, axis=0)
        src_cov = np.cov(src_in_raw, rowvar=False)
        ref_cov = np.cov(ref_in_raw, rowvar=False)

        src_covs = fractional_matrix_power(src_cov, 0.5)
        src_covsr = fractional_matrix_power(src_cov, -0.5)

        M = np.dot(src_covsr, np.dot(fractional_matrix_power(np.dot(src_covs, np.dot(ref_cov, src_covs)), 0.5), src_covsr))
        f_out = np.dot(src_in_raw - src_mean, M) + ref_mean

        lab_new = f_out[:,:3]
        lab_new = lab_new.reshape(src.get_num_vertices(), 1, 3)

        # Remove imaginary part
        lab_new = np.real(lab_new)

        lab_new = ColorSpaces.lalphabeta_to_rgb(np.ascontiguousarray(lab_new, dtype=np.float32))
        lab_new = np.clip(lab_new, 0.0, 1.0)


        return lab_new

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src, ref, opt):
        if opt.version == "IGD":
            out = Goude21.IGD(src, ref, opt.pca)
        else:
            out = Goude21.MGD(src, ref, opt.pca)
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_pointcloud(src, ref, opt):
        out_img = deepcopy(src)

        out_colors = Goude21.__color_transfer(src, ref, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp