"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""


import numpy as np
from fcmeans import FCM
import networkx as nx
import time
from copy import deepcopy

from ColorTransferLib.Utils.ColorSpaces import ColorSpaces
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: An efficient fuzzy clustering-based color transfer method
#   Author: XiaoYan Qian, BangFeng Wang, Lei Han
#   Published in: Seventh International Conference on Fuzzy Systems and Knowledge Discovery
#   Year of Publication: 2010
#
# Abstract:
#   Each image has its own color content that greatly influences the perception of human observer. Recently, color
#   transfer among different images has been under investigation. In this paper, after a brief review on the few
#   efficient works performed in the field, a novel fuzzy clustering based color transfer method is proposed. The
#   proposed method accomplishes the transformation based on a set of corresponding fuzzy clustering
#   algorithm-selected regions in images along with membership degree factors. Results show the presented algorithm is
#   highly automatically and more effective.
#
# Info:
#   Name: FuzzyColorTransfer
#   Identifier: FUZ
#   Link: https://doi.org/10.1109/FSKD.2010.5569560
#
# Implementation Details:
#   Number of Clusters: 3
#   Fuzzier: 2.0
#   Max Iterations: 100
#   Error: 1e-04
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class FUZ:
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

        if src.get_type() == "Image":
            out_obj = FUZ.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = FUZ.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = FUZ.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = FUZ.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = FUZ.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = FUZ.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = FUZ.__apply_mesh(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer_old(src_color, ref_color, opt):
        # [1] Extract parameters needed for the algorithm
        src_pix_num = src_color.shape[0]
        ref_pix_num = ref_color.shape[0]
        dim = src_color.shape[2]
        clusters = opt.cluster_num
        max_iter = opt.max_iterations
        fuzzier = opt.fuzzier
        term_error = opt.error

        # [2] Convert colors from RGB to Lab color space
        src_color = ColorSpaces.rgb_to_lab_cpu(src_color)
        ref_color = ColorSpaces.rgb_to_lab_cpu(ref_color)


        # [3] reshaping to (num_pixels, 3) because input is of size (num_pixels, 1, 3)
        src_reshape = src_color.reshape(src_pix_num, dim)
        ref_reshape = ref_color.reshape(ref_pix_num, dim)

        # [4] Apply Fuzzy C-Means clustering to both source and reference colors
        fcm_src = FCM(n_clusters=clusters, max_iter=max_iter, m=fuzzier, error=term_error)
        fcm_src.fit(src_reshape)

        fcm_ref = FCM(n_clusters=clusters, max_iter=max_iter, m=fuzzier, error=term_error)
        fcm_ref.fit(ref_reshape)


        # [5] Calculate cluster directions
        membership_s = fcm_src.u
        norm_factor_s = np.sum(membership_s, axis=0)
        centers_s = fcm_src.centers

        std_s = np.zeros((clusters, 3))
        weights_s = np.zeros(clusters)
        for c in range(clusters):
            sig_l = np.sqrt(np.sum(membership_s[:,c:c+1] * np.power(src_reshape[:,0:1] - centers_s[c][0], 2) / norm_factor_s[c]))
            sig_a = np.sqrt(np.sum(membership_s[:,c:c+1] * np.power(src_reshape[:,1:2] - centers_s[c][1], 2) / norm_factor_s[c]))
            sig_b = np.sqrt(np.sum(membership_s[:,c:c+1] * np.power(src_reshape[:,2:3] - centers_s[c][2], 2) / norm_factor_s[c]))
            std_s[c] = np.array([sig_l, sig_a, sig_b])
            weights_s[c] = (1/3)*sig_l + (1/3)*sig_a + (1/3)*sig_b

        membership_r = fcm_ref.u
        norm_factor_r = np.sum(membership_r, axis=0)
        centers_r = fcm_ref.centers
        std_r = np.zeros((clusters, 3))
        weights_r = np.zeros(clusters)
        for c in range(clusters):
            sig_l = np.sqrt(np.sum(membership_r[:,c:c+1] * np.power(ref_reshape[:,0:1] - centers_r[c][0], 2) / norm_factor_r[c]))
            sig_a = np.sqrt(np.sum(membership_r[:,c:c+1] * np.power(ref_reshape[:,1:2] - centers_r[c][1], 2) / norm_factor_r[c]))
            sig_b = np.sqrt(np.sum(membership_r[:,c:c+1] * np.power(ref_reshape[:,2:3] - centers_r[c][2], 2) / norm_factor_r[c]))
            std_r[c] = np.array([sig_l, sig_a, sig_b])
            weights_r[c] = (1/3)*sig_l + (1/3)*sig_a + (1/3)*sig_b

        # [6] Calculate cluster directions by bipartite matching
        mapping = np.arange(clusters)
        B = nx.Graph()
        B.add_nodes_from(np.arange(clusters), bipartite=0)
        B.add_nodes_from(np.arange(2*clusters), bipartite=1)

        for ks, ws in enumerate(weights_s):
            for kr, wr in enumerate(weights_r):
                B.add_edge(ks, kr+clusters, weight=np.linalg.norm(ws-wr))

        my_matching = nx.bipartite.matching.minimum_weight_full_matching(B, np.arange(clusters), "weight")

        for i in range(clusters):
            mapping[i] = my_matching[i] - clusters

        # [7] Apply Reinhard's Color Transfer per cluster combination
        lab_new = np.zeros((dim, src_pix_num))

        for c in range(clusters):
            l_c = (std_r[mapping[c]][0]/std_s[c][0]) * (src_reshape[:,0:1] - centers_s[c][0]) + centers_r[mapping[c]][0]
            a_c = (std_r[mapping[c]][1]/std_s[c][1]) * (src_reshape[:,1:2] - centers_s[c][1]) + centers_r[mapping[c]][1]
            b_c = (std_r[mapping[c]][2]/std_s[c][2]) * (src_reshape[:,2:3] - centers_s[c][2]) + centers_r[mapping[c]][2]

            lab_new[0] += np.sum(l_c * membership_s[:,c:c+1], axis=1)
            lab_new[1] += np.sum(a_c * membership_s[:,c:c+1], axis=1)
            lab_new[2] += np.sum(b_c * membership_s[:,c:c+1], axis=1)

        lab_new = lab_new.T.reshape((src_pix_num, dim))

        # [8] Convert the resulting Lab colors back to RGB
        lab_new = lab_new.reshape(src_pix_num, 1, dim)
        lab_new = ColorSpaces.lab_to_rgb_cpu(lab_new)
        lab_new = np.clip(lab_new, 0.0, 1.0)

        return lab_new

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_color, ref_color, opt):
        src_pix_num, ref_pix_num = src_color.shape[0], ref_color.shape[0]
        dim = src_color.shape[2]
        c = opt.cluster_num
        m = opt.fuzzier
        max_iter = opt.max_iterations
        term_error = opt.error
        eps = 1e-5  # for numerical stability

        # Convert RGB to Lab color space
        src_lab = ColorSpaces.rgb_to_lab_cpu(src_color).reshape(src_pix_num, dim)
        ref_lab = ColorSpaces.rgb_to_lab_cpu(ref_color).reshape(ref_pix_num, dim)

        # Validate input
        if not np.all(np.isfinite(src_lab)) or not np.all(np.isfinite(ref_lab)):
            raise ValueError("Invalid color values detected in src_lab or ref_lab (NaN or Inf)")
        if np.std(src_lab) < 1e-5 or np.std(ref_lab) < 1e-5:
            raise ValueError("Insufficient color variance – image may be nearly uniform")

        # Run FCM clustering (explicit .fit call, no chaining)
        fcm_src = FCM(n_clusters=c, max_iter=max_iter, m=m, error=term_error, random_state=42)
        fcm_src.fit(src_lab)

        fcm_ref = FCM(n_clusters=c, max_iter=max_iter, m=m, error=term_error, random_state=42)
        fcm_ref.fit(ref_lab)

        # Helper: compute per-cluster standard deviation and weight
        def compute_std_weights(data, centers, membership):
            norm_factor = membership.sum(axis=0)
            std = np.zeros_like(centers)
            weights = np.zeros(c)
            for i in range(c):
                delta = data - centers[i]
                weighted_sq = membership[:, i][:, None] * (delta ** 2)
                std[i] = np.sqrt(weighted_sq.sum(axis=0) / max(norm_factor[i], eps))
                weights[i] = np.mean(std[i])
            return std, weights

        std_s, weights_s = compute_std_weights(src_lab, fcm_src.centers, fcm_src.u)
        std_r, weights_r = compute_std_weights(ref_lab, fcm_ref.centers, fcm_ref.u)

        # Validate weights
        if not np.all(np.isfinite(weights_s)) or not np.all(np.isfinite(weights_r)):
            weights_s = np.nan_to_num(weights_s, nan=0.0, posinf=1e3, neginf=-1e3)
            weights_r = np.nan_to_num(weights_r, nan=0.0, posinf=1e3, neginf=-1e3)

        # Perform cluster matching (bipartite graph)
        G = nx.Graph()
        G.add_nodes_from(range(c), bipartite=0)
        G.add_nodes_from(range(c, 2 * c), bipartite=1)
        for i in range(c):
            for j in range(c):
                diff = np.linalg.norm(weights_s[i] - weights_r[j])
                G.add_edge(i, j + c, weight=diff)

        try:
            matching = nx.bipartite.minimum_weight_full_matching(G, range(c), "weight")
        except ValueError as e:
            raise ValueError(f"Cluster matching failed: {e}. Cluster weights may be invalid.")

        mapping = [matching[i] - c for i in range(c)]

        # Perform color transfer using soft cluster assignments
        lab_new = np.zeros((src_pix_num, dim))
        for i in range(c):
            ms = fcm_src.u[:, i][:, None]
            mu_s = fcm_src.centers[i]
            mu_r = fcm_ref.centers[mapping[i]]
            std_s_i = np.where(std_s[i] < eps, eps, std_s[i])
            scale = std_r[mapping[i]] / std_s_i
            shifted = (src_lab - mu_s) * scale + mu_r
            lab_new += ms * shifted

        # Convert Lab back to RGB
        lab_new = lab_new.reshape(src_pix_num, 1, dim)
        rgb_new = ColorSpaces.lab_to_rgb_cpu(lab_new)
        return np.clip(rgb_new, 0.0, 1.0)


    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = FUZ.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_video(src, ref, opt): 
        # check if type is video
        out_colors_arr = []
        src_colors = src.get_colors()

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_images()[0])

            out_colors = FUZ.__color_transfer(src_color, ref_color, opt)

            out_img.set_colors(out_colors)
            out_colors_arr.append(out_img)

        outp = Video(imgs=out_colors_arr, fps=src.get_fps())

        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_volumetricvideo(src, ref, opt): 
        out_colors_arr = []
        src_colors = src.get_colors()

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = FUZ.__color_transfer(src_color, ref_color, opt)

            out_img.set_colors(out_colors)
            out_colors_arr.append(out_img)
            outp = VolumetricVideo(meshes=out_colors_arr, file_name=src.get_file_name())

        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_lightfield(src, ref, opt):
        src_lightfield_array = src.get_image_array()
        out = deepcopy(src)
        out_lightfield_array = out.get_image_array()

        for row in range(src.get_grid_size()[0]):
            for col in range(src.get_grid_size()[1]):
                src_color = src_lightfield_array[row][col].get_colors()
                ref_color = ref.get_colors()

                out_colors = FUZ.__color_transfer(src_color, ref_color, opt)

                out_lightfield_array[row][col].set_colors(out_colors)

        return out
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_gaussiansplatting(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = FUZ.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_pointcloud(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = FUZ.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = FUZ.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

