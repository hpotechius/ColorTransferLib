"""
Copyright 2026 by Herbert Potechius,
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
import skfuzzy as fuzz

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
#   Identifier: Qian10
#   Link: https://doi.org/10.1109/FSKD.2010.5569560
#
# Implementation Details:
#   Number of Clusters: 3
#   Fuzzier: 2.0
#   Max Iterations: 100
#   Error: 1e-04
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Qian10:
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
            out_obj = Qian10.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = Qian10.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = Qian10.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = Qian10.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = Qian10.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = Qian10.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = Qian10.__apply_mesh(src, ref, opt)
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
    def __color_transfer(src_color, ref_color, opt):
        # ensure float32 and valid shape
        src_arr = np.asarray(src_color, dtype=np.float32)
        ref_arr = np.asarray(ref_color, dtype=np.float32)
        if src_arr.shape[-1] != 3 or ref_arr.shape[-1] != 3:
            raise ValueError(
                f"__color_transfer expects last dimension = 3, "
                f"got src_shape={src_arr.shape}, ref_shape={ref_arr.shape}"
            )

        src_pix_num, ref_pix_num = src_arr.shape[0], ref_arr.shape[0]
        dim = src_arr.shape[2]
        c = opt.cluster_num
        m = opt.fuzzier
        max_iter = opt.max_iterations
        term_error = opt.error
        eps = 1e-5  # for numerical stability

        # ------------------------------------------------------------------
        # choose working colorspace
        # ------------------------------------------------------------------
        if opt.colorspace == "cielab":
            src_cs = ColorSpaces.rgb_to_cielab(src_arr).reshape(src_pix_num, dim)
            ref_cs = ColorSpaces.rgb_to_cielab(ref_arr).reshape(ref_pix_num, dim)
        elif opt.colorspace == "lalphabeta":
            src_cs = ColorSpaces.rgb_to_lalphabeta(src_arr).reshape(src_pix_num, dim)
            ref_cs = ColorSpaces.rgb_to_lalphabeta(ref_arr).reshape(ref_pix_num, dim)
        elif opt.colorspace == "rgb":
            src_cs = src_arr.reshape(src_pix_num, dim)
            ref_cs = ref_arr.reshape(ref_pix_num, dim)
        else:
            raise ValueError(f"Unsupported colorspace: {opt.colorspace}")

        # Validate input
        if not np.all(np.isfinite(src_cs)) or not np.all(np.isfinite(ref_cs)):
            raise ValueError("Invalid color values detected in working colorspace (NaN or Inf)")
        if np.std(src_cs) < 1e-5 or np.std(ref_cs) < 1e-5:
            raise ValueError("Insufficient color variance – image may be nearly uniform")

        # ------------------------------------------------------------------
        # FCM clustering (use skfuzzy.cmeans with correct data orientation)
        # ------------------------------------------------------------------
        # skfuzzy.cmeans expects data with shape (features, N_samples)
        # so we pass src_cs.T / ref_cs.T and transpose membership back.
        src_cntr, src_u, src_u0, src_d, src_jm, src_p, src_fpc = fuzz.cluster.cmeans(
            src_cs.T,              # (dim, N)
            c=c,
            m=2.0,
            error=term_error,
            maxiter=max_iter,
            init=None,
        )
        # src_u: (c, N) -> (N, c) for our helper
        src_u = src_u.T

        ref_cntr, ref_u, ref_u0, ref_d, ref_jm, ref_p, ref_fpc = fuzz.cluster.cmeans(
            ref_cs.T,              # (dim, N_ref)
            c=c,
            m=2.0,
            error=term_error,
            maxiter=max_iter,
            init=None,
        )
        # ref_u: (c, N_ref) -> (N_ref, c)
        ref_u = ref_u.T

        # Helper: compute per-cluster standard deviation and weight
        def compute_std_weights(data, centers, membership):
            """
            data       : (N, dim)
            centers    : (c, dim)
            membership : (N, c)  (after transposing u from cmeans)
            """
            norm_factor = membership.sum(axis=0)    # (c,)
            std = np.zeros_like(centers)           # (c, dim)
            weights = np.zeros(c)
            for i in range(c):
                delta = data - centers[i]          # (N, dim)
                weighted_sq = membership[:, i][:, None] * (delta ** 2)  # (N, dim)
                std[i] = np.sqrt(
                    weighted_sq.sum(axis=0) / max(norm_factor[i], eps)
                )
                weights[i] = np.mean(std[i])
            return std, weights

        std_s, weights_s = compute_std_weights(src_cs, src_cntr, src_u)
        std_r, weights_r = compute_std_weights(ref_cs, ref_cntr, ref_u)

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

        # Perform color transfer using soft cluster assignments in working colorspace
        cs_new = np.zeros((src_pix_num, dim))
        for i in range(c):
            ms = src_u[:, i][:, None]
            mu_s = src_cntr[i]
            mu_r = ref_cntr[mapping[i]]
            std_s_i = np.where(std_s[i] < eps, eps, std_s[i])
            scale = std_r[mapping[i]] / std_s_i
            shifted = (src_cs - mu_s) * scale + mu_r
            cs_new += ms * shifted

        # Convert back to RGB
        cs_new = cs_new.reshape(src_pix_num, 1, dim)
        if opt.colorspace == "cielab":
            rgb_new = ColorSpaces.cielab_to_rgb(cs_new)
        elif opt.colorspace == "lalphabeta":
            rgb_new = ColorSpaces.lalphabeta_to_rgb(cs_new)
        else:  # "rgb"
            rgb_new = cs_new

        return np.clip(rgb_new, 0.0, 1.0)


    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = Qian10.__color_transfer(src_color, ref_color, opt)

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

            out_colors = Qian10.__color_transfer(src_color, ref_color, opt)

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

            out_colors = Qian10.__color_transfer(src_color, ref_color, opt)

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

                out_colors = Qian10.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Qian10.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Qian10.__color_transfer(src_color, ref_color, opt)

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

        out_colors = Qian10.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

