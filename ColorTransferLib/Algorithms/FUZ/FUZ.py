"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from fcmeans import FCM
import networkx as nx
import time
from copy import deepcopy

from ColorTransferLib.ImageProcessing.ColorSpaces import ColorSpaces
from ColorTransferLib.Utils.Helper import check_compatibility


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
    compatibility = {
        "src": ["Image", "Mesh", "PointCloud"],
        "ref": ["Image", "Mesh", "PointCloud"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Returns basic information of the corresponding publication.
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "FuzzyColorTransfer",
            "title": "An efficient fuzzy clustering-based color transfer method",
            "year": 2010,
            "abstract": "Each image has its own color content that greatly influences the perception of human "
                        "observer. Recently, color transfer among different images has been under investigation. In "
                        "this paper, after a brief review on the few efficient works performed in the field, a novel "
                        "fuzzy clustering based color transfer method is proposed. The proposed method accomplishes "
                        "the transformation based on a set of corresponding fuzzy clustering algorithm-selected "
                        "regions in images along with membership degree factors. Results show the presented algorithm "
                        "is highly automatically and more effective.",
            "types": ["Image", "Mesh", "PointCloud"]
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        # Record the start time for performance measurement
        start_time = time.time()

        # Check if the source and reference images are compatible for color transfer
        output = check_compatibility(src, ref, FUZ.compatibility)

        # If not compatible, return the error output
        if output["status_code"] == -1:
            output["response"] = "Incompatible type."
            return output

        # Extract colors from the source and reference images
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

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

        # [9] Set the new colors to the output image
        out_img.set_colors(lab_new)

        # Prepare and return the output with performance metrics
        output = {
            "status_code": 0,
            "response": "",
            "object": out_img,
            "process_time": time.time() - start_time
        }

        return output
