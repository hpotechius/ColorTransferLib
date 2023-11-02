"""
Copyright 2023 by Herbert Potechius,
Ernst-Abbe-Hochschule Jena - University of Applied Sciences - Department of Electrical Engineering and Information
Technology - Immersive Media and AR/VR Research Group.
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import configparser
from pprint import pprint as ppt
import tensorflow as tf
import os

from ColorTransferLib.Algorithms.PSN.psnet import PSNet
from ColorTransferLib.Algorithms.PSN.utils import *
from ColorTransferLib.Utils.Helper import check_compatibility

opj = os.path.join
ope = os.path.exists
om = os.mkdir


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color
#   Author: Cao, Xu and Wang, Weimin and Nagao, Katashi and Nakamura, Ryosuke
#   Published in: IEEE Computer Graphics and Applications
#   Year of Publication: 2020
#
# Abstract:
#   We propose a neural style transfer method for colored point clouds which allows stylizing the geometry and/or color
#   property of a point cloud from another. The stylization is achieved by manipulating the content representations and
#   Gram-based style representations extracted from a pre-trained PointNet-based classification network for colored
#   point clouds. As Gram-based style representation is invariant to the number or the order of points, the style can
#   also be an image in the case of stylizing the color property of a point cloud by merely treating the image as a set
#   of pixels.Experimental results and analysis demonstrate the capability of the proposed method for stylizing a
#   point cloud either from another point cloud or an image.
#
# Info:
#   Name: PSNetStyleTransfer
#   Identifier: PSN
#   Link: https://doi.org/10.1109/WACV45572.2020.9093513
#   Source: https://github.com/hoshino042/psnet
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PSN:
    compatibility = {
        "src": ["PointCloud"],
        "ref": ["PointCloud"]
    }

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_info():
        info = {
            "identifier": "PSN",
            "title": "PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color",
            "year": 2020,
            "abstract": "We propose a neural style transfer method for colored point clouds which allows stylizing the "
                        "geometry and/or color property of a point cloud from another. The stylization is achieved by "
                        "manipulating the content representations and Gram-based style representations extracted from "
                        "a pre-trained PointNet-based classification network for colored point clouds. As Gram-based "
                        "style representation is invariant to the number or the order of points, the style can also be "
                        "an image in the case of stylizing the color property of a point cloud by merely treating the "
                        "image as a set of pixels.Experimental results and analysis demonstrate the capability of the "
                        "proposed method for stylizing a point cloud either from another point cloud or an image.",
            "types": ["PointCloud"]
        }

        return info

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt): 
        # check if method is compatible with provided source and reference objects
        output = check_compatibility(src, ref, PSN.compatibility)
        if output["status_code"] != 0:
            output["response"] = "Incompatible type."
            return output
        
        iteration = opt.iterations  # iteration number for style transfer
        geotransfer = opt.geotransfer

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Uncomment this line if you are using macOS

        np.random.seed(42)
        trained_model = "Models/PSN/model"

        config = configparser.ConfigParser()
        config.read("Models/PSN/base_config.ini")

        content_layer = list(map(lambda x: int(x), config["style_transfer"]["content_layer"].split(",")))
        style_layer = list(map(lambda x: int(x), config["style_transfer"]["content_layer"].split(",")))
        use_content_color = ["FE_COLOR_FE_{}".format(i) for i in content_layer]
        use_style_color = ["FE_COLOR_FE_{}".format(i) for i in style_layer]

        content_geo = src.get_vertex_positions().reshape(src.get_num_vertices(), 3)
        content_ncolor = src.get_colors().reshape(src.get_num_vertices(), 3)
        content_color = (255 * content_ncolor).astype(np.int16)

        # get content representations
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        psnet = PSNet(config=config,
                        sess=sess,
                        train_dir="",
                        mode="test",
                        num_pts=content_ncolor.shape[0])
        psnet.restore_model(trained_model)
        ppt(list(psnet.node.keys()))

        obtained_content_fvs = sess.run(psnet.node, feed_dict={psnet.color: content_ncolor[None, ...],
                                                                psnet.geo: content_geo[None, ...],
                                                                psnet.bn_pl: False,
                                                                psnet.dropout_prob_pl: 1.0})
        sess.close()


        #if not (style_path.endswith("ply") or style_path.endswith("npy")):
        if ref.get_type() == "Image":
            style_ncolor = ref.get_colors().reshape(ref.get_pixelnum(), 3) * 2.0 - 1.0
            from_image = True            
        else:
            from_image = False
            style_geo = ref.get_vertex_positions().reshape(ref.get_num_vertices(), 3)
            style_ncolor = ref.get_colors().reshape(ref.get_num_vertices(), 3)

        # get style representations
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        psnet = PSNet(config=config, sess=sess, train_dir="", mode="test", num_pts=style_ncolor.shape[0])
        psnet.restore_model(trained_model)
        if from_image:
            obtained_style_fvs = sess.run({i: psnet.node_color[i] for i in use_style_color},
                                            feed_dict={psnet.color: style_ncolor[None, ...],
                                                        psnet.bn_pl: False,
                                                        psnet.dropout_prob_pl: 1.0})
        else:
            obtained_style_fvs = sess.run(psnet.node,
                                            feed_dict={psnet.color: style_ncolor[None, ...],
                                                        psnet.geo: style_geo[None, ...],
                                                        psnet.bn_pl: False,
                                                        psnet.dropout_prob_pl: 1.0})
        obtained_style_fvs_gram = dict()
        for layer, fvs in obtained_style_fvs.items():
            gram = []
            for row in fvs:
                gram.append(np.matmul(row.T, row) / row.size)
            obtained_style_fvs_gram[layer] = np.array(gram)

        sess.close()

        # code for style transfer
        tf.compat.v1.reset_default_graph()
        with tf.Graph().as_default() as graph:
            sess = tf.compat.v1.Session()
            psnet = PSNet(config=config,
                          sess=sess,
                          train_dir="",
                          mode="styletransfer",
                          num_pts=content_geo.shape[0],  # should be the same as content
                          target_content=obtained_content_fvs,
                          target_style=obtained_style_fvs_gram,
                          geo_init=tf.compat.v1.constant_initializer(value=content_geo),
                          color_init=tf.compat.v1.constant_initializer(value=content_ncolor),
                          from_image=from_image)
            psnet.restore_model(trained_model)
            previous_loss = float("inf")
            for i in range(iteration):
                psnet.style_transfer_one_step()
                if from_image:
                    current_total_loss = sess.run(psnet.total_loss_color, feed_dict={
                        psnet.bn_pl: False,
                        psnet.dropout_prob_pl: 1.0})
                else:
                    current_total_loss = sess.run(psnet.total_loss_color, feed_dict={
                        psnet.bn_pl: False,
                        psnet.dropout_prob_pl: 1.0}) + sess.run(psnet.total_loss_geo, feed_dict={
                        psnet.bn_pl: False,
                        psnet.dropout_prob_pl: 1.0})
                # stop criteria for style transfer, and save results (ply and png)
                if abs(previous_loss - current_total_loss) < 1e-7 or i == iteration - 1:
                    transferred_color = (127.5 * (np.squeeze(np.clip(sess.run(psnet.color), -1, 1)) + 1)).astype(float) / 255.0
                    transferred_color = transferred_color.reshape((transferred_color.shape[0], 1, 3))
                    src.set_colors(transferred_color)
                    
                    if not from_image and geotransfer:
                        transferred_geo = np.squeeze(sess.run(psnet.geo))
                        src.set_vertex_positions(transferred_geo.reshape(src.get_num_vertices(), 3))

                    break
                previous_loss = current_total_loss
            sess.close()

        output = {
            "status_code": 0,
            "response": "",
            "object": src
        }

        return output