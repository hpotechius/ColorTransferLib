"""
Copyright 2025 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import sys
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "1"



import numpy as np
import configparser
from pprint import pprint as ppt
import tensorflow as tf
import json
import sys
import pickle
from copy import deepcopy
import time

from ColorTransferLib.Algorithms.PSN.psnet import PSNet
from ColorTransferLib.Algorithms.PSN.utils import *
from ColorTransferLib.Utils.Helper import init_model_files, get_cache_dir

opj = os.path.join
ope = os.path.exists
om = os.mkdir

# Eager Execution deaktivieren
tf.compat.v1.disable_eager_execution()
# print(tf.__version__)
# 
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color
#   Author: Cao, Xu and Wang, Weimin and Nagao, Katashi and Nakamura, Ryosuke
#   Published in: IEEE Winter Conference on Applications of Computer Vision (WACV)
#   Year of Publication: 2020
#
# Info:
#   Name: PSNetStyleTransfer
#   Identifier: PSN
#   Link: https://doi.org/10.1109/WACV45572.2020.9093513
#   Source: https://github.com/hoshino042/psnet
#
# Notes:
#   TensorFlow 2.14.0 has issues with large point clouds. Use TensorFlow 2.13.0 if necessary.
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class PSN:
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

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField" or ref.get_type() == "GaussianSplatting" or ref.get_type() == "Mesh" or ref.get_type() == "Image":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "PointCloud":
            out_obj = PSN.__apply_pointcloud(src, ref, opt)
        else:
            out_obj = None
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src, ref, opt): 
        iteration = opt.iterations  # iteration number for style transfer
        geotransfer = opt.geotransfer

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Uncomment this line if you are using macOS

        model_file_paths = init_model_files("PSN", [
            "model.data-00000-of-00001",
            "model.index",
            "model.meta",
            "base_config.ini"
        ])

        np.random.seed(42)
        # trained_model = "Models/PSN/model"
        trained_model = os.path.join(get_cache_dir(), "PSN/model")

        config = configparser.ConfigParser()
        config.read(model_file_paths["base_config.ini"])

        content_layer = list(map(lambda x: int(x), config["style_transfer"]["content_layer"].split(",")))
        style_layer = list(map(lambda x: int(x), config["style_transfer"]["content_layer"].split(",")))
        use_content_color = ["FE_COLOR_FE_{}".format(i) for i in content_layer]
        use_style_color = ["FE_COLOR_FE_{}".format(i) for i in style_layer]

        content_geo = src.get_vertex_positions().reshape(src.get_num_vertices(), 3)
        content_ncolor = src.get_colors().reshape(src.get_num_vertices(), 3)
        # colors have to be in range [-1, 1]
        # content_ncolor = (content_ncolor * 2 - 1).astype(np.int16)
        content_ncolor = content_ncolor * 2.0 - 1.0
        # content_color = (255 * content_ncolor).astype(np.int16)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0], 
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # Max. 4GB verwenden
                    )
                    #tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

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
            style_ncolor = ref.get_colors().reshape(ref.get_num_vertices(), 3) * 2.0 - 1.0

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

        return src

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_pointcloud(src, ref, opt):
        # suppress output
        # devnull = open(os.devnull, 'w')
        # old_stdout = sys.stdout
        # sys.stdout = devnull
        out = PSN.__color_transfer(src, ref, opt)
        # sys.stdout = old_stdout
        # devnull.close()
        return out