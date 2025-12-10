"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.

Adaptation of https://github.com/cysmith/neural-style-tf
"""

import tensorflow as tf
import numpy as np
import scipy.io


"""
Error: CUBLAS_STATUS_NOT_iNITIALIZED 
Source: https://forums.developer.nvidia.com/t/cublas-status-not-initialized/177955
If that solution fixes it, the problem is due to the fact that TF has a greedy allocation method (when you don’t set 
allow_growth). This greedy allocation method uses up nearly all GPU memory. When CUBLAS is asked to initialize (later), 
it requires some GPU memory to initialize. There is not enough memory left for CUBLAS to initialize, so the CUBLAS 
initialization fails.
"""
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Model:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_img, opt):
        self.opt = opt
        self.net = self.build_model(input_img, opt)

    # ----------------------------------------------------------------------------------------------------------------------
    # pre-trained vgg19 convolutional neural network
    # remark: layers are manually initialized for clarity.
    # ----------------------------------------------------------------------------------------------------------------------
    def build_model(self, input_img, opt):
        if opt.verbose:
            print('\nBUILDING VGG-19 NETWORK')
        net = {}
        _, h, w, d = input_img.shape

        if opt.verbose:
            print('loading model weights...')
        vgg_rawnet = scipy.io.loadmat(opt.model_weights)
        vgg_layers = vgg_rawnet['layers'][0]
        if opt.verbose:
            print('constructing layers...')
        net['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

        if opt.verbose:
            print('LAYER GROUP 1')
        net['conv1_1'] = self.conv_layer('conv1_1', net['input'], W=self.get_weights(vgg_layers, 0))
        net['relu1_1'] = self.relu_layer('relu1_1', net['conv1_1'], b=self.get_bias(vgg_layers, 0))

        net['conv1_2'] = self.conv_layer('conv1_2', net['relu1_1'], W=self.get_weights(vgg_layers, 2))
        net['relu1_2'] = self.relu_layer('relu1_2', net['conv1_2'], b=self.get_bias(vgg_layers, 2))

        net['pool1'] = self.pool_layer('pool1', net['relu1_2'])

        if opt.verbose:
            print('LAYER GROUP 2')
        net['conv2_1'] = self.conv_layer('conv2_1', net['pool1'], W=self.get_weights(vgg_layers, 5))
        net['relu2_1'] = self.relu_layer('relu2_1', net['conv2_1'], b=self.get_bias(vgg_layers, 5))

        net['conv2_2'] = self.conv_layer('conv2_2', net['relu2_1'], W=self.get_weights(vgg_layers, 7))
        net['relu2_2'] = self.relu_layer('relu2_2', net['conv2_2'], b=self.get_bias(vgg_layers, 7))

        net['pool2'] = self.pool_layer('pool2', net['relu2_2'])

        if opt.verbose:
            print('LAYER GROUP 3')
        net['conv3_1'] = self.conv_layer('conv3_1', net['pool2'], W=self.get_weights(vgg_layers, 10))
        net['relu3_1'] = self.relu_layer('relu3_1', net['conv3_1'], b=self.get_bias(vgg_layers, 10))

        net['conv3_2'] = self.conv_layer('conv3_2', net['relu3_1'], W=self.get_weights(vgg_layers, 12))
        net['relu3_2'] = self.relu_layer('relu3_2', net['conv3_2'], b=self.get_bias(vgg_layers, 12))

        net['conv3_3'] = self.conv_layer('conv3_3', net['relu3_2'], W=self.get_weights(vgg_layers, 14))
        net['relu3_3'] = self.relu_layer('relu3_3', net['conv3_3'], b=self.get_bias(vgg_layers, 14))

        net['conv3_4'] = self.conv_layer('conv3_4', net['relu3_3'], W=self.get_weights(vgg_layers, 16))
        net['relu3_4'] = self.relu_layer('relu3_4', net['conv3_4'], b=self.get_bias(vgg_layers, 16))

        net['pool3'] = self.pool_layer('pool3', net['relu3_4'])

        if opt.verbose:
            print('LAYER GROUP 4')
        net['conv4_1'] = self.conv_layer('conv4_1', net['pool3'], W=self.get_weights(vgg_layers, 19))
        net['relu4_1'] = self.relu_layer('relu4_1', net['conv4_1'], b=self.get_bias(vgg_layers, 19))

        net['conv4_2'] = self.conv_layer('conv4_2', net['relu4_1'], W=self.get_weights(vgg_layers, 21))
        net['relu4_2'] = self.relu_layer('relu4_2', net['conv4_2'], b=self.get_bias(vgg_layers, 21))

        net['conv4_3'] = self.conv_layer('conv4_3', net['relu4_2'], W=self.get_weights(vgg_layers, 23))
        net['relu4_3'] = self.relu_layer('relu4_3', net['conv4_3'], b=self.get_bias(vgg_layers, 23))

        net['conv4_4'] = self.conv_layer('conv4_4', net['relu4_3'], W=self.get_weights(vgg_layers, 25))
        net['relu4_4'] = self.relu_layer('relu4_4', net['conv4_4'], b=self.get_bias(vgg_layers, 25))

        net['pool4'] = self.pool_layer('pool4', net['relu4_4'])

        if opt.verbose:
            print('LAYER GROUP 5')
        net['conv5_1'] = self.conv_layer('conv5_1', net['pool4'], W=self.get_weights(vgg_layers, 28))
        net['relu5_1'] = self.relu_layer('relu5_1', net['conv5_1'], b=self.get_bias(vgg_layers, 28))

        net['conv5_2'] = self.conv_layer('conv5_2', net['relu5_1'], W=self.get_weights(vgg_layers, 30))
        net['relu5_2'] = self.relu_layer('relu5_2', net['conv5_2'], b=self.get_bias(vgg_layers, 30))

        net['conv5_3'] = self.conv_layer('conv5_3', net['relu5_2'], W=self.get_weights(vgg_layers, 32))
        net['relu5_3'] = self.relu_layer('relu5_3', net['conv5_3'], b=self.get_bias(vgg_layers, 32))

        net['conv5_4'] = self.conv_layer('conv5_4', net['relu5_3'], W=self.get_weights(vgg_layers, 34))
        net['relu5_4'] = self.relu_layer('relu5_4', net['conv5_4'], b=self.get_bias(vgg_layers, 34))

        net['pool5'] = self.pool_layer('pool5', net['relu5_4'])

        return net

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def conv_layer(self, layer_name, layer_input, W):
        conv = tf.nn.conv2d(input=layer_input, filters=W, strides=[1, 1, 1, 1], padding='SAME')
        if self.opt.verbose:
            print('--{} | shape={} | weights_shape={}'.format(layer_name, conv.get_shape(), W.get_shape()))
        return conv

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def relu_layer(self, layer_name, layer_input, b):
        relu = tf.nn.relu(layer_input + b)
        if self.opt.verbose:
            print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), b.get_shape()))
        return relu

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def pool_layer(self, layer_name, layer_input):
        if self.opt.pooling_type == 'avg':
            pool = tf.nn.avg_pool2d(input=layer_input,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        elif self.opt.pooling_type == 'max':
            pool = tf.nn.max_pool2d(input=layer_input,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        if self.opt.verbose:
            print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
        return pool

    # ----------------------------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------------------------
    def get_weights(self, vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        W = tf.constant(weights)
        return W

    # ----------------------------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------------------------
    def get_bias(self, vgg_layers, i):
        bias = vgg_layers[i][0][0][2][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return b
