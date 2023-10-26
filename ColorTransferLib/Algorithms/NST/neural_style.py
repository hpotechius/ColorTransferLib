import tensorflow as tf
import numpy as np
import scipy.io
import argparse
import struct
import errno
import time
import cv2
import os
import gc


# ----------------------------------------------------------------------------------------------------------------------
# parsing and configuration
# ----------------------------------------------------------------------------------------------------------------------
def set_args():
    # OPTIONS FOR SINGLE IMAGE

    # Boolean flag indicating if statements should be printed to the console.
    args.verbose = True
    # Filename of the output image.
    args.img_name = "result"
    # Filenames of the style images (example: starry-night.jpg)
    args.style_imgs = "reference.jpg"
    # Interpolation weights of each of the style images. (example: 0.5 0.5)
    args.style_imgs_weights = [1.0]
    # Filename of the content image (example: lion.jpg)
    args.content_img = "source.jpg"
    # Directory path to the style images. (default: %(default)s)
    args.style_imgs_dir = "./styles"
    # Directory path to the content image. (default: %(default)s)
    args.content_img_dir = "./image_input"
    # Image used to initialize the network. (default: %(default)s). Choices: ['random', 'content', 'style']
    args.init_img_type = "content"
    # Maximum width or height of the input images. (default: %(default)s)
    args.max_size = 512
    # Weight for the content loss function. (default: %(default)s)
    args.content_weight = 5e0
    # Weight for the style loss function. (default: %(default)s)
    args.style_weight = 1e4
    # Weight for the total variational loss function. Set small (e.g. 1e-3). (default: %(default)s)
    args.tv_weight = 1e-3
    # Weight for the temporal loss function. (default: %(default)s)
    args.temporal_weight = 2e2
    # Different constants for the content layer loss function. (default: %(default)s). Choices [1, 2, 3]
    args.content_loss_function = 1
    # VGG19 layers used for the content image. (default: %(default)s)
    args.content_layers = ['conv4_2']
    # VGG19 layers used for the style image. (default: %(default)s)
    args.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    # Contributions (weights) of each content layer to loss. (default: %(default)s)
    args.content_layer_weights = [1.0]
    # Contributions (weights) of each style layer to loss. (default: %(default)s)
    args.style_layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    # Transfer the style but not the colors.
    args.original_colors = True
    # Color space for conversion to original colors (default: %(default)s). Choices: ['yuv', 'ycrcb', 'luv', 'lab']
    args.color_convert_type = "yuv"
    # Time (before or after) to convert to original colors (default: %(default)s). Choices: ['after', 'before']
    args.color_convert_time = "after"
    # Transfer the style to masked regions.
    args.style_mask = True
    # Filenames of the style mask images (example: face_mask.png) (default: %(default)s)
    args.style_mask_imgs = None
    # Interpolation value between the content image and noise image if the network is initialized with 'random'.
    args.noise_ratio = 1.0
    # Seed for the random number generator. (default: %(default)s)
    args.seed = 0
    # Weights and biases of the VGG-19 network.
    args.model_weights = "imagenet-vgg-verydeep-19.mat"
    # Type of pooling in convolutional neural network. (default: %(default)s). Choices: ['avg', 'max']
    args.pooling_type = "avg"
    # GPU or CPU mode.  GPU mode requires NVIDIA CUDA. (default|recommended: %(default)s). Choices: ['/gpu:0', '/cpu:0']
    args.device = "/gpu:0"
    # Relative or absolute directory path to output image and data.
    args.img_output_dir = "./image_output"

    # OPTIMIZATION

    # Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. (default|recommended:
    # %(default)s). Choices: ['lbfgs', 'adam']. Info: "lbfgs" is not working on tensorflow2
    args.optimizer = "adam"
    # Learning rate parameter for the Adam optimizer. (default: %(default)s)
    args.learning_rate = 1e0
    # Max number of iterations for the Adam or L-BFGS optimizer. (default: %(default)s)
    args.max_iterations = 10
    # Number of iterations between optimizer print statements. (default: %(default)s)
    args.print_iterations = 50

    # normalize weights
    args.style_layer_weights = normalize(args.style_layer_weights)
    args.content_layer_weights = normalize(args.content_layer_weights)
    args.style_imgs_weights = normalize(args.style_imgs_weights)

    return args


# ----------------------------------------------------------------------------------------------------------------------
# pre-trained vgg19 convolutional neural network
# remark: layers are manually initialized for clarity.
# ----------------------------------------------------------------------------------------------------------------------
def build_model(input_img):
    if args.verbose:
        print('\nBUILDING VGG-19 NETWORK')
    net = {}
    _, h, w, d = input_img.shape

    if args.verbose: print('loading model weights...')
    vgg_rawnet = scipy.io.loadmat(args.model_weights)
    vgg_layers = vgg_rawnet['layers'][0]
    if args.verbose: print('constructing layers...')
    net['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

    if args.verbose: print('LAYER GROUP 1')
    net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
    net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))

    net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
    net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))

    net['pool1'] = pool_layer('pool1', net['relu1_2'])

    if args.verbose: print('LAYER GROUP 2')
    net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))

    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
    net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))

    net['pool2'] = pool_layer('pool2', net['relu2_2'])

    if args.verbose: print('LAYER GROUP 3')
    net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
    net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))

    net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
    net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))

    net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
    net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))

    net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
    net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))

    net['pool3'] = pool_layer('pool3', net['relu3_4'])

    if args.verbose: print('LAYER GROUP 4')
    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))

    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))

    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))

    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))

    net['pool4'] = pool_layer('pool4', net['relu4_4'])

    if args.verbose: print('LAYER GROUP 5')
    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))

    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))

    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))

    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))

    net['pool5'] = pool_layer('pool5', net['relu5_4'])

    return net


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def conv_layer(layer_name, layer_input, W):
    conv = tf.nn.conv2d(input=layer_input, filters=W, strides=[1, 1, 1, 1], padding='SAME')
    if args.verbose: print('--{} | shape={} | weights_shape={}'.format(layer_name,
                                                                       conv.get_shape(), W.get_shape()))
    return conv


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def relu_layer(layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    if args.verbose:
        print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(),
                                                       b.get_shape()))
    return relu


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def pool_layer(layer_name, layer_input):
    if args.pooling_type == 'avg':
        pool = tf.nn.avg_pool2d(input=layer_input, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
    elif args.pooling_type == 'max':
        pool = tf.nn.max_pool2d(input=layer_input, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
    if args.verbose:
        print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
    return pool


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_weights(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    W = tf.constant(weights)
    return W


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_bias(vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    return b


# ----------------------------------------------------------------------------------------------------------------------
# 'a neural algorithm for artistic style' loss functions
# ----------------------------------------------------------------------------------------------------------------------
def content_layer_loss(p, x):
    _, h, w, d = p.get_shape()
    M = h * w
    N = d
    if args.content_loss_function == 1:
        K = 1. / (2. * N ** 0.5 * M ** 0.5)
    elif args.content_loss_function == 2:
        K = 1. / (N * M)
    elif args.content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(input_tensor=tf.pow((x - p), 2))
    return loss


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h * w
    N = d
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(input_tensor=tf.pow((G - A), 2))
    return loss


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(a=F), F)
    return G


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def mask_style_layer(a, x, mask_img):
    _, h, w, d = a.get_shape()
    mask = get_mask_image(mask_img, w, h)
    mask = tf.convert_to_tensor(value=mask)
    tensors = []
    for _ in range(d):
        tensors.append(mask)
    mask = tf.stack(tensors, axis=2)
    mask = tf.stack(mask, axis=0)
    mask = tf.expand_dims(mask, 0)
    a = tf.multiply(a, mask)
    x = tf.multiply(x, mask)
    return a, x


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def sum_masked_style_losses(sess, net, style_imgs):
    total_style_loss = 0.
    weights = args.style_imgs_weights
    masks = args.style_mask_imgs
    for img, img_weight, img_mask in zip(style_imgs, weights, masks):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(args.style_layers, args.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(value=a)
            a, x = mask_style_layer(a, x, img_mask)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(args.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def sum_style_losses(sess, net, style_imgs):
    total_style_loss = 0.
    weights = args.style_imgs_weights
    for img, img_weight in zip(style_imgs, weights):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(args.style_layers, args.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(value=a)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(args.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def sum_content_losses(sess, net, content_img):
    sess.run(net['input'].assign(content_img))
    content_loss = 0.
    for layer, weight in zip(args.content_layers, args.content_layer_weights):
        p = sess.run(net[layer])
        x = net[layer]
        p = tf.convert_to_tensor(value=p)
        content_loss += content_layer_loss(p, x) * weight
    content_loss /= float(len(args.content_layers))
    return content_loss


# ----------------------------------------------------------------------------------------------------------------------
# 'artistic style transfer for videos' loss functions
# ----------------------------------------------------------------------------------------------------------------------
def temporal_loss(x, w, c):
    c = c[np.newaxis, :, :, :]
    D = float(x.size)
    loss = (1. / D) * tf.reduce_sum(input_tensor=c * tf.nn.l2_loss(x - w))
    loss = tf.cast(loss, tf.float32)
    return loss


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_longterm_weights(i, j):
    c_sum = 0.
    for k in range(args.prev_frame_indices):
        if i - k > i - j:
            c_sum += get_content_weights(i, i - k)
    c = get_content_weights(i, i - j)
    c_max = tf.maximum(c - c_sum, 0.)
    return c_max


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def sum_longterm_temporal_losses(sess, net, frame, input_img):
    x = sess.run(net['input'].assign(input_img))
    loss = 0.
    for j in range(args.prev_frame_indices):
        prev_frame = frame - j
        w = get_prev_warped_frame(frame)
        c = get_longterm_weights(frame, prev_frame)
        loss += temporal_loss(x, w, c)
    return loss


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def sum_shortterm_temporal_losses(sess, net, frame, input_img):
    x = sess.run(net['input'].assign(input_img))
    prev_frame = frame - 1
    w = get_prev_warped_frame(frame)
    c = get_content_weights(frame, prev_frame)
    loss = temporal_loss(x, w, c)
    return loss


# ----------------------------------------------------------------------------------------------------------------------
# utilities and i/o
# ----------------------------------------------------------------------------------------------------------------------
def read_image(path):
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = preprocess(img)
    return img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def write_image(path, img):
    img = postprocess(img)
    cv2.imwrite(path, img)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def preprocess(img):
    imgpre = np.copy(img)
    # bgr to rgb
    imgpre = imgpre[..., ::-1]
    # shape (h, w, d) to (1, h, w, d)
    imgpre = imgpre[np.newaxis, :, :, :]
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return imgpre


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def postprocess(img):
    imgpost = np.copy(img)
    imgpost += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    # shape (1, h, w, d) to (h, w, d)
    imgpost = imgpost[0]
    imgpost = np.clip(imgpost, 0, 255).astype('uint8')
    # rgb to bgr
    imgpost = imgpost[..., ::-1]
    return imgpost


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def read_flow_file(path):
    with open(path, 'rb') as f:
        # 4 bytes header
        header = struct.unpack('4s', f.read(4))[0]
        # 4 bytes width, height
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        flow = np.ndarray((2, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                flow[0, y, x] = struct.unpack('f', f.read(4))[0]
                flow[1, y, x] = struct.unpack('f', f.read(4))[0]
    return flow


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def read_weights_file(path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        vals[i - 1] = np.array(list(map(np.float32, line)))
        vals[i - 1] = list(map(lambda x: 0. if x < 255. else 1., vals[i - 1]))
    # expand to 3 channels
    weights = np.dstack([vals.astype(np.float32)] * 3)
    return weights


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def normalize(weights):
    denom = sum(weights)
    if denom > 0.:
        return [float(i) / denom for i in weights]
    else:
        return [0.] * len(weights)


# ----------------------------------------------------------------------------------------------------------------------
# rendering -- where the magic happens
# ----------------------------------------------------------------------------------------------------------------------
def stylize(content_img, style_imgs, init_img, frame=None):
    with tf.device(args.device), tf.compat.v1.Session() as sess:
        # setup network
        net = build_model(content_img)

        # style loss
        if args.style_mask:
            L_style = sum_masked_style_losses(sess, net, style_imgs)
        else:
            L_style = sum_style_losses(sess, net, style_imgs)

        # content loss
        L_content = sum_content_losses(sess, net, content_img)

        # denoising loss
        L_tv = tf.image.total_variation(net['input'])

        # loss weights
        alpha = args.content_weight
        beta = args.style_weight
        theta = args.tv_weight

        # total loss
        L_total = alpha * L_content
        L_total += beta * L_style
        L_total += theta * L_tv

        # video temporal loss
        if args.video and frame > 1:
            gamma = args.temporal_weight
            L_temporal = sum_shortterm_temporal_losses(sess, net, frame, init_img)
            L_total += gamma * L_temporal

        # optimization algorithm
        optimizer = get_optimizer(L_total)

        if args.optimizer == 'adam':
            minimize_with_adam(sess, net, optimizer, init_img, L_total)
        elif args.optimizer == 'lbfgs':
            minimize_with_lbfgs(sess, net, optimizer, init_img)

        output_img = sess.run(net['input'])

        if args.original_colors:
            output_img = convert_to_original_colors(np.copy(content_img), output_img)


        #write_image_output(output_img, content_img, style_imgs, init_img)
        return output_img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def minimize_with_lbfgs(sess, net, optimizer, init_img):
    if args.verbose: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def minimize_with_adam(sess, net, optimizer, init_img, loss):
    if args.verbose: print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
    train_op = optimizer.minimize(loss)
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < args.max_iterations):
        sess.run(train_op)
        if iterations % args.print_iterations == 0 and args.verbose:
            curr_loss = loss.eval()
            print("At iterate {}\tf=  {}".format(iterations, curr_loss))
        iterations += 1


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_optimizer(loss):
    print_iterations = args.print_iterations if args.verbose else 0
    if args.optimizer == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B',
            options={'maxiter': args.max_iterations,
                     'disp': print_iterations})
    elif args.optimizer == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(args.learning_rate)
    return optimizer


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def write_video_output(frame, output_img):
    fn = args.content_frame_frmt.format(str(frame).zfill(4))
    path = os.path.join(args.video_output_dir, fn)
    write_image(path, output_img)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def write_image_output(output_img, content_img, style_imgs, init_img):
    out_dir = os.path.join(args.img_output_dir, args.img_name)

    img_path = os.path.join(out_dir, args.img_name + '.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)
    index = 0
    for style_img in style_imgs:
        path = os.path.join(out_dir, 'style_' + str(index) + '.png')
        write_image(path, style_img)
        index += 1

    # save the configuration settings
    out_file = os.path.join(out_dir, 'meta_data.txt')
    f = open(out_file, 'w')
    f.write('image_name: {}\n'.format(args.img_name))
    f.write('content: {}\n'.format(args.content_img))
    index = 0
    for style_img, weight in zip(args.style_imgs, args.style_imgs_weights):
        f.write('styles[' + str(index) + ']: {} * {}\n'.format(weight, style_img))
        index += 1
    index = 0
    if args.style_mask_imgs is not None:
        for mask in args.style_mask_imgs:
            f.write('style_masks[' + str(index) + ']: {}\n'.format(mask))
            index += 1
    f.write('init_type: {}\n'.format(args.init_img_type))
    f.write('content_weight: {}\n'.format(args.content_weight))
    f.write('style_weight: {}\n'.format(args.style_weight))
    f.write('tv_weight: {}\n'.format(args.tv_weight))
    f.write('content_layers: {}\n'.format(args.content_layers))
    f.write('style_layers: {}\n'.format(args.style_layers))
    f.write('optimizer_type: {}\n'.format(args.optimizer))
    f.write('max_iterations: {}\n'.format(args.max_iterations))
    f.write('max_image_size: {}\n'.format(args.max_size))
    f.close()


# ----------------------------------------------------------------------------------------------------------------------
# image loading and processing
# ----------------------------------------------------------------------------------------------------------------------
def get_init_image(init_type, content_img, style_imgs, frame=None):
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        init_img = get_noise_image(args.noise_ratio, content_img)
        return init_img
    # only for video frames
    elif init_type == 'prev':
        init_img = get_prev_frame(frame)
        return init_img
    elif init_type == 'prev_warped':
        init_img = get_prev_warped_frame(frame)
        return init_img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_content_frame(frame):
    fn = args.content_frame_frmt.format(str(frame).zfill(4))
    path = os.path.join(args.video_input_dir, fn)
    img = read_image(path)
    return img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_content_image(content_img):
    path = os.path.join(args.content_img_dir, content_img)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    img = img.astype(np.float32)
    h, w, d = img.shape
    mx = args.max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_style_images(content_img):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in args.style_imgs:
        path = os.path.join(args.style_imgs_dir, style_fn)
        # bgr image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        check_image(img, path)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        style_imgs.append(img)
    return style_imgs


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_noise_image(noise_ratio, content_img):
    np.random.seed(args.seed)
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1. - noise_ratio) * content_img
    return img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_mask_image(mask_img, width, height):
    path = os.path.join(args.content_img_dir, mask_img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    check_image(img, path)
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    mx = np.amax(img)
    img /= mx
    return img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_prev_frame(frame):
    # previously stylized frame
    prev_frame = frame - 1
    fn = args.content_frame_frmt.format(str(prev_frame).zfill(4))
    path = os.path.join(args.video_output_dir, fn)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    return img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_prev_warped_frame(frame):
    prev_img = get_prev_frame(frame)
    prev_frame = frame - 1
    # backwards flow: current frame -> previous frame
    fn = args.backward_optical_flow_frmt.format(str(frame), str(prev_frame))
    path = os.path.join(args.video_input_dir, fn)
    flow = read_flow_file(path)
    warped_img = warp_image(prev_img, flow).astype(np.float32)
    img = preprocess(warped_img)
    return img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def get_content_weights(frame, prev_frame):
    forward_fn = args.content_weights_frmt.format(str(prev_frame), str(frame))
    backward_fn = args.content_weights_frmt.format(str(frame), str(prev_frame))
    forward_path = os.path.join(args.video_input_dir, forward_fn)
    backward_path = os.path.join(args.video_input_dir, backward_fn)
    forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    return forward_weights  # , backward_weights


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def warp_image(src, flow):
    _, h, w = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[1, y, :] = float(y) + flow[1, y, :]
    for x in range(w):
        flow_map[0, :, x] = float(x) + flow[0, :, x]
    # remap pixels to optical flow
    dst = cv2.remap(
        src, flow_map[0], flow_map[1],
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def convert_to_original_colors(content_img, stylized_img):
    content_img = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if args.color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif args.color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif args.color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif args.color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = preprocess(dst)
    return dst


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def render_single_image(content_img, style_img):
    #content_img = get_content_image(args.content_img)
    #style_imgs = get_style_images(content_img)
    with tf.Graph().as_default():
        global args
        args = set_args()
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        #init_img = get_init_image(args.init_img_type, content_img, style_img)
        tick = time.time()
        output_img = stylize(content_img, style_img, content_img)
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))
        return output_img


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def main():
    global args
    args = set_args()
    render_single_image()


if __name__ == '__main__':
    main()
