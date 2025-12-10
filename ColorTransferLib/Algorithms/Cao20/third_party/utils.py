import os

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import tf_slim as slim
from PIL import Image
from pyntcloud import PyntCloud
from sklearn.metrics import auc, roc_curve

import matplotlib
try:
    matplotlib.use('TkAgg')#uncomment this line if you are using macos
except:
    pass
import matplotlib.pyplot as plt

def show_all_trainable_variables():
    model_vars = tf.compat.v1.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def display_point(pts, color, color_label=None, title=None, fname=None, axis="on", marker_size=5):
    pts = np.squeeze(pts)
    if isinstance(color, np.ndarray):
        color = np_color_to_hex_str(np.squeeze(color))
    DPI = 300
    PIX_h = 1000
    MARKER_SIZE = marker_size
    if color_label is None:
        PIX_w = PIX_h
    else:
        PIX_w = PIX_h * 2
    X = pts[:, 0]
    Y = pts[:, 2]
    Z = pts[:, 1]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    fig = plt.figure()
    fig.set_size_inches(PIX_w / DPI, PIX_h / DPI)
    if axis == "off":
        plt.subplots_adjust(top=1.2, bottom=-0.2, right=1.5, left=-0.5, hspace=0, wspace=-0.7)
    plt.margins(0, 0)
    if color_label is None:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, - Y, Z, c=color, edgecolors="none", s=MARKER_SIZE, depthshade=True)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.axis(axis)
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        # ax.set_zticklabels([])
        # ax.set_axis_off()
        if title is not None:
            ax.set_title(title, fontdict={'fontsize': 30})

        ax.set_aspect("auto")
        # ax.grid("off")
        if fname:
            plt.savefig(fname, transparent=True, dpi=DPI)
            plt.close(fig)
        else:
            plt.show()
    else:
        ax = fig.add_subplot(121, projection='3d')
        bx = fig.add_subplot(122, projection='3d')
        ax.scatter(X, Y, Z, c=color, edgecolors="none", s=MARKER_SIZE, depthshade=True)
        bx.scatter(X, Y, Z, c=color_label, edgecolors="none", s=MARKER_SIZE, depthshade=True)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        bx.set_xlim(mid_x - max_range, mid_x + max_range)
        bx.set_ylim(mid_y - max_range, mid_y + max_range)
        bx.set_zlim(mid_z - max_range, mid_z + max_range)
        bx.patch.set_alpha(0)
        ax.set_aspect("equal")
        ax.grid("off")
        bx.set_aspect("equal")
        bx.grid("off")
        ax.axis('off')
        bx.axis("off")
        plt.axis('off')
        if fname:
            plt.savefig(fname, transparent=True, dpi=DPI)
            plt.close(fig)

        else:
            plt.show()


def int16_to_hex_str(color):
    hex_str = ""
    color_map = {i: str(i) for i in range(10)}
    color_map.update({10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "F"})
    # print(color_map)
    hex_str += color_map[color // 16]
    hex_str += color_map[color % 16]
    return hex_str


def horizontal_concatnate_pic(fout, *fnames):
    images = [Image.open(i).convert('RGB') for i in fnames]
    # images = map(Image.open, fnames)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(fout)


def vertical_concatnate_pic(fout, *fnames):
    images = [Image.open(i).convert('RGB') for i in fnames]
    # images = map(Image.open, fnames)
    widths, heights = zip(*(i.size for i in images))
    max_widths = max(widths)
    width_ratio = [max_widths / width for width in widths]
    new_height = [int(width_ratio[idx]) * height for idx, height in enumerate(heights)]

    new_images = [i.resize((max_widths, new_height[idx])) for idx, i in enumerate(images)]
    total_heights = sum(new_height)

    new_im = Image.new('RGB', (max_widths, total_heights))

    x_offset = 0
    for im in new_images:
        new_im.paste(im, (0, x_offset))
        x_offset += im.size[1]

    new_im.save(fout)


def rgb_to_hex_str(*rgb):
    hex_str = "#"
    for item in rgb:
        hex_str += int16_to_hex_str(item)
    return hex_str


def np_color_to_hex_str(color):
    """
    :param color: an numpy array of shape (N, 3)
    :return: a list of hex color strings
    """
    hex_list = []
    for rgb in color:
        hex_list.append(rgb_to_hex_str(rgb[0], rgb[1], rgb[2]))
    return hex_list


def load_h5(path, *kwd):
    f = h5py.File(path)
    list_ = []
    for item in kwd:
        list_.append(f[item][:])
        print("{0} of shape {1} loaded!".format(item, f[item][:].shape))
        if item == "ndata" or item == "data":
            pass  # print(np.mean(f[item][:], axis=1))
        if item == "color":
            print("color is of type {}".format(f[item][:].dtype))
    return list_


def load_single_cat_h5(cat, num_pts, type, *kwd):
    fpath = os.path.join("./data/category_h5py", cat, "PTS_{}".format(num_pts), "ply_data_{}.h5".format(type))
    return load_h5(fpath, *kwd)


def printout(flog, data):  # follow up
    print(data)
    flog.write(data + '\n')


def save_ply(data, color, fname):
    color = color.astype(np.uint8)
    df1 = pd.DataFrame(data, columns=["x", "y", "z"])
    df2 = pd.DataFrame(color, columns=["red", "green", "blue"])
    pc = PyntCloud(pd.concat([df1, df2], axis=1))
    pc.to_file(fname)


def label2onehot(labels, m):
    idx = np.eye(m)
    onehot_labels = np.zeros(shape=(labels.shape[0], m))
    for idx, i in enumerate(labels):
        onehot_labels[idx] = idx[i]
    return onehot_labels


def multiclass_AUC(y_true, y_pred):
    """

    :param y_true: shape (num_instance,)
    :param y_pred: shape (num_instance, num_class)
    :return:
    """
    num_classes = np.unique(y_true).shape[0]
    print(num_classes)
    tpr = dict()
    fpr = dict()
    roc_auc = dict()
    num_instance = dict()
    total_roc_auc = 0
    for i in range(num_classes):
        binary_label = np.where(y_true == i, 1, 0)
        class_score = y_pred[:, i]
        num_instance[i] = np.sum(y_true == i)
        fpr[i], tpr[i], _ = roc_curve(binary_label, class_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
        total_roc_auc += roc_auc[i]
    return total_roc_auc / 16
    # print(roc_auc, num_instance)


def softmax(logits):
    """

    :param logits: of shape (num_instance, num_classes)
    :return: prob of the same shape as that of logits, with each row summed up to 1
    """
    assert logits.shape[-1] == 16
    regulazation = np.max(logits, axis=-1)  # (num_instance,)
    logits -= regulazation[:, np.newaxis]
    prob = np.exp(logits) / np.sum(np.exp(logits), axis=-1)[:, np.newaxis]
    assert prob.shape == logits.shape
    return prob


def construct_label_weight(label, weight):
    """

    :param label: a numpy ndarray of shape (batch_size,) with each entry in [0, num_class)
    :param weight: weight list of length num_class
    :return: a numpy ndarray of the same shape as that of label
    """
    return [weight[i] for i in label]


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def generate_sphere(num_pts, radius):
    # http://electron9.phys.utk.edu/vectors/3dcoordinates.htm
    r = np.random.rand(num_pts, 1) * radius
    f = np.random.rand(num_pts, 1) * np.pi * 2
    q = np.random.rand(num_pts, 1) * np.pi
    x = r * np.sin(q) * np.cos(f)
    y = r * np.sin(q) * np.sin(f)
    z = r * np.cos(q)
    return np.concatenate((x, y, z), axis=-1)


def generate_sphere_surface(num_pts, radius):
    r = radius
    f = np.random.rand(num_pts, 1) * np.pi * 2
    q = np.random.rand(num_pts, 1) * np.pi
    x = r * np.sin(q) * np.cos(f)
    y = r * np.sin(q) * np.sin(f)
    z = r * np.cos(q)
    return np.concatenate((x, y, z), axis=-1)


def generate_cube(num_pts, length):
    return (np.random.rand(num_pts, 3) - 0.5) * length


def generate_cube_surface(num_pts, length):
    pass


def generate_ncolor(num_pts):
    return ((np.ones((num_pts, 3)) * 127) - 127.5) / 127.5


def generate_plane(num_pts, length, type="xy"):
    pts = (np.random.rand(num_pts, 2) - 0.5) * length
    if type == "xy":
        pts = np.insert(pts, 2, 0, axis=1)
    elif type == "xz":
        pts = np.insert(pts, 1, 0, axis=1)
    elif type == "yz":
        pts = np.insert(pts, 0, 0, axis=1)
    return pts


def img_to_set(img_path, img_name):
    img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
    w, h = img.size
    # img = img.resize((int(w/10), int(h/10)), Image.LANCZOS)
    # img.save(os.path.join(img_path, img_name.split(".")[0] + "_resized.jpg"), "jpeg")
    print(img.size)
    return np.reshape(np.array(img), [-1, 3])


def prepare_content_or_style(path, downsample_points=None):
    if path.endswith("ply"):
        content = PyntCloud.from_file(path).points.values
        if downsample_points:
            mask = np.random.choice(content.shape[0], downsample_points)
            content = content[mask]
        content_ndata = content[:, :3]
        content_ncolor = (content[:, 3:6] - 127.5) / 127.5
        return content_ndata, content_ncolor
    elif path.endswith("npy"):
        content = np.load(path)
        if downsample_points:
            mask = np.random.choice(content.shape[0], downsample_points)
            content = content[mask]
        content_ndata = content[:, :3]
        content_ncolor = (content[:, 3:6] - 127.5) / 127.5
        return content_ndata, content_ncolor
    else:
        img = Image.open(path).convert("RGB")
        style_color = np.reshape(np.array(img), [-1, 3])
        style_color = (style_color - 127.5) / 127.5
        if downsample_points:
            mask = np.random.choice(style_color.shape[0], downsample_points)
            style_color = style_color[mask]
        return style_color
