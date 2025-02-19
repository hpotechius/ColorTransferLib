import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', type=int, help='an integer')
parser.add_argument('-k', type=int,
                    help='an integer')


args = parser.parse_args()

def adjust_pts_order(pts_2ds):

    ''' sort rectangle points by counterclockwise '''

    cen_x, cen_y = np.mean(pts_2ds, axis=0)
    #refer_line = np.array([10,0])

    d2s = []
    for i in range(len(pts_2ds)):

        o_x = pts_2ds[i][0] - cen_x
        o_y = pts_2ds[i][1] - cen_y

        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([pts_2ds[i], atan2])

    d2s = sorted(d2s, key=lambda x:x[1])

    order_2ds = np.array([x[0] for x in d2s])

    return order_2ds


semantic_array = np.load('./memory_build/semantic_color/semantic_array_10000.npy')
color_array = np.load('./memory_build/semantic_color/color_array_10000.npy')
print('read done')
# hyper setting
mmm = args.m
kkk = args.k
assert mmm
assert kkk

print('m = ', mmm, 'k = ', kkk)
pca = PCA(kkk)
semantic_array = pca.fit_transform(semantic_array)
print('PCA done')

estimator = KMeans(n_clusters=mmm, verbose=1, n_init=10)
estimator.fit(semantic_array)
label_pred = estimator.labels_
semantic_embed = estimator.cluster_centers_
inertia = estimator.inertia_

print('Semantic clustering done')

color_embed = np.zeros((4, mmm, 2))

for i in range(mmm):
    print(i)
    sub_color = color_array[label_pred == i, :]
    print(sub_color.shape)
    if sub_color.shape[0] < 4:
        centroids = np.zeros((4, 2))
        centroids[:sub_color.shape[0], :] = sub_color
        centroids = np.round(centroids).astype('int')
        color_embed[:,i,:] = centroids
        continue
    estimator = KMeans(n_clusters=4, n_init=10)
    estimator.fit(sub_color)
    centroids = estimator.cluster_centers_
    centroids = np.round(centroids).astype('int')
    color_embed[:,i,:] = centroids

print('Color clustering done')

os.makedirs('./memory_build/semantic_color_cluster/', exist_ok=True)
np.save(f'./memory_build/semantic_color_cluster/semantic_embed_10k_m{mmm}_k{kkk}', semantic_embed)
np.save(f'./memory_build/semantic_color_cluster/color_embed_10k_m{mmm}_k{kkk}', color_embed)

print('saved')