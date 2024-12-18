import numpy as np
import cv2
from scipy.ndimage.filters import maximum_filter

def calculate(mat, thresh):
    (w,h) = mat.shape
    sum_local_max = mat[0][0]
    count_local_max = 0
    global_max = mat[0][0]
    for i in range(1, w-1):
        for j in range(1, h-1):
            if mat[i][j] > max(mat[i-1][j-1],mat[i-1][j],mat[i-1][j+1],
                               mat[i][j-1],              mat[i][j+1],
                               mat[i+1][j-1],mat[i+1][j],mat[i+1][j+1]) and mat[i,j]>thresh:
                if mat[i][j] > global_max:
                    global_max = mat[i][j]

                sum_local_max += mat[i][j]
                count_local_max +=1

    if count_local_max > 0:
        local_max_avg = float(sum_local_max)/float(count_local_max)
    else:
        local_max_avg = 0.0
    return global_max, count_local_max, local_max_avg

def processNormalization(mat):
    M = 10
    thresh  = M/10
    mat = cv2.normalize(mat, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mat = mat*M
    g_max, c_max, l_max_avg = calculate(mat, thresh)
    # print g_max, c_max, l_max_avg
    if c_max>1:
        res = mat* (M-l_max_avg)**2
    elif c_max == 1:
        res = mat * M**2
    else:
        res = mat

    return res

def process2(mat):
     M = 8.0 # an arbitrary global maxima for which the image is scaled
     mat = cv2.convertScaleAbs(mat, alpha=M/mat.max(), beta = 0.0)
     w, h = mat.shape
     maxima = maximum_filter(mat, size=(1, 1))
     maxima = (mat == maxima)
     mnum = maxima.sum()
     maxima = np.multiply(maxima, mat)
     mbar = float(maxima.sum()) / mnum
     return mat * (M-mbar)**2