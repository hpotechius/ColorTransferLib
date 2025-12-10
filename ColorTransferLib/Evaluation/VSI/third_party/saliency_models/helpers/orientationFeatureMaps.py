from .gaborKernelCalculator import getGaborKernels
import cv2
import numpy as np

def compute(L, gaborparams, thetas):
    # L = Intensity Map
    # L = np.maximum(np.maximum(r, g), b)

    kernels = getGaborKernels(gaborparams, thetas)
    featMaps = []
    for th in thetas:
        kernel_0  = kernels[th]['0']
        kernel_90 = kernels[th]['90']
        o1 = cv2.filter2D(L, -1, kernel_0, borderType=cv2.BORDER_REPLICATE)
        o2 = cv2.filter2D(L, -1, kernel_90, borderType=cv2.BORDER_REPLICATE)
        o = np.add(abs(o1), abs(o2))
        featMaps.append(o)

    return featMaps

