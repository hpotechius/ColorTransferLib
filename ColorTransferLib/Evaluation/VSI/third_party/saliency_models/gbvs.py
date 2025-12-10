import time
import cv2
from .helpers import colorFeatureMaps, graphBasedActivation, orientationFeatureMaps
import numpy as np
from matplotlib import pyplot as plt

def calculateFeatureMaps(r, g, b, L, params):
    colorMaps = colorFeatureMaps.compute(r, g, b, L)
    orientationMaps = orientationFeatureMaps.compute(L, params['gaborparams'], params['thetas'])
    allFeatureMaps = {
        0: colorMaps['CBY'],
        1: colorMaps['CRG'],
        2: colorMaps['L'],
        3: orientationMaps
    }
    return allFeatureMaps

def getPyramids(image, max_level):
    imagePyr = [cv2.pyrDown(image)]
    for i in range(1, max_level):
        # imagePyr.append(cv2.resize(p, (32, 28), interpolation=cv2.INTER_CUBIC))
        imagePyr.append(cv2.pyrDown(imagePyr[i-1]))
    return imagePyr[1:]

def run(image, params):
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    L = np.maximum(np.maximum(r, g), b)

    b_pyr = getPyramids(b, params['max_level'])
    g_pyr = getPyramids(g, params['max_level'])
    r_pyr = getPyramids(r, params['max_level'])
    L_pyr = getPyramids(L, params['max_level'])

    featMaps = {
        0: [],
        1: [],
        2: [],
        3: []
    }

    # calculating feature maps

    for i in range(0, len(b_pyr)):
        p_r = r_pyr[i]
        p_g = g_pyr[i]
        p_b = b_pyr[i]
        p_L = L_pyr[i]

        maps = calculateFeatureMaps(p_r, p_g, p_b, p_L, params)
        # we calculate feature maps and then resize
        for i in range(0,3):
            resized_m = cv2.resize(maps[i], (32, 28), interpolation=cv2.INTER_CUBIC)
            featMaps[i].append(resized_m)

        for m in maps[3]:
            resized_m = cv2.resize(m, (32, 28), interpolation=cv2.INTER_CUBIC)
            featMaps[3].append(resized_m)
        # featMaps[0].append(maps[0])
        # featMaps[1].append(maps[1])
        # featMaps[2].append(maps[2])

    # calculating activation maps

    activationMaps = []
    activation_sigma = params['sigma_frac_act']*np.mean([32, 28]) # the shape of map

    for i in range(0,4):
        for map in featMaps[i]:
            activationMaps.append(graphBasedActivation.calculate(map, activation_sigma))


    # normalizing activation maps

    normalisedActivationMaps = []
    normalisation_sigma = params['sigma_frac_norm']*np.mean([32, 28])

    for map in activationMaps:
        normalisedActivationMaps.append(graphBasedActivation.normalize(map, normalisation_sigma))


    # combine normalised maps

    mastermap = normalisedActivationMaps[0]
    for i in range(1, len(normalisedActivationMaps)):
        mastermap = np.add(normalisedActivationMaps[i], mastermap)


    # post process

    gray = cv2.normalize(mastermap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # blurred = cv2.GaussianBlur(gray,(4,4), 4)
    # gray2 = cv2.normalize(blurred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mastermap_res = cv2.resize(gray, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    return mastermap_res

def setupParams():
    gaborparams = {
        'stddev': 2,
        'elongation': 2,
        'filterSize': -1,
        'filterPeriod': np.pi
    }

    params = {
        'gaborparams': gaborparams,
        'sigma_frac_act': 0.15,
        'sigma_frac_norm': 0.06,
        'max_level': 4,
        'thetas': [0, 45, 90, 135]
    }

    return params


def compute_saliency(input_image):
    if type(input_image) is str:
        input_image = cv2.imread(input_image)

    params = setupParams()
    return run(image=input_image / 255.0, params=params) * 255.0
