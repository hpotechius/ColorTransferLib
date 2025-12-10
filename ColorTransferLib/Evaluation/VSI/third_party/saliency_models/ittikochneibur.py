import time
import cv2
from saliency_models.helpers import ittiColorFeatureMaps, localMaximas, \
    ittiKochCenterSurroundFeatures, orientationFeatureMaps
import numpy as np
from matplotlib import pyplot as plt

def norm01(mat):
    return cv2.normalize(mat, None, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def calculateFeatureMaps(r, g, b, L, params):
    colorMaps = ittiColorFeatureMaps.compute(r, g, b, L)
    orientationMaps = orientationFeatureMaps.compute(L, params['gaborparams'], params['thetas'])
    allFeatureMaps = {
        0: colorMaps[0],
        1: colorMaps[1],
        2: colorMaps[2],
        3: orientationMaps
    }
    return allFeatureMaps

def getPyramid(image, max_level):
    imagePyramid = {
        0: image
    } # scale zero = 1:1

    for i in range(1, max_level):
        imagePyramid[i] = cv2.pyrDown(imagePyramid[i-1])

    return imagePyramid

def run(image, params):
    b = image[:,:,0]/255.
    g = image[:,:,1]/255.
    r = image[:,:,2]/255.
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.

    b_pyr = getPyramid(b, params['max_level'])
    g_pyr = getPyramid(g, params['max_level'])
    r_pyr = getPyramid(r, params['max_level'])
    I_pyr = getPyramid(I, params['max_level'])

    # calculating scale-wise feature maps
    scaledFeaturePyramids = {}

    for i in range(2, len(b_pyr)):
        p_r = r_pyr[i]
        p_g = g_pyr[i]
        p_b = b_pyr[i]
        p_L = I_pyr[i]

        maps = calculateFeatureMaps(p_r, p_g, p_b, p_L, params)

        scaledFeaturePyramids[i] = maps


    # calculating center surround feature maps

    centerSurroundFeatureMaps = ittiKochCenterSurroundFeatures.compute(scaledFeaturePyramids)


    # normalizing activation maps
    normalised_maps =[]
    norm_maps = centerSurroundFeatureMaps.copy()
    for i in range(0,4):
        for mat in norm_maps[i]:
            # Resizing to sigma = 4 maps
            nmap = localMaximas.processNormalization(mat)
            nmap = cv2.resize(nmap, (b_pyr[4].shape[1], b_pyr[4].shape[0]), interpolation=cv2.INTER_CUBIC)
            normalised_maps.append(nmap)


    # combine normalised maps
    comb_maps = []
    cfn = len(norm_maps[0])+len(norm_maps[1])
    ifn = len(norm_maps[2])
    ofn = len(norm_maps[3])

    comb_maps.append(normalised_maps[0])
    for i in range(1, cfn):
        comb_maps[0] = np.add(comb_maps[0], normalised_maps[i])

    comb_maps.append(normalised_maps[cfn])
    for i in range(cfn+1, cfn + ifn):
        comb_maps[1] = np.add(comb_maps[1], normalised_maps[i])

    comb_maps.append(normalised_maps[cfn + ifn])
    for i in range(cfn + ifn + 1, cfn + ifn + ofn):
        comb_maps[2] = np.add(comb_maps[2], normalised_maps[i])


    # normlaise top channle maps
    ntcmaps = [None]*3
    for i in range(0,3):
        ntcmaps[i] = localMaximas.processNormalization(comb_maps[i])

    # add all of them
    mastermap = (ntcmaps[0] + ntcmaps[1] + ntcmaps[2])/3.0

    #post processing
    gray = norm01(mastermap)
    # blurred = cv2.GaussianBlur(gray,(3,3), 4)
    # gray = norm01(blurred)
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
        'max_level': 9,
        'thetas': [0, 45, 90, 135]
    }

    return params


def compute_saliency(input_image):
    if type(input_image) is str:
        input_image = cv2.imread(input_image)

    params = setupParams()
    return run(image=input_image, params=params) * 255.0
