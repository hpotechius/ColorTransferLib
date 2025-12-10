import numpy as np
import numpy.matlib
import math


def getGaborKernel(gaborparams, angle, phase):
    gp = gaborparams
    major_sd = gp['stddev']
    minor_sd = major_sd * gp['elongation']
    max_sd = max(major_sd, minor_sd)

    sz = gp['filterSize']
    if sz == -1:
        sz = math.ceil(max_sd * math.sqrt(10))
    else:
        sz = math.floor(sz / 2)

    psi = np.pi / 180 * phase
    rtDeg = np.pi / 180 * angle

    omega = 2 * np.pi / gp['filterPeriod']
    co = math.cos(rtDeg)
    si = -math.sin(rtDeg)
    major_sigq = 2 * pow(major_sd, 2)
    minor_sigq = 2 * pow(minor_sd, 2)

    vec = range(-int(sz), int(sz) + 1)
    vlen = len(vec)
    vco = [i * co for i in vec]
    vsi = [i * si for i in vec]

    # major = np.matlib.repmat(np.asarray(vco).transpose(), 1, vlen) + np.matlib.repmat(vsi, vlen, 1)
    a = np.tile(np.asarray(vco).transpose(), (vlen, 1)).transpose()
    b = np.matlib.repmat(vsi, vlen, 1)
    major = a + b
    major2 = np.power(major, 2)

    # minor = np.matlib.repmat(np.asarray(vsi).transpose(), 1, vlen) - np.matlib.repmat(vco, vlen, 1)
    a = np.tile(np.asarray(vsi).transpose(), (vlen, 1)).transpose()
    b = np.matlib.repmat(vco, vlen, 1)
    minor = a + b
    minor2 = np.power(minor, 2)

    a = np.cos(omega * major + psi)
    b = np.exp(-major2 / major_sigq - minor2 / minor_sigq)
    # result = np.cos(omega * major + psi) * exp(-major2/major_sigq - minor2/minor_sigq)
    result = np.multiply(a, b)

    filter1 = np.subtract(result, np.mean(result.reshape(-1)))
    filter1 = np.divide(filter1, np.sqrt(np.sum(np.power(filter1.reshape(-1), 2))))
    return filter1


def getGaborKernels(gaborparams, thetas):
    gaborKernels = {}
    for th in thetas:
        gaborKernels[th] = {}
        gaborKernels[th]['0'] = getGaborKernel(gaborparams, th, 0)
        gaborKernels[th]['90'] = getGaborKernel(gaborparams, th, 90)

    return gaborKernels
