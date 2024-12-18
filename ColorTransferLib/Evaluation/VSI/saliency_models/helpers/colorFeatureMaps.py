import numpy as np
def compute(r, g, b, L):
    # Input is the r, g, b channels of the image

    #CBY Feature Map
    min_rg = np.minimum(r, g)
    b_min_rg = np.abs(np.subtract(b, min_rg))
    CBY = np.divide(b_min_rg, L, out=np.zeros_like(L), where=L != 0)

    #CRG Feature Map
    r_g = np.abs(np.subtract(r,g))
    CRG = np.divide(r_g, L, out=np.zeros_like(L), where=L != 0)

    featMaps = {}
    featMaps['CBY'] = CBY
    featMaps['CRG'] = CRG
    featMaps['L'] = L
    return featMaps
