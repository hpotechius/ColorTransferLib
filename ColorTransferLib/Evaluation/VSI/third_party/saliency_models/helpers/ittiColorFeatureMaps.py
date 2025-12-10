import cv2
import numpy as np
def compute(r, g, b, I):
    # intensity I is (r+g+b)/3

    max_I = I.max()
    # normalisation, for decoupling hue from intensity #ittikoch98pami
    r = np.divide(r, I, out=np.zeros_like(r), where=I>max_I/10.)
    g = np.divide(g, I, out=np.zeros_like(g), where=I>max_I/10.)
    b = np.divide(b, I, out=np.zeros_like(b), where=I>max_I/10.)

    # calculating broadly-tuned  color channels
    R = r - (g+b)/2.
    R = R*(R>=0)

    G = g - (r+b)/2.
    G = G*(G>=0)

    B = b - (r+g)/2.
    B = B*(B>=0)

    Y = (r+g)/2 - cv2.absdiff(r, g)/2. - b
    Y = Y*(Y>=0)

    RG = cv2.absdiff(R, G)
    BY = cv2.absdiff(B, Y)

    featMaps = {
        0: RG,
        1: BY,
        2: I
    }
    return featMaps
