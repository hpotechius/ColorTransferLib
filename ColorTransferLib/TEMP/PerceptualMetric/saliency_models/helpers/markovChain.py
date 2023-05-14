import numpy as np

def solve(mat, tolerance):
    w,h = mat.shape
    diff = 1
    v = np.divide(np.ones((w, 1), dtype=np.float32), w)
    oldv = v
    oldoldv = v

    while diff > tolerance :
        oldv = v
        oldoldv = oldv
        v = np.dot(mat,v)
        diff = np.linalg.norm(oldv - v, ord=2)
        s = sum(v)
        if s>=0 and s< np.inf:
            continue
        else:
            v = oldoldv
            break

    v = np.divide(v, sum(v))

    return v