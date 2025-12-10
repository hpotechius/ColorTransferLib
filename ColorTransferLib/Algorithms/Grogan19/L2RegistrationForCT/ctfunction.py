import numpy as np
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans


def mg_apply_kmeans(image, n_colors, sample_limit=300 * 350, random_state=0):
    """
    Approximate MATLAB mg_applyKMeans:
    - optional subsampling for speed
    - KMeans on colour pixels
    - returns cluster centres in the working colour space (shape: [n_colors, 3])
    """
    img = np.asarray(image, dtype=np.float64)
    h, w, c = img.shape
    pixels = img.reshape(-1, c)

    n_pixels = pixels.shape[0]
    if n_pixels > sample_limit:
        idx = np.random.default_rng(random_state).choice(n_pixels, sample_limit, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    kmeans = KMeans(n_clusters=int(n_colors), n_init=10, random_state=random_state)
    kmeans.fit(sample)
    centers = kmeans.cluster_centers_.astype(np.float64)
    return centers


def mg_quant_image(image, n_colors, sample_limit=300 * 350, random_state=1):
    """
    Approximate MATLAB mg_quantImage (MVQ) using KMeans-based vector quantisation.
    Returns cluster centres (shape: [n_colors, 3]).
    """
    # For simplicity, reuse KMeans-based clustering as MVQ.
    return mg_apply_kmeans(image, n_colors, sample_limit=sample_limit, random_state=random_state)


def _fit_linear_map(X, Y, reg=1e-3):
    """
    Fit affine map f(c) = A c + b from X to Y using regularized least squares.

    X, Y : (N,3)
    Returns:
        M : (4,3)  where [c,1] @ M ≈ Y
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, d = X.shape
    assert d == 3, "Expected 3D colour vectors."

    X_aug = np.concatenate([X, np.ones((n, 1))], axis=1)  # (n,4)
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y

    # Tikhonov regularization
    reg_mat = reg * np.eye(XtX.shape[0], dtype=np.float64)
    M = np.linalg.solve(XtX + reg_mat, XtY)  # (4,3)
    return M


def _apply_linear_map(points, M):
    """
    Apply affine map defined by M (4,3) to points (M,3).
    """
    pts = np.asarray(points, dtype=np.float64)
    m, d = pts.shape
    assert d == 3
    pts_aug = np.concatenate([pts, np.ones((m, 1))], axis=1)  # (m,4)
    return pts_aug @ M  # (m,3)


def mg_initialize_config(X, Y, colour_space):
    """
    Minimal Python configuration similar to MATLAB mg_initialize_config.
    Now for linear mapping.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    config = {
        "X": X,
        "Y": Y,
        "colour_space": colour_space,
        "AnnSteps": 1,      # single step; can be increased if needed
        "iter": 1,
        "lambda": 1e-3,     # regularization for linear map
    }
    return config


def gmmreg_rbf_L2(config):
    """
    Python replacement for gmmreg_rbf_L2.m using a regularized linear map.
    Computes mapping from X to Y in colour space.

    Returns:
        param            : dict with {'M'} (affine map)
        transformed_model: X mapped by M
        history          : None
        config_out       : unchanged config dict
    """
    X = config["X"]
    Y = config["Y"]
    reg = float(config.get("lambda", 1e-3))

    M = _fit_linear_map(X, Y, reg=reg)
    param = {"M": M}
    transformed_model = _apply_linear_map(X, M)
    history = None
    return param, transformed_model, history, config


def mg_transform_tps_parallel(param, full_transform, ctrl_pts_unused):
    """
    Apply the learned linear map to all pixels of the target image.
    Kept name for compatibility with original ctfunction.
    """
    M = param["M"]
    pts = np.asarray(full_transform, dtype=np.float64)
    return _apply_linear_map(pts, M)


def ctfunction(he2, he1, cluster_fun="MVQ", n_colors=50, colour_space="RGB"):
    """
    Python port of Grogan19/L2RegistrationForCT/ctfunction.m using
    KMeans clustering + regularized linear colour mapping.
    """
    he1 = np.asarray(he1, dtype=np.float64)
    he2 = np.asarray(he2, dtype=np.float64)

    # bring to 0..255
    if he1.max() <= 1.0:
        he1 = he1 * 255.0
    if he2.max() <= 1.0:
        he2 = he2 * 255.0

    # colour space for target
    if colour_space == "CIELab" and cluster_fun == "KMeans":
        he1_work = rgb2lab(he1 / 255.0)
    elif colour_space == "CIELab" and cluster_fun == "MVQ":
        # MVQ not supported in Lab here, fall back to RGB
        colour_space = "RGB"
        he1_work = he1
    else:
        he1_work = he1

    fnrows, fncols, _ = he1_work.shape
    full_transform = he1_work.reshape(-1, 3)

    # palette colour space
    if colour_space == "CIELab":
        he2_work = rgb2lab(he2 / 255.0)
    else:
        he2_work = he2

    # clustering
    if cluster_fun == "KMeans":
        X = mg_apply_kmeans(he1_work, n_colors)
        Y = mg_apply_kmeans(he2_work, n_colors)
    elif cluster_fun == "MVQ":
        X = mg_quant_image(he1_work, n_colors)
        Y = mg_quant_image(he2_work, n_colors)
    else:
        raise ValueError(f"Unsupported cluster_fun: {cluster_fun}")

    # registration config
    config = mg_initialize_config(X, Y, colour_space)

    # single (or few) annealing-like steps; here just one
    ann_steps = int(config["AnnSteps"])
    for i in range(ann_steps):
        config["iter"] = ann_steps - i
        param, transformed_model, history, config = gmmreg_rbf_L2(config)
        # scale parameter currently unused in linear version

    # apply colour mapping to all pixels
    full_transform_out = mg_transform_tps_parallel(
        param, full_transform, None
    )

    if colour_space == "CIELab":
        # back to RGB in [0, 255]
        lab_img = full_transform_out.reshape(fnrows, fncols, 3)
        rgb_img = lab2rgb(lab_img)
        rgb_img = np.clip(rgb_img, 0.0, 1.0)
        final_result = (rgb_img * 255.0).astype(np.uint8)
    else:
        rgb_img = full_transform_out.reshape(fnrows, fncols, 3)
        final_result = np.clip(rgb_img, 0.0, 255.0).astype(np.uint8)

    return final_result
