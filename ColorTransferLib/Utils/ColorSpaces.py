"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
import math

# ----------------------------------------------------------------------------------------------------------------------
# Ruderman l-alpha-beta color space conversions (pure NumPy, CPU only)
# ----------------------------------------------------------------------------------------------------------------------
class ColorSpaces:
    # ------------------------------------------------------------------------------------------------------------------
    # RGB -> l-alpha-beta (Ruderman lαβ, NOT CIE L*a*b*)
    # img: array with shape (..., 3), values in [0,1] or [0,255] (used as-is)
    # returns array with same shape (..., 3)
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rgb_to_lalphabeta(img):
        img_arr = np.asarray(img, dtype=np.float32)
        if img_arr.shape[-1] != 3:
            raise ValueError(f"rgb_to_lalphabeta expects last dimension = 3, got shape {img_arr.shape}")

        orig_shape = img_arr.shape
        flat = img_arr.reshape(-1, 3)  # (N,3)

        m_rgb2lms = np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ], dtype=np.float32)

        m_lms2lab1 = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -2.0],
            [1.0, -1.0, 0.0]
        ], dtype=np.float32)

        m_lms2lab2 = np.array([
            [1.0 / math.sqrt(3.0), 0.0, 0.0],
            [0.0, 1.0 / math.sqrt(6.0), 0.0],
            [0.0, 0.0, 1.0 / math.sqrt(2.0)]
        ], dtype=np.float32)

        # RGB -> LMS
        lms = flat @ m_rgb2lms.T  # (N,3)

        # avoid log(0) / negative
        eps = 1e-12
        lms = np.maximum(lms, eps)
        lms_log = np.log(lms)

        # LMS -> lαβ
        tmp = lms_log @ m_lms2lab1.T
        lab_flat = tmp @ m_lms2lab2.T  # (N,3)

        return lab_flat.reshape(orig_shape)

    # ------------------------------------------------------------------------------------------------------------------
    # l-alpha-beta -> RGB (Ruderman lαβ, NOT CIE L*a*b*)
    # img: array with shape (..., 3) in l-alpha-beta space
    # returns array with same shape (..., 3) in RGB space
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def lalphabeta_to_rgb(img):
        img_arr = np.asarray(img, dtype=np.float32)
        if img_arr.shape[-1] != 3:
            raise ValueError(f"lalphabeta_to_rgb expects last dimension = 3, got shape {img_arr.shape}")

        orig_shape = img_arr.shape
        flat = img_arr.reshape(-1, 3)  # (N,3)

        m_lms2rgb = np.array([
            [4.4679, -3.5873, 0.1193],
            [-1.2186, 2.3809, -0.1624],
            [0.0497, -0.2439, 1.2045]
        ], dtype=np.float32)

        m_lab2lms1 = np.array([
            [math.sqrt(3.0) / 3.0, 0.0, 0.0],
            [0.0, math.sqrt(6.0) / 6.0, 0.0],
            [0.0, 0.0, math.sqrt(2.0) / 2.0]
        ], dtype=np.float32)

        m_lab2lms2 = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -2.0, 0.0]
        ], dtype=np.float32)

        # lαβ -> LMS (log domain)
        tmp = flat @ m_lab2lms1.T
        lms_log = tmp @ m_lab2lms2.T

        # exponentiate, clamp to avoid overflow
        lms = np.exp(np.clip(lms_log, -50.0, 50.0))

        # LMS -> RGB
        rgb_flat = lms @ m_lms2rgb.T  # (N,3)

        return rgb_flat.reshape(orig_shape)

    # ------------------------------------------------------------------------------------------------------------------
    # sRGB (D65) -> CIE L*a*b* (D65)
    # img: array with shape (..., 3); values can be in [0,1] or [0,255] (auto-detected)
    # returns array with same shape (..., 3) where channels are (L*, a*, b*)
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rgb_to_cielab(img):
        img_arr = np.asarray(img, dtype=np.float32)
        if img_arr.shape[-1] != 3:
            raise ValueError(f"rgb_to_cielab expects last dimension = 3, got shape {img_arr.shape}")

        orig_shape = img_arr.shape
        rgb = img_arr.reshape(-1, 3)  # (N,3)

        # normalize to [0,1] if input looks like 0..255
        if rgb.max() > 1.1:
            rgb = rgb / 255.0

        # sRGB -> linear RGB
        def srgb_to_linear(v):
            return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

        rgb_lin = srgb_to_linear(rgb)

        # linear RGB -> XYZ (sRGB D65)
        M_rgb2xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=np.float32)
        xyz = rgb_lin @ M_rgb2xyz.T  # (N,3)

        # normalize by D65 white point (Xn,Yn,Zn) with Yn = 1.0
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        x = xyz[:, 0] / Xn
        y = xyz[:, 1] / Yn
        z = xyz[:, 2] / Zn

        # helper f(t)
        delta = 6.0 / 29.0
        delta3 = delta ** 3
        def f(t):
            return np.where(t > delta3, np.cbrt(t), t / (3 * delta * delta) + 4.0 / 29.0)

        fx = f(x)
        fy = f(y)
        fz = f(z)

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)

        lab = np.stack([L, a, b], axis=-1)  # (N,3)
        return lab.reshape(orig_shape)

    # ------------------------------------------------------------------------------------------------------------------
    # CIE L*a*b* (D65) -> sRGB (D65)
    # img: array with shape (..., 3) where channels are (L*, a*, b*)
    # returns array with same shape (..., 3) in sRGB, values in [0,1], dtype float32
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def cielab_to_rgb(img):
        lab_arr = np.asarray(img, dtype=np.float32)
        if lab_arr.shape[-1] != 3:
            raise ValueError(f"cielab_to_rgb expects last dimension = 3, got shape {lab_arr.shape}")

        orig_shape = lab_arr.shape
        lab = lab_arr.reshape(-1, 3)  # (N,3)

        L = lab[:, 0]
        a = lab[:, 1]
        b = lab[:, 2]

        # inverse f
        delta = 6.0 / 29.0

        def f_inv(t):
            return np.where(t > delta, t ** 3, 3 * delta * delta * (t - 4.0 / 29.0))

        fy = (L + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - b / 200.0

        x = f_inv(fx)
        y = f_inv(fy)
        z = f_inv(fz)

        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        X = x * Xn
        Y = y * Yn
        Z = z * Zn

        xyz = np.stack([X, Y, Z], axis=-1)  # (N,3)

        # XYZ -> linear RGB
        M_xyz2rgb = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ], dtype=np.float32)
        rgb_lin = xyz @ M_xyz2rgb.T  # (N,3)

        # linear RGB -> sRGB
        def linear_to_srgb(v):
            v = np.clip(v, 0.0, None)  # no negative radiances
            return np.where(
                v <= 0.0031308,
                12.92 * v,
                1.055 * np.power(v, 1.0 / 2.4) - 0.055,
            )

        rgb = linear_to_srgb(rgb_lin)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        return rgb.reshape(orig_shape)
