# src/retinex.py
# Retinex recombination formulas:
#   S = R * I
#   R = S / (I + eps)
#   S_enh = R * I_enh

import numpy as np


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def retinex_decompose(S_rgb01: np.ndarray, I_01: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Reflectance (per-channel):
      R = S / (I + eps)
    Where I is a single-channel illumination broadcast across RGB.

    Returns R in [0,1] (clipped).
    """
    S = S_rgb01.astype(np.float32)
    I = I_01.astype(np.float32)
    I3 = I[..., None]  # broadcast to (H,W,1)

    R = S / (I3 + float(eps))
    return clamp01(R)


def retinex_recombine(R_rgb01: np.ndarray, I_enh_01: np.ndarray) -> np.ndarray:
    """
    Recombine:
      S_enh = R * I_enh
    """
    R = R_rgb01.astype(np.float32)
    I = I_enh_01.astype(np.float32)
    I3 = I[..., None]

    S_enh = R * I3
    return clamp01(S_enh)


def retinex_enhance(S_rgb01: np.ndarray,
                    I_orig_01: np.ndarray,
                    I_enh_01: np.ndarray,
                    eps: float = 1e-4) -> np.ndarray:
    """
    Full Retinex enhancement:
      R = S / (I_orig + eps)
      S_enh = R * I_enh
    """
    R = retinex_decompose(S_rgb01, I_orig_01, eps=eps)
    out = retinex_recombine(R, I_enh_01)
    return out