# src/preprocess.py
import cv2
import numpy as np
from utils import clamp01

# ============================================================
# STEP B: Gray-world white balance (FORMULA IN CODE)
# ============================================================

def gray_world_white_balance(rgb01: np.ndarray, strength: float = 0.9, eps: float = 1e-6) -> np.ndarray:
    """
    Gray-world white balance:
      mean_rgb = [meanR, meanG, meanB]
      mean_gray = mean(mean_rgb)
      scale = mean_gray / (mean_rgb + eps)
      scale_blend = 1 + strength*(scale - 1)
      out = rgb * scale_blend

    strength: 0 (no WB) -> 1 (full WB)
    """
    rgb = rgb01.astype(np.float32)

    mean_rgb = np.mean(rgb, axis=(0, 1))            # [meanR, meanG, meanB]
    mean_gray = float(np.mean(mean_rgb))           # scalar

    scale = mean_gray / (mean_rgb + eps)           # per-channel
    scale = 1.0 + float(strength) * (scale - 1.0)  # blended correction

    out = rgb * scale[None, None, :]
    return clamp01(out)

# ============================================================
# STEP C: Luminance (Rec.709 FORMULA IN CODE)
# ============================================================

def luminance_from_rgb(rgb01: np.ndarray) -> np.ndarray:
    """
    Rec.709 luminance:
      L = 0.2126*R + 0.7152*G + 0.0722*B
    """
    rgb = rgb01.astype(np.float32)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return clamp01(L)

# ============================================================
# STEP D: Max-channel illumination (FORMULA IN CODE)
# ============================================================

def max_channel_illumination(rgb01: np.ndarray) -> np.ndarray:
    """
    Ic_max(x,y) = max(R,G,B)
    """
    rgb = rgb01.astype(np.float32)
    Ic_max = np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    return clamp01(Ic_max)

# ============================================================
# STEP E: Blend max-channel + luminance (FORMULA IN CODE)
# ============================================================

def blend_illumination(Ic_max: np.ndarray, L: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    I_base = alpha*Ic_max + (1-alpha)*L
    """
    a = float(alpha)
    I_base = a * Ic_max + (1.0 - a) * L
    return clamp01(I_base)

# ============================================================
# STEP F: Edge-aware smoothing (bilateral)
# (not a formula step; OK to use OpenCV filter)
# ============================================================

def edge_aware_smooth_bilateral(I_base: np.ndarray,
                               d: int = 9,
                               sigma_color: float = 0.10,
                               sigma_space: float = 15.0) -> np.ndarray:
    """
    Edge-aware illumination smoothing using bilateral filter.
    Input/Output in [0,1].
    """
    I32 = I_base.astype(np.float32)
    I_smooth = cv2.bilateralFilter(I32, d=int(d),
                                   sigmaColor=float(sigma_color),
                                   sigmaSpace=float(sigma_space))
    return clamp01(I_smooth)

# ============================================================
# Combined "first part" function: S -> Ic
# ============================================================

def compute_coarse_illumination_Ic(rgb01: np.ndarray,
                                  wb_strength: float = 0.9,
                                  alpha: float = 0.6,
                                  bilateral_d: int = 9,
                                  bilateral_sigma_color: float = 0.10,
                                  bilateral_sigma_space: float = 15.0) -> tuple[np.ndarray, dict]:
    """
    Runs the full FIRST PART:
      1) WB
      2) Luminance
      3) Max-channel Ic_max
      4) Blend I_base
      5) Bilateral edge-aware smoothing -> Ic

    Returns:
      Ic (H,W) in [0,1]
      info dict with intermediates for debugging
    """
    info = {}

    S_wb = gray_world_white_balance(rgb01, strength=wb_strength)
    info["S_wb"] = S_wb

    L = luminance_from_rgb(S_wb)
    info["L"] = L

    Ic_max = max_channel_illumination(S_wb)
    info["Ic_max"] = Ic_max

    I_base = blend_illumination(Ic_max, L, alpha=alpha)
    info["I_base"] = I_base

    Ic = edge_aware_smooth_bilateral(I_base,
                                 d=bilateral_d,
                                 sigma_color=bilateral_sigma_color,
                                 sigma_space=bilateral_sigma_space)

# CRITICAL: prevent zeros in illumination (avoids Retinex explosion)
    Ic = np.clip(Ic, 0.05, 1.0)

    info["Ic"] = Ic

    # some debug scalars
    info["mean_L"] = float(np.mean(L))
    info["mean_Ic_max"] = float(np.mean(Ic_max))
    info["mean_Ic"] = float(np.mean(Ic))

    return Ic, info