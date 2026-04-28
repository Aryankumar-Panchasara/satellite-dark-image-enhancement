# src/tone.py
# Tone boost on fused illumination:
#   - adaptive gamma (formula)
#   - CLAHE (OpenCV primitive)
#   - mix CLAHE + gamma to avoid over-contrast

import numpy as np
import cv2


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ============================================================
# Adaptive Gamma (FORMULA IN CODE)
# ============================================================
def adaptive_gamma(I: np.ndarray,
                   target_mean: float = 0.55,
                   gamma_min: float = 0.55,
                   gamma_max: float = 1.20,
                   eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """
    Let m = mean(I). We choose gamma based on how dark the illumination is.

      ratio = (m + eps) / (target_mean + eps)
      gamma = clip(ratio, gamma_min, gamma_max)
      I_g = I ^ gamma

    Returns: (I_g, gamma_used)
    """
    I0 = clamp01(I.astype(np.float32))
    m = float(np.mean(I0))
    ratio = (m + eps) / (float(target_mean) + eps)
    gamma = float(np.clip(ratio, float(gamma_min), float(gamma_max)))

    I_g = np.power(I0, gamma).astype(np.float32)
    return clamp01(I_g), gamma


# ============================================================
# CLAHE on Illumination (uses OpenCV primitive)
# ============================================================
def clahe_on_illumination(I: np.ndarray,
                          clip_limit: float = 2.2,
                          tile_grid_size: int = 8) -> np.ndarray:
    """
    CLAHE expects uint8. We apply it on the illumination map.
    """
    I0 = clamp01(I)
    I8 = (I0 * 255.0 + 0.5).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    out8 = clahe.apply(I8)
    out = out8.astype(np.float32) / 255.0
    return clamp01(out)


# ============================================================
# Main tone boost for fused illumination
# ============================================================
def tone_boost_illumination(I_fused: np.ndarray,
                            target_mean: float = 0.55,
                            gamma_min: float = 0.55,
                            gamma_max: float = 1.20,
                            clahe_clip: float = 2.2,
                            clahe_tile: int = 8,
                            mix_clahe: float = 0.55) -> tuple[np.ndarray, dict]:
    """
    Steps:
      1) I_g = adaptive_gamma(I_fused)
      2) I_c = CLAHE(I_g)
      3) I_final = mix_clahe * I_c + (1-mix_clahe) * I_g

    mix_clahe controls local contrast strength.
    """
    info = {}

    I_g, gamma_used = adaptive_gamma(I_fused,
                                     target_mean=target_mean,
                                     gamma_min=gamma_min,
                                     gamma_max=gamma_max)
    info["gamma_used"] = gamma_used
    info["mean_before"] = float(np.mean(clamp01(I_fused)))
    info["mean_after_gamma"] = float(np.mean(I_g))

    I_c = clahe_on_illumination(I_g, clip_limit=clahe_clip, tile_grid_size=clahe_tile)
    info["mean_after_clahe"] = float(np.mean(I_c))

    a = float(mix_clahe)
    I_final = a * I_c + (1.0 - a) * I_g
    I_final = clamp01(I_final)
    info["mean_final"] = float(np.mean(I_final))

    return I_final, info