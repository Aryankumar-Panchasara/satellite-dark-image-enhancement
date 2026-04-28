# src/postprocess.py
# Post-processing for "day-like" look:
#  1) Highlight compression (formula-based)
#  2) Saturation adjust (HSV; OpenCV conversion is ok)
#  3) Denoise (OpenCV primitive)
#  4) Unsharp mask (formula-based)

import numpy as np
import cv2


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ============================================================
# 1) Luminance (Rec.709) (FORMULA)
# ============================================================

def luminance_from_rgb(rgb01: np.ndarray) -> np.ndarray:
    rgb = rgb01.astype(np.float32)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return clamp01(L)


# ============================================================
# 2) Highlight compression (FORMULA)
# ============================================================

def highlight_compress(rgb01: np.ndarray,
                       threshold: float = 0.92,
                       strength: float = 0.60) -> np.ndarray:
    """
    Prevent blown highlights after enhancement.

    L = luminance(rgb)
    mask = clip((L - threshold)/(1-threshold), 0,1)

    out = rgb / (1 + strength*mask*(rgb - 1))

    This gently pulls values near 1 down in bright regions.
    """
    img = clamp01(rgb01)
    L = luminance_from_rgb(img)

    t = float(threshold)
    s = float(strength)

    mask = (L - t) / (1.0 - t + 1e-6)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    mask3 = mask[..., None]

    out = img / (1.0 + s * mask3 * (img - 1.0))
    return clamp01(out)


# ============================================================
# 3) Saturation adjustment (HSV) (uses OpenCV conversion)
# ============================================================

def adjust_saturation(rgb01: np.ndarray, sat_mult: float = 1.08) -> np.ndarray:
    """
    Multiply saturation in HSV space.
    sat_mult > 1 increases saturation; < 1 decreases.
    """
    img = clamp01(rgb01)
    rgb8 = (img * 255.0 + 0.5).astype(np.uint8)

    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[..., 1] *= float(sat_mult)
    hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 255.0)

    bgr2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return clamp01(rgb2)


# ============================================================
# 4) Denoise (OpenCV primitive; not a formula step)
# ============================================================

def denoise_color(rgb01: np.ndarray,
                  method: str = "nlmeans",
                  h: float = 6.0) -> np.ndarray:
    """
    Denoise after enhancement.
    - nlmeans: strong for low-light noise
    - bilateral: faster, weaker
    """
    img = clamp01(rgb01)
    rgb8 = (img * 255.0 + 0.5).astype(np.uint8)
    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)

    if method == "bilateral":
        bgr_d = cv2.bilateralFilter(bgr8, d=7, sigmaColor=35, sigmaSpace=35)
    else:
        bgr_d = cv2.fastNlMeansDenoisingColored(
            bgr8, None,
            h=float(h), hColor=float(h),
            templateWindowSize=7,
            searchWindowSize=21
        )

    rgb_d = cv2.cvtColor(bgr_d, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return clamp01(rgb_d)


# ============================================================
# 5) Unsharp mask (FORMULA)
# ============================================================

def gaussian_blur_rgb(rgb01: np.ndarray, sigma: float, radius: int | None = None) -> np.ndarray:
    """
    Gaussian blur helper for unsharp mask.
    Uses OpenCV GaussianBlur (primitive). The sharpening formula itself is explicit.
    """
    img = clamp01(rgb01).astype(np.float32)
    s = float(sigma)

    if radius is None:
        radius = int(np.ceil(3.0 * s))
    radius = max(1, int(radius))
    k = 2 * radius + 1
    if k % 2 == 0:
        k += 1

    blur = cv2.GaussianBlur(img, (k, k), sigmaX=s, sigmaY=s)
    return blur.astype(np.float32)


def unsharp_mask(rgb01: np.ndarray,
                 sigma: float = 1.0,
                 amount: float = 0.55,
                 threshold: float = 0.0) -> np.ndarray:
    """
    Unsharp masking:
      blur = G_sigma(img)
      detail = img - blur
      if threshold>0: detail = detail * (|detail|>threshold)
      sharp = img + amount * detail
    """
    img = clamp01(rgb01).astype(np.float32)
    blur = gaussian_blur_rgb(img, sigma=float(sigma))

    detail = img - blur
    if float(threshold) > 0.0:
        mask = (np.abs(detail) > float(threshold)).astype(np.float32)
        detail = detail * mask

    sharp = img + float(amount) * detail
    return clamp01(sharp)


# ============================================================
# 6) Full postprocess block (one-call)
# ============================================================

def postprocess_daylook(rgb01: np.ndarray,
                        highlight_thr: float = 0.92,
                        highlight_strength: float = 0.60,
                        sat_mult: float = 1.08,
                        denoise_method: str = "nlmeans",
                        denoise_h: float = 6.0,
                        sharp_sigma: float = 1.0,
                        sharp_amount: float = 0.55) -> tuple[np.ndarray, dict]:
    """
    Order:
      1) highlight compress
      2) saturation
      3) denoise
      4) sharpen
    """
    info = {}

    out = highlight_compress(rgb01, threshold=highlight_thr, strength=highlight_strength)
    out = adjust_saturation(out, sat_mult=sat_mult)
    out = denoise_color(out, method=denoise_method, h=denoise_h)
    out = unsharp_mask(out, sigma=sharp_sigma, amount=sharp_amount, threshold=0.0)

    info["mean_out"] = float(np.mean(out))
    return out, info