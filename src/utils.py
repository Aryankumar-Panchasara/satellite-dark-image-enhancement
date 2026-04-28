# src/utils.py
import os
import cv2
import numpy as np

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def imread_rgb_float01(path: str) -> np.ndarray:
    """
    Read image using OpenCV (BGR) -> RGB float32 in [0,1].
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

def imwrite_rgb_uint8(path: str, rgb01: np.ndarray) -> None:
    """
    Save RGB float [0,1] image using OpenCV.
    """
    rgb01 = clamp01(rgb01)
    rgb8 = (rgb01 * 255.0 + 0.5).astype(np.uint8)
    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, bgr8)