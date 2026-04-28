# src/weights.py
# Weight maps for illumination fusion:
#   - Luminance weight
#   - Contrast weight (Laplacian magnitude) via vectorized convolution
#   - Saliency weight (blur - mean) via vectorized Gaussian separable convolution
# Then combine and normalize weights across all illumination instances.

import numpy as np
from utils import clamp01

try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception as e:
    sliding_window_view = None


# ============================================================
# 1) 2D convolution (vectorized, no OpenCV)
# ============================================================

def conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2D convolution with 'same' output size using edge padding.
    img: (H,W)
    kernel: (kh,kw)
    Returns: (H,W)
    """
    if sliding_window_view is None:
        raise ImportError(
            "NumPy sliding_window_view not available. "
            "Upgrade NumPy (>=1.20) for vectorized convolution."
        )

    img = img.astype(np.float32, copy=False)
    k = kernel.astype(np.float32, copy=False)

    kh, kw = k.shape
    pad_y = kh // 2
    pad_x = kw // 2

    # edge padding
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")

    # flip kernel for convolution
    kflip = np.flipud(np.fliplr(k)).astype(np.float32, copy=False)

    # windows: (H, W, kh, kw)
    windows = sliding_window_view(padded, (kh, kw))

    # out[y,x] = sum_{i,j} windows[y,x,i,j] * kflip[i,j]
    out = np.tensordot(windows, kflip, axes=([2, 3], [0, 1]))
    return out.astype(np.float32, copy=False)


# ============================================================
# 2) Gaussian blur (vectorized, separable)
# ============================================================

def gaussian_1d_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    """
    Create a 1D Gaussian kernel:
      G(x) = exp(-(x^2)/(2*sigma^2)) / sum(...)
    radius defaults to ceil(3*sigma).
    Returns shape (K,)
    """
    s = float(sigma)
    if s <= 0:
        return np.array([1.0], dtype=np.float32)

    if radius is None:
        radius = int(np.ceil(3.0 * s))
    radius = max(1, int(radius))

    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * s * s)).astype(np.float32)
    k /= (np.sum(k) + 1e-12)
    return k


def conv1d_horiz_same(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """
    Horizontal 1D convolution with 'same' size using edge padding.
    img: (H,W)
    k1d: (K,)
    """
    if sliding_window_view is None:
        raise ImportError(
            "NumPy sliding_window_view not available. "
            "Upgrade NumPy (>=1.20) for vectorized convolution."
        )

    img = img.astype(np.float32, copy=False)
    k = k1d.astype(np.float32, copy=False)
    r = k.size // 2

    padded = np.pad(img, ((0, 0), (r, r)), mode="edge")
    kflip = k[::-1].astype(np.float32, copy=False)

    # windows: (H, W, K) along axis=1
    windows = sliding_window_view(padded, window_shape=k.size, axis=1)

    # out[y,x] = sum_{t} windows[y,x,t] * kflip[t]
    out = np.tensordot(windows, kflip, axes=([2], [0]))
    return out.astype(np.float32, copy=False)


def conv1d_vert_same(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """
    Vertical 1D convolution with 'same' size using edge padding.
    img: (H,W)
    k1d: (K,)
    """
    if sliding_window_view is None:
        raise ImportError(
            "NumPy sliding_window_view not available. "
            "Upgrade NumPy (>=1.20) for vectorized convolution."
        )

    img = img.astype(np.float32, copy=False)
    k = k1d.astype(np.float32, copy=False)
    r = k.size // 2

    padded = np.pad(img, ((r, r), (0, 0)), mode="edge")
    kflip = k[::-1].astype(np.float32, copy=False)

    # windows: (H, W, K) along axis=0
    windows = sliding_window_view(padded, window_shape=k.size, axis=0)

    out = np.tensordot(windows, kflip, axes=([2], [0]))
    return out.astype(np.float32, copy=False)


def gaussian_blur_same(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Separable Gaussian blur:
      blur(img) = conv_y(conv_x(img, G), G)
    """
    k = gaussian_1d_kernel(sigma)
    tmp = conv1d_horiz_same(img, k)
    out = conv1d_vert_same(tmp, k)
    return out.astype(np.float32, copy=False)


# ============================================================
# 3) Weight components (FORMULAS)
# ============================================================

def luminance_weight(I: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    W_L = I + eps
    """
    return (I.astype(np.float32, copy=False) + float(eps)).astype(np.float32, copy=False)


def contrast_weight_laplacian(I: np.ndarray) -> np.ndarray:
    """
    Contrast weight:
      W_C = |Laplacian(I)|

    Using discrete Laplacian kernel:
      [ 0  1  0
        1 -4  1
        0  1  0 ]
    """
    I = I.astype(np.float32, copy=False)
    lap_kernel = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=np.float32)
    lap = conv2d_same(I, lap_kernel)
    return np.abs(lap).astype(np.float32, copy=False)


def saliency_weight_blur_minus_mean(I: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Simple saliency:
      blur = GaussianBlur(I, sigma)
      mean = mean(I)
      W_S = |blur - mean|
    """
    I = I.astype(np.float32, copy=False)
    blur = gaussian_blur_same(I, sigma=float(sigma))
    m = float(np.mean(I))
    W = np.abs(blur - m).astype(np.float32, copy=False)
    return W


# ============================================================
# 4) Combine + normalize (FORMULA)
# ============================================================

def combine_weights(WL: np.ndarray, WC: np.ndarray, WS: np.ndarray,
                    wL: float = 1.0, wC: float = 1.0, wS: float = 1.0,
                    eps: float = 1e-12) -> np.ndarray:
    """
    W = wL*WL + wC*WC + wS*WS + eps
    """
    W = (float(wL) * WL.astype(np.float32, copy=False) +
         float(wC) * WC.astype(np.float32, copy=False) +
         float(wS) * WS.astype(np.float32, copy=False) +
         float(eps))
    return W.astype(np.float32, copy=False)


def normalize_weights(Ws: list[np.ndarray], eps: float = 1e-12) -> list[np.ndarray]:
    """
    Given list [W1,W2,W3,W4], normalize per-pixel so sum=1:
      Wn_i = W_i / (sum_j W_j + eps)
    """
    if len(Ws) == 0:
        raise ValueError("Ws must be non-empty")

    stack = np.stack([w.astype(np.float32, copy=False) for w in Ws], axis=0)  # (K,H,W)
    denom = np.sum(stack, axis=0) + float(eps)  # (H,W)

    # vectorized normalization
    norm = stack / denom[None, :, :]
    return [norm[i].astype(np.float32, copy=False) for i in range(norm.shape[0])]


# ============================================================
# 5) Main API: compute weights for I1..I4
# ============================================================

def compute_weights_for_instances(I_list: list[np.ndarray],
                                  saliency_sigma: float = 3.0,
                                  wL: float = 1.0, wC: float = 1.0, wS: float = 1.0) -> tuple[list[np.ndarray], dict]:
    """
    Input: list of illuminations [I1,I2,I3,I4] each (H,W) float in [0,1]
    Output:
      Wn_list: [W1n,W2n,W3n,W4n] normalized weights
      info: optional debug scalars
    """
    if len(I_list) != 4:
        raise ValueError("Expected exactly 4 illumination instances [I1,I2,I3,I4]")

    Ws = []
    info = {}

    for idx, I in enumerate(I_list, 1):
        I0 = clamp01(I.astype(np.float32, copy=False))

        WL = luminance_weight(I0)
        WC = contrast_weight_laplacian(I0)
        WS = saliency_weight_blur_minus_mean(I0, sigma=float(saliency_sigma))

        W = combine_weights(WL, WC, WS, wL=wL, wC=wC, wS=wS)
        Ws.append(W)

        info[f"I{idx}_mean"] = float(np.mean(I0))
        info[f"W{idx}_mean"] = float(np.mean(W))

    Wn = normalize_weights(Ws)

    sumW = np.sum(np.stack(Wn, axis=0), axis=0)
    info["mean_sum_Wn"] = float(np.mean(sumW))
    info["min_sum_Wn"] = float(np.min(sumW))
    info["max_sum_Wn"] = float(np.max(sumW))

    return Wn, info