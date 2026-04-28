# src/illumination.py
# Generate multiple illumination instances I1..I4 from coarse illumination Ic.
# Includes a REAL Weighted Least Squares (WLS) smoothing implementation
# based on minimizing an energy function and solving a sparse linear system.

import numpy as np
import cv2
from utils import clamp01


# -----------------------------
# Helper: gradients (finite differences) - vectorized
# -----------------------------
def _grad_x(img: np.ndarray) -> np.ndarray:
    # forward difference in x direction (columns)
    img = img.astype(np.float32, copy=False)
    gx = np.zeros_like(img, dtype=np.float32)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    # gx[:, -1] already 0
    return gx

def _grad_y(img: np.ndarray) -> np.ndarray:
    # forward difference in y direction (rows)
    img = img.astype(np.float32, copy=False)
    gy = np.zeros_like(img, dtype=np.float32)
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    # gy[-1, :] already 0
    return gy


# ============================================================
# I1: WLS smoothing (REAL FORMULA)
# ============================================================

def compute_I1_wls(Ic: np.ndarray,
                   lam: float = 0.8,
                   alpha: float = 1.2,
                   eps: float = 1e-4,
                   solver: str = "cg",
                   max_iter: int = 200,
                   tol: float = 1e-6) -> np.ndarray:
    """
    Weighted Least Squares edge-preserving smoothing of Ic.
    (Same math as your version; optimized fallback loop.)
    """
    g = Ic.astype(np.float32, copy=False)
    H, W = g.shape
    N = H * W

    gx = _grad_x(g)
    gy = _grad_y(g)

    wx = 1.0 / (np.power(np.abs(gx), float(alpha)) + float(eps))  # (H,W)
    wy = 1.0 / (np.power(np.abs(gy), float(alpha)) + float(eps))  # (H,W)

    # Try SciPy sparse solve (fastest)
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        use_scipy = True
    except Exception:
        use_scipy = False

    b = g.reshape(-1).astype(np.float32, copy=False)

    if use_scipy:
        wx_right = wx.copy()
        wx_right[:, -1] = 0.0

        wy_down = wy.copy()
        wy_down[-1, :] = 0.0

        wx_left = np.zeros_like(wx_right)
        wx_left[:, 1:] = wx_right[:, :-1]

        wy_up = np.zeros_like(wy_down)
        wy_up[1:, :] = wy_down[:-1, :]

        diag = (1.0 + float(lam) * (wx_left + wx_right + wy_up + wy_down)).reshape(-1)

        off_right = (-float(lam) * wx_right).reshape(-1)
        off_left  = (-float(lam) * wx_left).reshape(-1)
        off_down  = (-float(lam) * wy_down).reshape(-1)
        off_up    = (-float(lam) * wy_up).reshape(-1)

        A = sp.diags(
            diagonals=[diag, off_right[:-1], off_left[1:], off_down[:-W], off_up[W:]],
            offsets=[0, 1, -1, W, -W],
            shape=(N, N),
            format="csr"
        )

        if solver.lower() == "cg":
            x, info = spla.cg(A, b, maxiter=int(max_iter), rtol=float(tol))
            if info != 0:
                x = spla.spsolve(A, b)
        else:
            x = spla.spsolve(A, b)

        I = x.reshape(H, W).astype(np.float32, copy=False)
        return clamp01(I)

    # -----------------------
    # Fallback: Jacobi (no scipy) - optimized
    # -----------------------
    I = g.copy()

    wx_right = wx.copy(); wx_right[:, -1] = 0.0
    wy_down  = wy.copy(); wy_down[-1, :] = 0.0
    wx_left  = np.zeros_like(wx_right); wx_left[:, 1:] = wx_right[:, :-1]
    wy_up    = np.zeros_like(wy_down);  wy_up[1:, :]   = wy_down[:-1, :]

    lamf = float(lam)
    denom = 1.0 + lamf * (wx_left + wx_right + wy_up + wy_down)
    denom = denom.astype(np.float32, copy=False)

    # Pre-allocate neighbor buffers (reduces per-iter allocations)
    I_right = np.empty_like(I)
    I_left  = np.empty_like(I)
    I_down  = np.empty_like(I)
    I_up    = np.empty_like(I)

    for _ in range(int(max_iter)):
        I_old = I  # old array reference

        # neighbors via roll (then zero-out wrapped boundary to match your old zeros)
        np.roll(I_old, -1, axis=1, out=I_right)
        I_right[:, -1] = 0.0

        np.roll(I_old,  1, axis=1, out=I_left)
        I_left[:, 0] = 0.0

        np.roll(I_old, -1, axis=0, out=I_down)
        I_down[-1, :] = 0.0

        np.roll(I_old,  1, axis=0, out=I_up)
        I_up[0, :] = 0.0

        num = (g + lamf * (
            wx_right * I_right +
            wx_left  * I_left  +
            wy_down  * I_down  +
            wy_up    * I_up
        ))

        I = num / (denom + 1e-12)

        diff = float(np.mean(np.abs(I - I_old)))
        if diff < float(tol):
            break

    return clamp01(I)


# ============================================================
# I2: Edge-aware smoothing (bilateral / guided optional)
# ============================================================

def compute_I2_edge_aware(Ic: np.ndarray,
                          method: str = "bilateral",
                          bilateral_d: int = 9,
                          bilateral_sigma_color: float = 0.05,
                          bilateral_sigma_space: float = 25.0,
                          guided_radius: int = 16,
                          guided_eps: float = 1e-3) -> np.ndarray:
    """
    Edge-aware smoothing of Ic to produce I2.
    Uses bilateral filter by default.
    """
    I = Ic.astype(np.float32, copy=False)

    if method == "guided":
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            out = cv2.ximgproc.guidedFilter(guide=I, src=I,
                                            radius=int(guided_radius),
                                            eps=float(guided_eps))
            return clamp01(out.astype(np.float32))
        method = "bilateral"

    out = cv2.bilateralFilter(I,
                              d=int(bilateral_d),
                              sigmaColor=float(bilateral_sigma_color),
                              sigmaSpace=float(bilateral_sigma_space))
    return clamp01(out.astype(np.float32))


# ============================================================
# I3: Arctan illumination enhancement (REAL FORMULA)
# ============================================================

def compute_I3_arctan(Ic: np.ndarray, k: float = 10.0) -> np.ndarray:
    """
    I3 = arctan(k*Ic) / arctan(k)
    """
    I = Ic.astype(np.float32, copy=False)
    denom = np.arctan(float(k)) + 1e-12
    out = np.arctan(float(k) * I) / denom
    return clamp01(out.astype(np.float32))


# ============================================================
# I4: Multi-scale Gaussian illumination (REAL FORMULA)
# ============================================================

def compute_I4_multiscale_gaussian(Ic: np.ndarray,
                                   sigmas: tuple = (1.0, 2.0, 4.0),
                                   weights: tuple = (0.5, 0.3, 0.2)) -> np.ndarray:
    """
    I4 = Σ_i w_i * GaussianBlur(Ic, sigma_i)
    """
    I = Ic.astype(np.float32, copy=False)
    if len(sigmas) != len(weights):
        raise ValueError("sigmas and weights must have same length")

    wsum = float(np.sum(weights))
    if wsum <= 0:
        raise ValueError("weights must sum to > 0")

    out = np.zeros_like(I, dtype=np.float32)
    for s, w in zip(sigmas, weights):
        sigma = float(s)
        k = int(2 * np.ceil(3 * sigma) + 1)
        k = max(3, k)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(I, (k, k), sigmaX=sigma, sigmaY=sigma)
        out += float(w) * blur.astype(np.float32, copy=False)

    out /= wsum
    return clamp01(out)


# ============================================================
# Optional: Adaptive gamma (REAL FORMULA)
# ============================================================

def adaptive_gamma(I: np.ndarray,
                   target_mean: float = 0.55,
                   gamma_min: float = 0.55,
                   gamma_max: float = 1.20,
                   eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """
    ratio = mean(I)/target_mean
    gamma = clip(ratio, gamma_min, gamma_max)
    I_g = I^gamma
    """
    I0 = clamp01(I.astype(np.float32, copy=False))
    m = float(np.mean(I0))
    ratio = (m + float(eps)) / (float(target_mean) + float(eps))
    gamma = float(np.clip(ratio, float(gamma_min), float(gamma_max)))
    Ig = np.power(I0, gamma).astype(np.float32, copy=False)
    return clamp01(Ig), gamma


# ============================================================
# One function to compute all instances from Ic
# ============================================================

def compute_illumination_instances(Ic: np.ndarray,
                                   wls_lam: float = 0.8,
                                   wls_alpha: float = 1.2,
                                   wls_eps: float = 1e-4,
                                   I2_method: str = "bilateral",
                                   arctan_k: float = 10.0,
                                   gauss_sigmas: tuple = (2.0, 6.0, 12.0),
                                   gauss_weights: tuple = (0.5, 0.3, 0.2),
                                   apply_adaptive_gamma: bool = True) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict]:
    """
    From Ic -> compute I1..I4 (optionally adaptive gamma each).
    Returns ((I1,I2,I3,I4), info_dict)
    """
    info = {}

    I1 = compute_I1_wls(Ic, lam=wls_lam, alpha=wls_alpha, eps=wls_eps)
    I2 = compute_I2_edge_aware(Ic, method=I2_method)
    I3 = compute_I3_arctan(Ic, k=arctan_k)
    I4 = compute_I4_multiscale_gaussian(Ic, sigmas=gauss_sigmas, weights=gauss_weights)

    info["I1_mean_before"] = float(np.mean(I1))
    info["I2_mean_before"] = float(np.mean(I2))
    info["I3_mean_before"] = float(np.mean(I3))
    info["I4_mean_before"] = float(np.mean(I4))

    if apply_adaptive_gamma:
        I1, g1 = adaptive_gamma(I1)
        I2, g2 = adaptive_gamma(I2)
        I3, g3 = adaptive_gamma(I3)
        I4, g4 = adaptive_gamma(I4)
        info["gamma_I1"] = g1
        info["gamma_I2"] = g2
        info["gamma_I3"] = g3
        info["gamma_I4"] = g4

    return (I1, I2, I3, I4), info