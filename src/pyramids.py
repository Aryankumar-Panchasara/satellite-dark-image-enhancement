# src/pyramids.py
# Multiscale pyramid fusion (Gaussian/Laplacian pyramids) with formula-based ops:
# - Gaussian blur: vectorized separable convolution (sliding_window_view + tensordot)
# - Downsample: vectorized 2x2 average
# - Upsample: vectorized bilinear interpolation
# - Laplacian pyramid construction + reconstruction
# - Fusion using weight Gaussian pyramids + illumination Laplacian pyramids

import numpy as np

try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception:
    sliding_window_view = None


# ============================================================
# 1) Basic helpers
# ============================================================

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def gaussian_1d_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    """
    1D Gaussian kernel:
      G(x) = exp(-(x^2)/(2*sigma^2)) / sum(...)
    radius default = ceil(3*sigma)
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


# ============================================================
# 2) Vectorized 1D conv (same) + Gaussian blur
# ============================================================

def conv1d_horiz_same(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """
    Horizontal 1D convolution with edge padding (same size).
    img: (H,W)
    """
    if sliding_window_view is None:
        raise ImportError("NumPy sliding_window_view not available. Upgrade NumPy (>=1.20).")

    img = img.astype(np.float32, copy=False)
    k = k1d.astype(np.float32, copy=False)
    r = k.size // 2

    padded = np.pad(img, ((0, 0), (r, r)), mode="edge")
    kflip = k[::-1].astype(np.float32, copy=False)

    # windows: (H, W, K) along axis=1
    windows = sliding_window_view(padded, window_shape=k.size, axis=1)
    out = np.tensordot(windows, kflip, axes=([2], [0]))
    return out.astype(np.float32, copy=False)

def conv1d_vert_same(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """
    Vertical 1D convolution with edge padding (same size).
    img: (H,W)
    """
    if sliding_window_view is None:
        raise ImportError("NumPy sliding_window_view not available. Upgrade NumPy (>=1.20).")

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
      blur = conv_y(conv_x(img, G), G)
    """
    k = gaussian_1d_kernel(float(sigma))
    tmp = conv1d_horiz_same(img, k)
    out = conv1d_vert_same(tmp, k)
    return out.astype(np.float32, copy=False)


# ============================================================
# 3) Downsample / Upsample (vectorized)
# ============================================================

def downsample_2x(img: np.ndarray) -> np.ndarray:
    """
    2x downsample using 2x2 average:
      out[y,x] = mean(img[2y:2y+2, 2x:2x+2])
    Handles odd sizes by dropping last row/col.
    """
    img = img.astype(np.float32, copy=False)
    H, W = img.shape
    H2 = H // 2
    W2 = W // 2

    # drop last row/col if odd
    cropped = img[: 2 * H2, : 2 * W2]

    # reshape into (H2,2,W2,2) then mean over the 2x2 blocks
    out = cropped.reshape(H2, 2, W2, 2).mean(axis=(1, 3))
    return out.astype(np.float32, copy=False)


def upsample_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Bilinear upsample from img(Hin,Win) to (out_h,out_w).

    For each output pixel (y,x):
      sy = y * (Hin-1)/(out_h-1)
      sx = x * (Win-1)/(out_w-1)
    then bilinear interpolate.
    """
    src = img.astype(np.float32, copy=False)
    Hin, Win = src.shape

    # edge cases: preserve your old behavior
    if out_h <= 1 or out_w <= 1:
        return np.full((out_h, out_w), float(src[0, 0]), dtype=np.float32)

    # coordinate grids (vectorized)
    ys = np.linspace(0.0, Hin - 1, out_h, dtype=np.float32)
    xs = np.linspace(0.0, Win - 1, out_w, dtype=np.float32)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)

    y1 = np.minimum(y0 + 1, Hin - 1)
    x1 = np.minimum(x0 + 1, Win - 1)

    dy = (ys - y0.astype(np.float32))[:, None]   # (out_h,1)
    dx = (xs - x0.astype(np.float32))[None, :]   # (1,out_w)

    # gather 4 neighbors via advanced indexing
    # shapes:
    #   v00, v01, v10, v11 -> (out_h, out_w)
    v00 = src[y0[:, None], x0[None, :]]
    v01 = src[y0[:, None], x1[None, :]]
    v10 = src[y1[:, None], x0[None, :]]
    v11 = src[y1[:, None], x1[None, :]]

    top = (1.0 - dx) * v00 + dx * v01
    bot = (1.0 - dx) * v10 + dx * v11
    out = (1.0 - dy) * top + dy * bot

    return out.astype(np.float32, copy=False)


# ============================================================
# 4) Gaussian / Laplacian pyramids (same logic, faster primitives)
# ============================================================

def gaussian_pyramid(img: np.ndarray, levels: int = 5, sigma: float = 1.0) -> list[np.ndarray]:
    """
    Build Gaussian pyramid:
      G0 = img
      G_{l+1} = downsample( blur(G_l) )
    Returns list [G0, G1, ..., G_{levels-1}]
    """
    cur = img.astype(np.float32, copy=False)
    G = [cur]
    for _ in range(1, int(levels)):
        cur = gaussian_blur_same(cur, sigma=float(sigma))
        cur = downsample_2x(cur)
        G.append(cur.astype(np.float32, copy=False))
    return G


def laplacian_pyramid(img: np.ndarray, levels: int = 5, sigma: float = 1.0) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Build Laplacian pyramid:
      Compute Gaussian pyramid G
      For l=0..levels-2:
         L_l = G_l - upsample(G_{l+1}, size(G_l))
      Top level is G_{levels-1}
    """
    G = gaussian_pyramid(img, levels=levels, sigma=sigma)
    L = []
    for l in range(levels - 1):
        Gh, Gw = G[l].shape
        up = upsample_bilinear(G[l + 1], Gh, Gw)
        L.append((G[l] - up).astype(np.float32, copy=False))
    top = G[-1].astype(np.float32, copy=False)
    return L, top


def reconstruct_from_laplacian(L_list: list[np.ndarray], top: np.ndarray) -> np.ndarray:
    """
    Reconstruct image from Laplacian pyramid:
      cur = top
      for l from levels-2 down to 0:
         cur = upsample(cur, size(L_l)) + L_l
    """
    cur = top.astype(np.float32, copy=False)
    for l in range(len(L_list) - 1, -1, -1):
        H, W = L_list[l].shape
        up = upsample_bilinear(cur, H, W)
        cur = (up + L_list[l]).astype(np.float32, copy=False)
    return cur.astype(np.float32, copy=False)


# ============================================================
# 5) Fusion (vectorized per-level using stacking)
# ============================================================

def fuse_multiscale(I_list: list[np.ndarray],
                    Wn_list: list[np.ndarray],
                    levels: int = 5,
                    sigma: float = 1.0) -> np.ndarray:
    """
    Multiscale fusion:
      L_fused[l] = Σ_i ( G(W_i)[l] * L_i[l] )
      top_fused  = Σ_i ( G(W_i)[top] * top_i )
    """
    if len(I_list) != 4 or len(Wn_list) != 4:
        raise ValueError("Expected 4 illuminations and 4 weight maps")

    I_list = [I.astype(np.float32, copy=False) for I in I_list]
    Wn_list = [w.astype(np.float32, copy=False) for w in Wn_list]

    L_pyrs = []
    top_levels = []
    W_pyrs = []

    for i in range(4):
        L_i, top_i = laplacian_pyramid(I_list[i], levels=levels, sigma=sigma)
        G_w = gaussian_pyramid(Wn_list[i], levels=levels, sigma=sigma)
        L_pyrs.append(L_i)
        top_levels.append(top_i)
        W_pyrs.append(G_w)

    # Fuse Laplacian levels
    fused_L = []
    for l in range(levels - 1):
        H, W = L_pyrs[0][l].shape

        # Stack L at level l: (4,H,W)
        L_stack = np.stack([L_pyrs[i][l] for i in range(4)], axis=0).astype(np.float32, copy=False)

        # Stack weights at level l, resizing if off-by-1 due to odd sizes
        W_level = []
        for i in range(4):
            w = W_pyrs[i][l]
            if w.shape != (H, W):
                w = upsample_bilinear(w, H, W)
            W_level.append(w.astype(np.float32, copy=False))
        W_stack = np.stack(W_level, axis=0)

        # Σ_i W_i * L_i  -> (H,W)
        acc = np.sum(W_stack * L_stack, axis=0)
        fused_L.append(acc.astype(np.float32, copy=False))

    # Fuse top level
    Ht, Wt = top_levels[0].shape
    top_stack = np.stack(top_levels, axis=0).astype(np.float32, copy=False)

    W_top = []
    for i in range(4):
        w = W_pyrs[i][-1]
        if w.shape != (Ht, Wt):
            w = upsample_bilinear(w, Ht, Wt)
        W_top.append(w.astype(np.float32, copy=False))
    W_top_stack = np.stack(W_top, axis=0)

    top_fused = np.sum(W_top_stack * top_stack, axis=0).astype(np.float32, copy=False)

    # Reconstruct
    I_fused = reconstruct_from_laplacian(fused_L, top_fused)
    return clamp01(I_fused)