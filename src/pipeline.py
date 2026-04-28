# src/pipeline.py
# Full enhancement pipeline (single-image):
#   S -> Ic -> I1..I4 -> weights -> pyramid fuse -> tone boost -> Retinex -> postprocess

from __future__ import annotations
import numpy as np

from preprocess import compute_coarse_illumination_Ic
from illumination import compute_illumination_instances
from weights import compute_weights_for_instances
from pyramids import fuse_multiscale
from tone import tone_boost_illumination
from retinex import retinex_enhance
from postprocess import postprocess_daylook


def enhance_one_image(S_rgb01: np.ndarray,
                      # ---------- Preprocess params ----------
                      wb_strength: float = 0.9,
                      alpha_blend: float = 0.6,
                      Ic_bilateral_d: int = 9,
                      Ic_sigma_color: float = 0.10,
                      Ic_sigma_space: float = 12.0,
                      # ---------- Illumination instances params ----------
                      wls_lam: float = 0.8,
                      wls_alpha: float = 1.2,
                      wls_eps: float = 1e-4,
                      I2_method: str = "bilateral",  # "guided" if available
                      arctan_k: float = 10.0,
                      gauss_sigmas: tuple[float, ...] = (2.0, 6.0, 12.0),
                      gauss_weights: tuple[float, ...] = (0.5, 0.3, 0.2),
                      apply_adaptive_gamma_each: bool = True,
                      # ---------- Weights params ----------
                      saliency_sigma: float = 3.0,
                      wL: float = 0.7,
                      wC: float = 1.5,
                      wS: float = 1.2,
                      # ---------- Pyramid fusion params ----------
                      pyramid_levels: int = 5,
                      pyramid_sigma: float = 1.0,
                      # ---------- Tone boost params ----------
                      tone_target_mean: float = 0.55,
                      tone_gamma_min: float = 0.55,
                      tone_gamma_max: float = 1.20,
                      clahe_clip: float = 2.2,
                      clahe_tile: int = 8,
                      clahe_mix: float = 0.55,
                      # ---------- Retinex recombination ----------
                      retinex_eps: float = 1e-4,
                      # ---------- Postprocess params ----------
                      highlight_thr: float = 0.92,
                      highlight_strength: float = 0.60,
                      sat_mult: float = 1.08,
                      denoise_method: str = "nlmeans",
                      denoise_h: float = 6.0,
                      sharp_sigma: float = 1.0,
                      sharp_amount: float = 0.55,
                      ) -> tuple[np.ndarray, dict]:
    """
    Input:
      S_rgb01: RGB float32 image in [0,1], shape (H,W,3)

    Output:
      S_final: enhanced RGB float32 in [0,1]
      info: dict of useful debug scalars/intermediates (optional)
    """
    info: dict = {}

    # -----------------------------
    # 1) Coarse illumination Ic
    # -----------------------------
    Ic, info_Ic = compute_coarse_illumination_Ic(
        S_rgb01,
        wb_strength=wb_strength,
        alpha=alpha_blend,
        bilateral_d=Ic_bilateral_d,
        bilateral_sigma_color=Ic_sigma_color,
        bilateral_sigma_space=Ic_sigma_space,
    )
    info["Ic"] = Ic
    info["Ic_stats"] = {
        "mean": float(np.mean(Ic)),
        "min": float(np.min(Ic)),
        "max": float(np.max(Ic)),
        "mean_L": info_Ic.get("mean_L"),
    }

    # -----------------------------
    # 2) Illumination instances I1..I4
    # -----------------------------
    (I1, I2, I3, I4), info_I = compute_illumination_instances(
        Ic,
        wls_lam=wls_lam,
        wls_alpha=wls_alpha,
        wls_eps=wls_eps,
        I2_method=I2_method,
        arctan_k=arctan_k,
        gauss_sigmas=gauss_sigmas,
        gauss_weights=gauss_weights,
        apply_adaptive_gamma=apply_adaptive_gamma_each
    )
    info["gammas_each"] = {
        "I1": info_I.get("gamma_I1"),
        "I2": info_I.get("gamma_I2"),
        "I3": info_I.get("gamma_I3"),
        "I4": info_I.get("gamma_I4"),
    }

    # -----------------------------
    # 3) Compute weights and normalize
    # -----------------------------
    Wn, info_W = compute_weights_for_instances(
        [I1, I2, I3, I4],
        saliency_sigma=saliency_sigma,
        wL=wL, wC=wC, wS=wS
    )
    info["weights_sum_check"] = {
        "mean_sum": info_W.get("mean_sum_Wn"),
        "min_sum": info_W.get("min_sum_Wn"),
        "max_sum": info_W.get("max_sum_Wn"),
    }

    # -----------------------------
    # 4) Multiscale pyramid fusion -> I_fused
    # -----------------------------
    I_fused = fuse_multiscale([I1, I2, I3, I4], Wn, levels=pyramid_levels, sigma=pyramid_sigma)
    info["I_fused_stats"] = {
        "mean": float(np.mean(I_fused)),
        "min": float(np.min(I_fused)),
        "max": float(np.max(I_fused)),
    }

    # -----------------------------
    # 5) Tone boost fused illumination -> I_final
    # -----------------------------
    I_final, tone_info = tone_boost_illumination(
        I_fused,
        target_mean=tone_target_mean,
        gamma_min=tone_gamma_min,
        gamma_max=tone_gamma_max,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
        mix_clahe=clahe_mix
    )
    info["tone_info"] = tone_info
    info["I_final"] = I_final

    # -----------------------------
    # 6) Retinex recombination
    # Use Ic as the "original illumination" reference
    # -----------------------------
    # 6) Retinex recombination (use white-balanced image consistently)
    S_wb = info_Ic["S_wb"]
    S_enh = retinex_enhance(S_wb, I_orig_01=Ic, I_enh_01=I_final, eps=retinex_eps)
    info["S_enh_mean"] = float(np.mean(S_enh))

    # -----------------------------
    # 7) Postprocess for day look
    # -----------------------------
    S_final, post_info = postprocess_daylook(
        S_enh,
        highlight_thr=highlight_thr,
        highlight_strength=highlight_strength,
        sat_mult=sat_mult,
        denoise_method=denoise_method,
        denoise_h=denoise_h,
        sharp_sigma=sharp_sigma,
        sharp_amount=sharp_amount,
    )
    info["post_info"] = post_info
    info["S_final_mean"] = float(np.mean(S_final))

    return S_final.astype(np.float32), info