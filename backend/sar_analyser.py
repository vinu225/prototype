"""
backend/sar_analyser.py
========================
Lightweight Sentinel-1 SAR statistical analyser for the Phase 4 demo mode.

NO VLM is invoked here. This module operates purely on numpy arrays
and returns honest statistics about VH / VV backscatter.

This is explicitly NOT a substitute for the full 12-channel multimodal pipeline.
The warning field in SARAnalysisResult makes this clear in every response.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ChannelStats:
    mean: float
    std:  float
    min:  float
    max:  float
    p25:  float
    p75:  float


@dataclass
class SARAnalysisResult:
    """All statistics are computed from the raw backscatter arrays."""
    success:         bool
    vh_stats:        ChannelStats
    vv_stats:        ChannelStats
    ratio_stats:     ChannelStats      # VH/VV element-wise ratio
    correlation:     float             # Pearson r between VH and VV
    interpretation:  str               # heuristic text (labelled as such)
    processing_time: float             # seconds
    shape:           list              # [H, W]
    warning:         str               # explicit SAR-only disclaimer
    error:           Optional[str] = None


# SAR-only disclaimer appended to every response
_DISCLAIMER = (
    "SAR-only demonstration. Statistics are derived from Sentinel-1 "
    "VH/VV backscatter only. The full multimodal model (requiring 12 channels "
    "including 10 Sentinel-2 optical bands) was NOT invoked. "
    "Interpretations are heuristic rules-of-thumb, not model outputs."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_sar_pair(
    vh: np.ndarray,
    vv: np.ndarray,
) -> SARAnalysisResult:
    """
    Compute statistics for a Sentinel-1 VH/VV pair.

    Parameters
    ----------
    vh : np.ndarray  shape [H, W]  VH polarisation backscatter
    vv : np.ndarray  shape [H, W]  VV polarisation backscatter

    Returns
    -------
    SARAnalysisResult with statistics and heuristic interpretation.
    """
    t0 = time.perf_counter()

    if vh.shape != vv.shape:
        return SARAnalysisResult(
            success=False,
            vh_stats=_zero_stats(),
            vv_stats=_zero_stats(),
            ratio_stats=_zero_stats(),
            correlation=0.0,
            interpretation="",
            processing_time=0.0,
            shape=[],
            warning=_DISCLAIMER,
            error=f"VH/VV shape mismatch: {list(vh.shape)} vs {list(vv.shape)}",
        )

    vh_f = vh.flatten().astype(np.float64)
    vv_f = vv.flatten().astype(np.float64)

    vh_stats   = _compute_stats(vh_f)
    vv_stats   = _compute_stats(vv_f)

    # VH/VV ratio -- avoid division by zero
    safe_vv    = np.where(np.abs(vv_f) > 1e-12, vv_f, np.sign(vv_f + 1e-12) * 1e-12)
    ratio      = np.nan_to_num(vh_f / safe_vv, nan=0.0, posinf=5.0, neginf=0.0)
    ratio_stats= _compute_stats(ratio)

    # Pearson correlation (handle zero-variance safely)
    with np.errstate(all='ignore'):
        corr_mat = np.corrcoef(vh_f, vv_f)
        raw_corr = corr_mat[0, 1] if (corr_mat.ndim == 2 and corr_mat.shape == (2, 2)) else 0.0
        corr = float(raw_corr) if not (np.isnan(raw_corr) or np.isinf(raw_corr)) else 0.0

    interpretation = _interpret(vh_stats, vv_stats, ratio_stats, corr)
    elapsed = round(time.perf_counter() - t0, 4)

    return SARAnalysisResult(
        success=True,
        vh_stats=vh_stats,
        vv_stats=vv_stats,
        ratio_stats=ratio_stats,
        correlation=corr,
        interpretation=interpretation,
        processing_time=elapsed,
        shape=list(vh.shape),
        warning=_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_float(val) -> float:
    if val is None or np.isnan(val) or np.isinf(val):
        return 0.0
    return float(val)


def _compute_stats(arr: np.ndarray) -> ChannelStats:
    clean_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return ChannelStats(
        mean = _clean_float(np.mean(clean_arr)),
        std  = _clean_float(np.std(clean_arr)),
        min  = _clean_float(np.min(clean_arr)),
        max  = _clean_float(np.max(clean_arr)),
        p25  = _clean_float(np.percentile(clean_arr, 25)),
        p75  = _clean_float(np.percentile(clean_arr, 75)),
    )


def _zero_stats() -> ChannelStats:
    return ChannelStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _interpret(
    vh: ChannelStats,
    vv: ChannelStats,
    ratio: ChannelStats,
    corr: float,
) -> str:
    """
    Heuristic SAR backscatter interpretation.
    Based on general remote-sensing rules-of-thumb for C-band SAR.
    These are NOT the output of a trained classifier.
    """
    lines: list[str] = [
        "Heuristic SAR backscatter interpretation (C-band, rules-of-thumb):",
        "",
    ]

    # -- VV level -------------------------------------------------------------
    if vv.mean < 0.04:
        lines.append(
            "VV backscatter: LOW  -- may indicate calm open water or very smooth "
            "surfaces (specular reflection away from sensor)."
        )
    elif vv.mean > 0.25:
        lines.append(
            "VV backscatter: HIGH -- may indicate urban structures, double-bounce, "
            "or rough terrain."
        )
    else:
        lines.append(
            "VV backscatter: MODERATE -- consistent with mixed terrain, vegetation, "
            "or agricultural areas."
        )

    # -- VH level -------------------------------------------------------------
    if vh.mean < 0.02:
        lines.append(
            "VH backscatter: LOW -- cross-polarisation is low, suggesting "
            "limited volume scattering (open water or smooth bare soil)."
        )
    elif vh.mean > 0.15:
        lines.append(
            "VH backscatter: HIGH -- strong cross-polarisation, consistent with "
            "dense vegetation (forest, tall crops) causing volume scattering."
        )
    else:
        lines.append(
            "VH backscatter: MODERATE -- moderate volume scattering; "
            "possible sparse vegetation or rough surface."
        )

    # -- VH/VV ratio ----------------------------------------------------------
    lines.append("")
    r = ratio.mean
    if r > 0.75:
        lines.append(
            f"VH/VV ratio: {r:.3f} (HIGH) -- dominant volume scattering mechanism, "
            "common in forested or densely vegetated areas."
        )
    elif r < 0.35:
        lines.append(
            f"VH/VV ratio: {r:.3f} (LOW) -- surface or double-bounce scattering "
            "dominant; consistent with water bodies, bare soil, or urban structures."
        )
    else:
        lines.append(
            f"VH/VV ratio: {r:.3f} (MODERATE) -- mixed scattering mechanisms."
        )

    # -- Correlation ----------------------------------------------------------
    lines.append("")
    if corr > 0.90:
        lines.append(
            f"VH/VV correlation: {corr:.3f} (HIGH) -- spatially coherent scene, "
            "uniform land cover likely."
        )
    elif corr < 0.55:
        lines.append(
            f"VH/VV correlation: {corr:.3f} (LOW) -- heterogeneous scene or "
            "complex mix of scattering mechanisms."
        )
    else:
        lines.append(
            f"VH/VV correlation: {corr:.3f} (MODERATE) -- mixed spatial pattern."
        )

    lines.append("")
    lines.append(
        "Note: These interpretations are heuristic rules, not the output of a "
        "trained model. Full multimodal analysis requires all 12 channels."
    )

    return "\n".join(lines)
