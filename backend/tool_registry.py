"""
backend/tool_registry.py
=========================
Central tool registry for SatQuery AI Phase 3.

All five specialist tools are instantiated once, sharing the VLM via
models/model_pool.py.  No model is loaded per-request.

Registry structure
------------------
TOOL_REGISTRY : dict[str, BaseSpecialistTool]
    Keys are task names matching SatQueryRouter output.

Tools
-----
  single_vqa          → SingleImageVQA
  captioning          → SceneCaptioning
  grounding           → VisualGrounding    (stub — not_implemented)
  change_detection    → BiTemporalChangeDetection
  optical_sar_fusion  → OpticalSARFusion
"""

from __future__ import annotations

import logging
import os

_LOG = logging.getLogger(__name__)

# Device selection — respects SATQUERY_DEVICE env var (default: cpu)
_DEVICE: str = os.environ.get("SATQUERY_DEVICE", "cpu")

# ── Lazy registry cache ────────────────────────────────────────────────────────
_registry: dict | None = None


def get_registry() -> dict:
    """
    Return the shared tool registry, initialising it on first call.

    All tools share the single VLM instance from models.model_pool.
    This function is safe to call multiple times — tools are instantiated once.
    """
    global _registry
    if _registry is not None:
        return _registry

    _LOG.info("Initialising SatQuery AI tool registry (device=%s)…", _DEVICE)

    # Imports deferred so the registry module can be imported without
    # immediately loading Torch / Transformers (useful for lightweight tests).
    from configilm.ConfigILM import ILMConfiguration  # noqa: F401 — patch must run first
    if not hasattr(ILMConfiguration, "items"):
        ILMConfiguration.items = lambda self: self.__dict__.items()

    from tools.single_vqa       import SingleImageVQA
    from tools.captioning       import SceneCaptioning
    from tools.grounding        import VisualGrounding
    from tools.change_detection import BiTemporalChangeDetection
    from tools.optical_sar_fusion import OpticalSARFusion

    _registry = {
        "single_vqa":         SingleImageVQA(device=_DEVICE),
        "captioning":         SceneCaptioning(device=_DEVICE),
        "grounding":          VisualGrounding(device=_DEVICE),
        "change_detection":   BiTemporalChangeDetection(device=_DEVICE),
        "optical_sar_fusion": OpticalSARFusion(device=_DEVICE),
    }

    _LOG.info(
        "Tool registry ready: %s", list(_registry.keys())
    )
    return _registry


def get_tool(task_name: str):
    """
    Retrieve a specialist tool from the registry by task name.

    Raises
    ------
    KeyError if task_name is not registered.
    """
    reg = get_registry()
    if task_name not in reg:
        raise KeyError(
            f"Unknown task {task_name!r}. "
            f"Available: {list(reg.keys())}"
        )
    return reg[task_name]
