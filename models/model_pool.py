"""
models/model_pool.py
=====================
Centralized lazy-loaded singleton for the three model components.

ALL specialist tools and the inference pipeline must obtain model instances
from this module — do NOT instantiate SatQueryVLM separately per-tool.

CPU inference timings measured in Phase 1:
    Encoder  :  ~6.5s per call
    Full VLM : ~104s per generation

Loading the VLM once (lazy, on first use) and sharing it across tools
avoids loading 988 MB of weights multiple times.

Usage
-----
    from models.model_pool import get_vlm

    vlm = get_vlm(device="cpu")
    # vlm.vision_encoder  — SatelliteEncoder (frozen)
    # vlm.projector        — VisionProjector  (trainable)
    # vlm.llm              — Qwen2.5-0.5B-Instruct (frozen)
    # vlm.tokenizer
"""

import torch
from models.vlm import SatQueryVLM, VISION_CHECKPOINT, LLM_NAME

_vlm_instance: SatQueryVLM | None = None
_vlm_device: str = "cpu"


def get_vlm(device: str = "cpu") -> SatQueryVLM:
    """
    Return the shared SatQueryVLM singleton.

    The first call loads encoder + projector + LLM into memory.
    Subsequent calls return the cached instance immediately.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'.  If the singleton is already loaded on a
        different device, a warning is printed (re-loading is expensive).

    Returns
    -------
    SatQueryVLM
        The shared model instance, in eval() mode.
    """
    global _vlm_instance, _vlm_device

    if _vlm_instance is None:
        _vlm_instance = SatQueryVLM(
            vision_checkpoint=VISION_CHECKPOINT,
            llm_name=LLM_NAME,
            device=device,
        )
        _vlm_instance.eval()
        _vlm_device = device

    elif device != _vlm_device:
        print(
            f"[model_pool] WARNING: requested device '{device}' but singleton "
            f"is already on '{_vlm_device}'. Returning existing instance."
        )

    return _vlm_instance
