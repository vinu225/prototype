"""
pipeline/inference.py
======================
Main inference interface.  Exposes a single clean function:

    result = analyze(image, query)

Internal flow (matches Colab exactly):
    image → preprocess → SatelliteEncoder → VisionProjector → VLM → text

SOURCE OF TRUTH: Colab cells 34 (SatQueryVLM.generate_answer), 36 (inference.py).

The returned dict reports every intermediate dimension so callers can verify
the pipeline is wired correctly.  Confidence is NOT reported (model produces
no calibrated probability output at this stage).
"""

import time
import torch

from pipeline.preprocess  import preprocess_image
from models.model_pool    import get_vlm
from models.vlm           import VISION_CHECKPOINT, LLM_NAME


def analyze(
    image,
    query:          str  = "What type of land cover is present in this satellite image?",
    device:         str  = "cpu",
    max_new_tokens: int  = 48,
) -> dict:
    """
    Run the full multimodal inference pipeline on one satellite patch.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Raw 12-channel patch.  Accepted shapes: [C,H,W] or [B,C,H,W].
        Must be [12, 120, 120] (or batched equivalent).
    query : str
        Natural-language question about the image.
    device : str
        'cuda' or 'cpu'.
    max_new_tokens : int
        Max tokens to generate (default 48, from Colab).

    Returns
    -------
    dict with keys:
        answer               : str    — generated text response
        input_shape          : list   — e.g. [1, 12, 120, 120]
        visual_feature_dim   : int    — 640  (encoder output)
        projected_feature_dim: int    — 896  (projector output / LLM hidden dim)
        vision_encoder       : str    — checkpoint id
        language_model       : str    — LLM id
        latency_seconds      : float  — wall-clock time for full pipeline
        confidence           : None   — not available (model produces no score)
    """
    t0 = time.perf_counter()

    # ── 1. Preprocess ───────────────────────────────────────────────────
    tensor = preprocess_image(image, device=device)   # [B, 12, 120, 120]
    input_shape = list(tensor.shape)

    # ── 2–5. Encoder → Projector → LLM (inside VLM) ────────────────────
    vlm = get_vlm(device=device)

    with torch.no_grad():
        # Verify intermediate shapes in a single forward debug pass
        visual_feats = vlm.vision_encoder(tensor)         # [B, 640]
        projected    = vlm.projector(visual_feats)        # [B, 896]
        visual_feature_dim    = visual_feats.shape[-1]
        projected_feature_dim = projected.shape[-1]

    # ── 6. Text generation ──────────────────────────────────────────────
    answer = vlm.generate_response(tensor, query, max_new_tokens=max_new_tokens)

    latency = time.perf_counter() - t0

    return {
        "answer":                answer,
        "input_shape":           input_shape,
        "visual_feature_dim":    visual_feature_dim,    # must be 640
        "projected_feature_dim": projected_feature_dim, # must be 896
        "vision_encoder":        VISION_CHECKPOINT,
        "language_model":        LLM_NAME,
        "latency_seconds":       round(latency, 3),
        "confidence":            None,  # not produced by this model
    }
