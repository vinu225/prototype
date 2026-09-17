"""
models/satellite_encoder.py
============================
Wraps the pretrained BigEarthNet MobileViT-S checkpoint from:
    BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.1.1

SOURCE OF TRUTH: Colab notebook cells 32, 93, 101, 117, 122.

Critical design note
--------------------
The `BigEarthNetv2_0_ImageClassifier` has an internal structure:
    classifier
        └─ model          (ConfigILM wrapper)
               └─ vision_encoder  (MobileViT-S backbone, timm)

Calling `classifier(x)` would run through the 19-class classification head
and return 19-D logits — which is WRONG for our projector (expects 640-D).

We BYPASS the head by calling:
    classifier.model.vision_encoder.forward_features(x)

This returns the raw spatial feature map [B, 640, H', W'], which we then
global-average-pool to [B, 640].

Verified in the Colab: vision_features.shape == [1, 640]
"""

import torch
import torch.nn as nn

# ── configilm monkey-patch ──────────────────────────────────────────────────
# configilm ≥ 0.7 tries to call ILMConfiguration.items() during hub lookups.
# The original class does not define items(), causing AttributeError on load.
# This patch is taken verbatim from the working Colab cells (20, 24, 91, 93…).
from configilm.ConfigILM import ILMConfiguration
if not hasattr(ILMConfiguration, "items"):
    ILMConfiguration.items = lambda self: self.__dict__.items()

from reben_publication.BigEarthNetv2_0_ImageClassifier import (
    BigEarthNetv2_0_ImageClassifier,
)

# Checkpoint name as used in the Colab
ENCODER_CHECKPOINT = "BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.1.1"

# Expected output dimensionality of the MobileViT-S backbone
VISION_FEATURE_DIM = 640

# Expected input shape constraints
EXPECTED_CHANNELS = 12
EXPECTED_SPATIAL   = 120   # height == width


class SatelliteEncoder(nn.Module):
    """
    Frozen BigEarthNet MobileViT-S visual encoder.

    Input  : torch.Tensor  [B, 12, 120, 120]  float32
    Output : torch.Tensor  [B, 640]            float32

    The model is always kept in eval() mode with all gradients disabled,
    matching the Colab's 'frozen backbone' approach.
    """

    def __init__(self, checkpoint_name: str = ENCODER_CHECKPOINT):
        super().__init__()

        # Load the full pretrained classifier wrapper.
        # We only need the backbone inside it — the classification head
        # (19-class sigmoid output) is never called.
        self.classifier = BigEarthNetv2_0_ImageClassifier.from_pretrained(
            checkpoint_name
        )

        # Freeze all parameters — nothing in the encoder trains.
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, 12, 120, 120], dtype float32.
            Channel order: B02 B03 B04 B08 B05 B06 B07 B11 B12 B8A VH VV

        Returns
        -------
        torch.Tensor
            Shape [B, 640]  —  global-average-pooled backbone features.
            NOT 19-class logits.
        """
        # ── Input validation ────────────────────────────────────────────
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4-D tensor [B, C, H, W], got shape {list(x.shape)}"
            )
        if x.shape[1] != EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected exactly {EXPECTED_CHANNELS} channels (B02…VV), "
                f"got {x.shape[1]}. Do NOT silently drop or add channels."
            )
        if x.shape[2] != EXPECTED_SPATIAL or x.shape[3] != EXPECTED_SPATIAL:
            raise ValueError(
                f"Expected spatial size {EXPECTED_SPATIAL}×{EXPECTED_SPATIAL}, "
                f"got {x.shape[2]}×{x.shape[3]}. Do NOT silently resize."
            )

        # ── Feature extraction (no grad, frozen) ────────────────────────
        with torch.no_grad():
            # self.classifier.model  → ConfigILM
            # .vision_encoder        → MobileViT-S timm model
            # .forward_features(x)   → spatial feature map, bypasses head
            vision_enc = self.classifier.model.vision_encoder

            if hasattr(vision_enc, "forward_features"):
                # Preferred path: timm models expose forward_features()
                # which stops before the classifier head.
                # Returns [B, 640, H', W'] for MobileViT-S.
                features = vision_enc.forward_features(x)
            else:
                # Fallback: call the encoder directly (some timm versions
                # already strip the head when num_classes=0 is set).
                features = vision_enc(x)

            # ── Pool to [B, 640] ────────────────────────────────────────
            if features.ndim == 4:
                # [B, C, H', W'] → global average pool → [B, C]
                features = torch.mean(features, dim=[2, 3])
            elif features.ndim == 3:
                # [B, T, C] token sequence → pool over token dim → [B, C]
                features = torch.mean(features, dim=1)
            # If already [B, C], no pooling needed.

        # ── Output validation ───────────────────────────────────────────
        assert features.shape[-1] == VISION_FEATURE_DIM, (
            f"Encoder produced {features.shape[-1]}-D features, "
            f"expected {VISION_FEATURE_DIM}. "
            "This means forward_features() returned unexpected output — "
            "check that the checkpoint is mobilevit_s-all-v0.1.1."
        )

        return features
