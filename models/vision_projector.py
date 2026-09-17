"""
models/vision_projector.py
===========================
Trainable alignment MLP that maps the 640-D visual feature produced by the
MobileViT-S encoder into the 896-D embedding space of Qwen2.5-0.5B-Instruct.

SOURCE OF TRUTH: Colab notebook cells 33, 93, 101, 117, 122.

Architecture (verbatim from Colab):
    Linear(640 → 1024)
    GELU()
    Linear(1024 → 896)

Qwen2.5-0.5B-Instruct has hidden_size = 896, which is why the output dim
is hard-coded to 896 here (and confirmed via `llm.config.hidden_size` in the
Colab's SatQueryVLM.__init__).

The encoder and LLM are frozen; only this projector trains.
"""

import torch
import torch.nn as nn

# Dimension constants — do NOT change without updating the Colab source.
INPUT_DIM  = 640    # MobileViT-S backbone output
HIDDEN_DIM = 1024   # intermediate expansion
OUTPUT_DIM = 896    # Qwen2.5-0.5B hidden_size


class VisionProjector(nn.Module):
    """
    Two-layer MLP: 640 → 1024 → 896.

    Input  : torch.Tensor  [B, 640]
    Output : torch.Tensor  [B, 896]
    """

    def __init__(
        self,
        input_dim:  int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
    ):
        super().__init__()

        # Exact sequential architecture from the Colab.
        # Activation: GELU (not ReLU, not SiLU).
        # No dropout or layer-norm at this stage (not present in Colab).
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, 640].  Must come directly from SatelliteEncoder.

        Returns
        -------
        torch.Tensor
            Shape [B, 896].  Ready to be unsqueeze(1) → [B, 1, 896]
            for prepending to text token embeddings.
        """
        # ── Shape assertion — catch dimension bugs immediately ──────────
        assert x.ndim == 2, (
            f"VisionProjector expects 2-D input [B, {INPUT_DIM}], "
            f"got shape {list(x.shape)}"
        )
        assert x.shape[-1] == INPUT_DIM, (
            f"VisionProjector expects last dim = {INPUT_DIM} (MobileViT-S output), "
            f"got {x.shape[-1]}. "
            "If this is 19, the encoder incorrectly returned classifier logits — "
            "fix SatelliteEncoder to use forward_features() instead."
        )

        projected = self.projector(x)

        # ── Output assertion ────────────────────────────────────────────
        assert projected.shape[-1] == OUTPUT_DIM, (
            f"Projector produced {projected.shape[-1]}-D output, "
            f"expected {OUTPUT_DIM}."
        )

        return projected
