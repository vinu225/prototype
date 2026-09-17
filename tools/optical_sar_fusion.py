"""
tools/optical_sar_fusion.py
============================
Specialist tool: Cross-modal optical + SAR analysis.

INPUT
-----
  optical_image : [B, C_opt, 120, 120]  — multispectral (Sentinel-2 bands)
  sar_image     : [B, C_sar, 120, 120]  — SAR (Sentinel-1 channels)
  query         : str

CHANNEL CONVENTIONS (matching Colab band order)
------------------------------------------------
  Full 12-ch  (C=12) : [B02 B03 B04 B08 B05 B06 B07 B11 B12 B8A VH VV]
  Optical only (C=10): [B02 B03 B04 B08 B05 B06 B07 B11 B12 B8A]  ← not supported by encoder
  SAR only     (C=2) : [VH VV]                                       ← not supported alone

SUPPORTED INPUT CONFIGURATIONS
--------------------------------
  Config A — both images are 12-channel (full Sentinel-1+2):
      Each is passed through the encoder separately → two 640-D vectors.
      Fused via element-wise average → single 640-D fused vector.

  Config B — manual modality labels:
      User labels which image is 'optical' and which is 'sar'.
      Both must still be 12-channel (the encoder requires exactly 12 channels).

FUSION STRATEGY (prototype)
----------------------------
  fused_features = (feat_optical + feat_sar) / 2.0   [B, 640]

  This is ELEMENT-WISE FEATURE AVERAGING.  It is a simple prototype fusion
  strategy — NOT a trained cross-modal attention or late-fusion model.

  The fused vector is passed through the standard VisionProjector → LLM.

HONEST LIMITATIONS
------------------
- Both inputs must still be 12-channel (the MobileViT-S encoder was
  trained on exactly 12 channels).
- Feature averaging is a rudimentary fusion strategy.
- This is NOT a validated cross-modal fusion model.
- For production use, train a dedicated optical-SAR fusion network
  (e.g. two-branch encoder with cross-attention).
"""

import time
import torch

from tools.base          import BaseSpecialistTool, ToolResult, clean_tool_output
from pipeline.preprocess import preprocess_image
from models.model_pool   import get_vlm
from models.vlm          import VISION_CHECKPOINT, LLM_NAME

_FUSION_SYSTEM_PROMPT = (
    "You are a remote-sensing multimodal analysis assistant.\n\n"
    "You are given two co-registered observations of the same geographic region:\n"
    "- Optical/multispectral imagery: provides spectral and visual information.\n"
    "- SAR imagery: provides structural/backscatter information and can provide complementary information under cloud cover.\n\n"
    "Use the complementary information from both modalities to answer the user's question.\n\n"
    "Do not simply describe the two modalities separately.\n"
    "Do not output instructions such as 'calculate percentage' unless the user explicitly asks for a calculation.\n"
    "Do not invent numerical percentages, geographic coordinates, object counts, or classifications that cannot be supported.\n\n"
    "Answer the user's query directly and concisely.\n"
    "If the two modalities provide insufficient evidence, say so."
)


class OpticalSARFusion(BaseSpecialistTool):
    """
    Prototype optical + SAR cross-modal analysis.

    Both images must be 12-channel Sentinel patches.
    Features are extracted independently and averaged for fusion.
    """

    TOOL_NAME = "optical_sar_fusion"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._vlm = get_vlm(device=device)

    def run(
        self,
        optical_image,
        sar_image,
        query: str = (
            "Using both the optical and SAR observations, "
            "identify the land-cover types and any notable features."
        ),
        max_new_tokens: int = 80,
        optical_label:  str = "optical",
        sar_label:      str = "sar",
    ) -> ToolResult:
        """
        Parameters
        ----------
        optical_image : array-like
            12-channel optical patch (Sentinel-2 + SAR layout).
        sar_image : array-like
            12-channel SAR patch (same layout — encoder requires 12 channels).
        query : str
            Natural-language query.
        max_new_tokens : int
            Max generation tokens.
        optical_label : str
            User-supplied label for the optical image (for metadata).
        sar_label : str
            User-supplied label for the SAR image (for metadata).

        Returns
        -------
        ToolResult with:
            answer           — VLM response to query
            confidence       — None  (no calibrated score)
            confidence_type  — 'not_available'
            visual_evidence  — feature cosine similarity between modalities
        """
        t0 = time.perf_counter()
        try:
            # ── Preprocess both modalities ───────────────────────────────
            t_optical = preprocess_image(optical_image, device=self.device)
            t_sar     = preprocess_image(sar_image,     device=self.device)

            # Validate compatible batch sizes
            if t_optical.shape[0] != t_sar.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: optical B={t_optical.shape[0]}, "
                    f"sar B={t_sar.shape[0]}."
                )

            encoder = self._vlm.vision_encoder

            # ── Extract features independently ───────────────────────────
            with torch.no_grad():
                feat_optical = encoder(t_optical)   # [B, 640]
                feat_sar     = encoder(t_sar)        # [B, 640]

                # ── Feature-level fusion (element-wise average) ──────────
                # Prototype strategy: average the two 640-D representations.
                # This preserves the 640-D shape expected by the projector.
                fused_features = (feat_optical + feat_sar) / 2.0  # [B, 640]

                # Compute cosine similarity between modalities (informational)
                cos_sim = torch.nn.functional.cosine_similarity(
                    feat_optical, feat_sar, dim=-1
                ).mean().item()

                # ── VLM generation on fused features ─────────────────────
                user_content = (
                    "Input Context:\n"
                    f"- Optical modality: Multispectral bands ({optical_label})\n"
                    f"- SAR modality: Polarimetric radar channels ({sar_label})\n"
                    f"- Modality Feature Cosine Similarity: {cos_sim:.4f}\n\n"
                    f"User Question:\n{query}\n\n"
                    "Answer Constraints:\n"
                    "State the identified land-cover types and features directly and concisely by synthesizing both modalities. "
                    "Do not output introductory explanations, meta-instructions, or calculation steps."
                )
                messages = [
                    {"role": "system", "content": _FUSION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                formatted_prompt = self._vlm.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                raw_answer = self._vlm.generate_from_visual_features(
                    fused_features,
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                )

                answer = clean_tool_output(raw_answer, query=query)

            elapsed = time.perf_counter() - t0

            return ToolResult(
                task="optical_sar_fusion",
                success=True,
                answer=answer,
                confidence=None,
                confidence_type="not_available",
                visual_evidence={
                    "modality_cosine_similarity": round(float(cos_sim), 4),
                    "fusion_strategy": "element_wise_average",
                    "note": (
                        "Cosine similarity between optical and SAR 640-D features "
                        "is informational only.  Feature average is a prototype "
                        "fusion method — not a trained cross-modal model."
                    ),
                },
                metadata={
                    "optical_label":    optical_label,
                    "sar_label":        sar_label,
                    "optical_shape":    list(t_optical.shape),
                    "sar_shape":        list(t_sar.shape),
                    "fused_feat_dim":   640,
                    "projected_dim":    896,
                    "limitation": (
                        "Both images must be 12-channel (encoder constraint). "
                        "Feature averaging is a rudimentary fusion strategy. "
                        "Replace with a dedicated optical-SAR fusion network "
                        "for production use."
                    ),
                },
                model_used=f"encoder={VISION_CHECKPOINT} | llm={LLM_NAME}",
                processing_time=elapsed,
            )

        except Exception as exc:
            return ToolResult(
                task="optical_sar_fusion",
                success=False,
                answer="",
                confidence=None,
                confidence_type="not_available",
                visual_evidence=None,
                processing_time=time.perf_counter() - t0,
                error=str(exc),
            )
