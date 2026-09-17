"""
tools/change_detection.py
==========================
Specialist tool: Bi-temporal change analysis.

INPUT
-----
  image_before : [B, 12, 120, 120]  — earlier acquisition
  image_after  : [B, 12, 120, 120]  — later acquisition
  query        : str                  — e.g. "What changed between these dates?"

PIPELINE (prototype feature-difference approach)
-------------------------------------------------
  1. Preprocess + validate both images independently.
  2. Extract 640-D features from each using the frozen MobileViT-S encoder.
  3. Compute:
       diff_vector = feat_after - feat_before          [B, 640]
       change_magnitude = L2-norm(diff_vector) / sqrt(640)
                                                        scalar, normalised ∈ [0, ∞)
  4. Heuristic change decision:
       change_detected = change_magnitude > CHANGE_THRESHOLD
  5. Pass diff_vector through the VLM projector + Qwen2.5 to generate
     a natural-language change description.

HONEST LIMITATIONS
------------------
- This is a PROTOTYPE feature-difference method, NOT a trained change-
  detection model (e.g. Siamese network, CD-TransUNet, BIT-CD, ChangeMamba).
- The L2 distance heuristic does not distinguish between radiometric
  variation (cloud, shadow, seasonal) and genuine land-cover change.
- The VLM is conditioned on the *difference feature vector*, which has no
  direct semantic interpretation in this prototype.
- Do NOT present this as a validated change-detection system.
- Pixel-level change maps are NOT produced (would require spatial feature
  maps from the encoder or a dedicated segmentation head).

FUTURE WORK
-----------
Replace with a dedicated bi-temporal remote-sensing change-detection
model (e.g. Siamese encoder with change head trained on LEVIR-CD or
WHU-CD datasets).
"""

import time
import torch

from tools.base          import BaseSpecialistTool, ToolResult, clean_tool_output
from pipeline.preprocess import preprocess_image
from models.model_pool   import get_vlm
from models.vlm          import VISION_CHECKPOINT, LLM_NAME

# Normalised L2 threshold for heuristic change detection.
# Features are 640-D; normalised magnitude > 1.0 suggests meaningful shift.
CHANGE_THRESHOLD = 0.80

_CHANGE_SYSTEM_PROMPT = (
    "You are a remote-sensing change-analysis assistant.\n\n"
    "Two observations of the same geographic area are provided:\n"
    "IMAGE BEFORE\n"
    "IMAGE AFTER\n\n"
    "A feature-difference signal has been computed between the two observations.\n\n"
    "Answer the user's question about changes between the observations.\n"
    "Focus on changes in land cover, vegetation, water, built-up areas, roads, structures, "
    "or other visible geographic features.\n\n"
    "Do not discuss satellite altitude, satellite speed, orbital parameters, frequency, "
    "or unrelated sensor physics unless explicitly asked.\n\n"
    "Do not invent a spatial change map or exact change location if no spatial change mask is available.\n\n"
    "If the evidence is insufficient, state that clearly.\n\n"
    "Return:\n"
    "1. Whether an apparent change is present.\n"
    "2. What type of geographic change is suggested.\n"
    "3. A concise explanation based on the available evidence."
)


class BiTemporalChangeDetection(BaseSpecialistTool):
    """
    Prototype bi-temporal change analysis tool.

    Uses feature-level L2 difference as a heuristic change indicator and
    the VLM to generate a text description of the change.
    """

    TOOL_NAME = "bitemporal_change_detection"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._vlm = get_vlm(device=device)

    def run(
        self,
        image_before,
        image_after,
        query: str = "What changed between these two satellite observations?",
        max_new_tokens: int = 80,
    ) -> ToolResult:
        """
        Parameters
        ----------
        image_before, image_after : array-like
            12-channel patches of the same area at two different times.
            Both must be [12, 120, 120] (or batched equivalent).
        query : str
            Natural-language question about the change.
        max_new_tokens : int
            Max tokens for the LLM response.

        Returns
        -------
        ToolResult with:
            answer           — VLM-generated change description
            confidence       — normalised L2 magnitude (heuristic)
            confidence_type  — 'heuristic'
            visual_evidence  — change_magnitude, change_detected
        """
        t0 = time.perf_counter()
        try:
            # ── Preprocess both images ───────────────────────────────────
            t_before = preprocess_image(image_before, device=self.device)
            t_after  = preprocess_image(image_after,  device=self.device)

            # Validate compatible batch sizes
            if t_before.shape[0] != t_after.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: image_before has B={t_before.shape[0]}, "
                    f"image_after has B={t_after.shape[0]}."
                )

            encoder = self._vlm.vision_encoder

            # ── Extract features from both images ────────────────────────
            with torch.no_grad():
                feat_before = encoder(t_before)   # [B, 640]
                feat_after  = encoder(t_after)    # [B, 640]

                # ── Feature difference (prototype fusion) ────────────────
                diff_vector = feat_after - feat_before   # [B, 640]

                # Normalised L2 magnitude per sample, averaged over batch
                l2_raw   = torch.norm(diff_vector, dim=-1)           # [B]
                l2_norm  = (l2_raw / (640 ** 0.5)).mean().item()     # scalar

                change_detected = l2_norm > CHANGE_THRESHOLD

                # ── VLM generation conditioned on difference vector ──────
                detected_str = "Yes (exceeds threshold)" if change_detected else "No (below threshold)"
                user_content = (
                    "Input Context:\n"
                    "- Observation 1: Earlier satellite acquisition (IMAGE BEFORE)\n"
                    "- Observation 2: Later satellite acquisition (IMAGE AFTER)\n"
                    f"- Computed Feature Difference Signal: L2 distance = {l2_norm:.4f} (Heuristic threshold = {CHANGE_THRESHOLD})\n"
                    f"- Apparent Change Detected: {detected_str}\n\n"
                    f"User Question:\n{query}\n\n"
                    "Answer Constraints:\n"
                    "Provide a direct assessment addressing only geographic and land-cover change:\n"
                    "1. State whether an apparent change is present.\n"
                    "2. Describe what geographic or surface change is indicated by the observations.\n"
                    "3. Provide a brief explanation based on the available evidence.\n"
                    "Do not discuss satellite flight parameters or sensor physics."
                )
                messages = [
                    {"role": "system", "content": _CHANGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                formatted_prompt = self._vlm.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                raw_change_answer = self._vlm.generate_from_visual_features(
                    diff_vector,
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                )

                change_answer = clean_tool_output(raw_change_answer, query=query)

            elapsed = time.perf_counter() - t0

            return ToolResult(
                task="bitemporal_change_detection",
                success=True,
                answer=change_answer,
                confidence=round(float(l2_norm), 4),
                confidence_type="heuristic",
                visual_evidence={
                    "change_detected":        bool(change_detected),
                    "change_magnitude_l2":    round(float(l2_norm), 4),
                    "change_threshold":       CHANGE_THRESHOLD,
                    "spatial_change_map":     None,   # not available yet
                    "note": (
                        "change_magnitude is normalised L2 distance between "
                        "640-D backbone features. This is a HEURISTIC metric, "
                        "not output from a trained change-detection model."
                    ),
                },
                metadata={
                    "before_shape":           list(t_before.shape),
                    "after_shape":            list(t_after.shape),
                    "feature_dim":            640,
                    "method":                 "feature_difference_l2",
                    "model_limitation": (
                        "Prototype only. Replace with a trained Siamese "
                        "change-detection network for production use."
                    ),
                },
                model_used=f"encoder={VISION_CHECKPOINT} | llm={LLM_NAME}",
                processing_time=elapsed,
            )

        except Exception as exc:
            return ToolResult(
                task="bitemporal_change_detection",
                success=False,
                answer="",
                confidence=None,
                confidence_type="not_available",
                visual_evidence=None,
                processing_time=time.perf_counter() - t0,
                error=str(exc),
            )
