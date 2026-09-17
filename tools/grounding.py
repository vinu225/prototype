"""
tools/grounding.py
===================
Specialist tool: Visual grounding (spatial localisation).

PURPOSE
-------
Given an image + a text query, identify the spatial region (bounding box)
corresponding to the requested object or concept.

CURRENT STATUS: NOT IMPLEMENTED
---------------------------------
No genuine grounding model is available in this prototype.

The current pipeline (MobileViT-S encoder → projector → Qwen2.5-0.5B) does
NOT produce bounding-box coordinates.  Qwen2.5-0.5B-Instruct is a pure
language model without spatial detection heads.

WHAT THIS MODULE DOES
----------------------
- Exposes the correct interface so Phase 3 routing can call it.
- Returns a ToolResult with success=False and a clear 'not_implemented'
  status rather than fabricated bounding boxes.

FUTURE IMPLEMENTATION
----------------------
To implement genuine grounding, a dedicated model such as:
  - Grounding DINO
  - OWL-ViT
  - SAM (Segment Anything Model)
  - A fine-tuned remote-sensing grounding model (e.g. RSVG)
would need to be integrated.  That work belongs to a future phase.

DO NOT generate arbitrary bounding boxes to satisfy the interface.
"""

import time

from tools.base import BaseSpecialistTool, ToolResult

_NOT_IMPLEMENTED_MSG = (
    "Visual grounding is not implemented in Phase 2. "
    "No genuine spatial detection model is available in this prototype. "
    "Bounding-box output would require a dedicated grounding model "
    "(e.g. Grounding DINO, OWL-ViT, or a remote-sensing grounding network). "
    "This capability is scheduled for a future phase."
)


class VisualGrounding(BaseSpecialistTool):
    """
    Stub specialist tool for visual grounding.

    The run() method always returns success=False with a clear
    'not_implemented' status.  No bounding boxes are fabricated.

    Expected output format (when implemented):
        visual_evidence = {
            "boxes": [
                {"x_min": int, "y_min": int, "x_max": int, "y_max": int,
                 "label": str, "score": float}
            ]
        }
    """

    TOOL_NAME = "visual_grounding"

    def __init__(self, device: str = "cpu"):
        self.device = device
        # No model loaded — grounding not available yet.

    def run(
        self,
        image,
        query: str = "",
    ) -> ToolResult:
        """
        Parameters
        ----------
        image : array-like
            Satellite patch (any shape — not validated here since no
            processing occurs).
        query : str
            Object or concept to localise.

        Returns
        -------
        ToolResult with success=False and status='not_implemented'.
        Bounding boxes are NOT generated.
        """
        t0 = time.perf_counter()

        return ToolResult(
            task="visual_grounding",
            success=False,
            answer=_NOT_IMPLEMENTED_MSG,
            confidence=None,
            confidence_type="not_available",
            visual_evidence={
                "status": "not_implemented",
                "boxes":  None,
                "note":   (
                    "Boxes are None because no genuine grounding model is "
                    "available.  Do not replace None with dummy coordinates."
                ),
            },
            metadata={
                "query":                query,
                "implementation_phase": "future",
                "suggested_models":     [
                    "Grounding DINO",
                    "OWL-ViT",
                    "SAM (Segment Anything)",
                    "RSVG (Remote Sensing Visual Grounding)",
                ],
            },
            model_used="none",
            processing_time=time.perf_counter() - t0,
            error="not_implemented",
        )
