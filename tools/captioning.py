"""
tools/captioning.py
====================
Specialist tool: Remote-sensing scene captioning.

Generates a concise textual description of the major land-cover types
and structures visible in a single 12-channel satellite image.

This satisfies the mandatory second single-image capability for Phase 2.

IMPORTANT: This uses the same Qwen2.5-0.5B-Instruct + MobileViT-S pipeline
as Phase 1 — it is NOT a dedicated image-captioning model (e.g., BLIP-2).
The quality of the caption is limited by the prototype nature of the
projector (randomly initialised, not trained on BigEarthNet QA pairs yet).

The result is clearly labelled as prototype output.
"""

import time
import torch

from tools.base          import BaseSpecialistTool, ToolResult, clean_tool_output
from pipeline.preprocess import preprocess_image
from models.model_pool   import get_vlm
from models.vlm          import VISION_CHECKPOINT, LLM_NAME

# Dedicated scene-description system prompt
_CAPTION_SYSTEM_PROMPT = (
    "You are describing a remote-sensing satellite image.\n"
    "Describe only observable or inferable scene content.\n"
    "Mention dominant land-cover characteristics, major visible structures/features, "
    "vegetation, water, and terrain when relevant.\n"
    "Do not invent geographic locations, dates, percentages, object counts, or sensor information.\n"
    "Return one concise paragraph.\n"
    "Do not describe the reasoning process.\n"
    "Do not generate a question or instructions."
)


class SceneCaptioning(BaseSpecialistTool):
    """
    Generate a scene caption for a single 12-channel remote-sensing image.

    Input
    -----
    image : np.ndarray or torch.Tensor  [C,H,W] or [B,C,H,W]
             Must have C=12, H=W=120.

    Output
    ------
    ToolResult with:
        answer           — generated caption text
        confidence       — None  (no calibrated score)
        confidence_type  — 'not_available'

    Limitations
    -----------
    - Uses the Phase 1 VLM pipeline (projector not yet trained).
    - Output quality improves after projector training in Phase 3+.
    - Not equivalent to a dedicated remote-sensing captioning model.
    """

    TOOL_NAME = "scene_captioning"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._vlm = get_vlm(device=device)

    def run(
        self,
        image,
        custom_prompt: str | None = None,
        max_new_tokens: int = 80,
    ) -> ToolResult:
        """
        Parameters
        ----------
        image : array-like
            12-channel satellite patch.
        custom_prompt : str, optional
            Override the default captioning prompt.
        max_new_tokens : int
            Max tokens to generate.

        Returns
        -------
        ToolResult
        """
        t0 = time.perf_counter()
        user_task = custom_prompt or "Describe the major land-cover types and visible features in this satellite image."

        try:
            tensor = preprocess_image(image, device=self.device)

            user_content = (
                "Input Context: A 12-channel multispectral remote-sensing satellite image is provided.\n\n"
                f"User Request:\n{user_task}\n\n"
                "Answer Constraints:\n"
                "Return exactly one concise factual scene description paragraph. "
                "Do not describe reasoning, tutorial steps, or follow-up questions."
            )
            messages = [
                {"role": "system", "content": _CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            formatted_prompt = self._vlm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            with torch.no_grad():
                raw_caption = self._vlm.generate_response(
                    tensor, formatted_prompt, max_new_tokens=max_new_tokens
                )

            caption = clean_tool_output(raw_caption, query=user_task)

            elapsed = time.perf_counter() - t0

            return ToolResult(
                task="scene_captioning",
                success=True,
                answer=caption,
                confidence=None,
                confidence_type="not_available",
                visual_evidence=None,
                metadata={
                    "input_shape": list(tensor.shape),
                    "prompt_used": user_task,
                    "note": (
                        "Prototype output. Projector not yet trained on "
                        "BigEarthNet QA pairs. Caption quality will improve "
                        "after Phase 3 training."
                    ),
                },
                model_used=f"encoder={VISION_CHECKPOINT} | llm={LLM_NAME}",
                processing_time=elapsed,
            )

        except Exception as exc:
            return ToolResult(
                task="scene_captioning",
                success=False,
                answer="",
                confidence=None,
                confidence_type="not_available",
                visual_evidence=None,
                processing_time=time.perf_counter() - t0,
                error=str(exc),
            )
