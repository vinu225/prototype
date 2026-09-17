"""
tools/single_vqa.py
====================
Specialist tool: Single-image Visual Question Answering.

Accepts one 12-channel remote-sensing image and a natural-language question.
Returns a structured ToolResult with the generated answer.

Uses the shared model pool (encoder + projector + VLM) from Phase 1 —
no duplicate model loading.

Example queries
---------------
    "Describe the land cover in this image."
    "Is there a water body?"
    "What type of terrain is visible?"
    "Are there agricultural fields?"
"""

import time
import torch

from tools.base          import BaseSpecialistTool, ToolResult, clean_tool_output
from pipeline.preprocess import preprocess_image
from models.model_pool   import get_vlm
from models.vlm          import VISION_CHECKPOINT, LLM_NAME

_VQA_SYSTEM_PROMPT = (
    "You are a remote-sensing image analysis assistant.\n"
    "Analyze the supplied satellite image features and answer ONLY the user's question.\n"
    "Do not create multiple-choice options unless the user explicitly asks for them.\n"
    "Do not explain how to analyze the image.\n"
    "Do not restate the question.\n"
    "Do not provide instructions for another model.\n"
    "Return a concise answer grounded in the available image representation.\n"
    "If the evidence is insufficient, explicitly say that the image representation does not provide enough evidence."
)


class SingleImageVQA(BaseSpecialistTool):
    """
    Answer a natural-language question about one remote-sensing image.

    Input
    -----
    image : np.ndarray or torch.Tensor  [C,H,W] or [B,C,H,W]
             Must have C=12, H=W=120.
    query : str
             Natural-language question.

    Output
    ------
    ToolResult with:
        answer           — generated text response
        confidence       — None  (Qwen2.5-0.5B produces no calibrated score)
        confidence_type  — 'not_available'
    """

    TOOL_NAME = "single_image_vqa"

    def __init__(self, device: str = "cpu"):
        self.device = device
        # Shared VLM — loaded once, reused across all tools
        self._vlm = get_vlm(device=device)

    def run(
        self,
        image,
        query: str = "Describe the land cover visible in this satellite image.",
        max_new_tokens: int = 64,
    ) -> ToolResult:
        """
        Parameters
        ----------
        image : array-like
            12-channel satellite patch.
        query : str
            Question to answer.
        max_new_tokens : int
            Max tokens to generate.

        Returns
        -------
        ToolResult
        """
        t0 = time.perf_counter()
        try:
            # Validate + convert to tensor
            tensor = preprocess_image(image, device=self.device)  # [B,12,120,120]

            # Build structured prompt adhering to Qwen ChatML template
            user_content = (
                "Input Context: A 12-channel multispectral satellite observation is provided.\n\n"
                f"User Question:\n{query}\n\n"
                "Answer Constraints:\n"
                "Return a direct, concise answer grounded in the available image representation. "
                "Do not create options, tutorials, or instructions."
            )
            messages = [
                {"role": "system", "content": _VQA_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            formatted_prompt = self._vlm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Run full pipeline through shared VLM
            with torch.no_grad():
                raw_answer = self._vlm.generate_response(
                    tensor, formatted_prompt, max_new_tokens=max_new_tokens
                )

            # Clean output artifacts
            answer = clean_tool_output(raw_answer, query=query)

            elapsed = time.perf_counter() - t0

            return ToolResult(
                task="single_image_vqa",
                success=True,
                answer=answer,
                confidence=None,
                confidence_type="not_available",
                visual_evidence=None,
                metadata={
                    "input_shape":         list(tensor.shape),
                    "query":               query,
                    "visual_feature_dim":  640,
                    "projected_dim":       896,
                },
                model_used=f"encoder={VISION_CHECKPOINT} | llm={LLM_NAME}",
                processing_time=elapsed,
            )

        except Exception as exc:
            return ToolResult(
                task="single_image_vqa",
                success=False,
                answer="",
                confidence=None,
                confidence_type="not_available",
                visual_evidence=None,
                processing_time=time.perf_counter() - t0,
                error=str(exc),
            )
