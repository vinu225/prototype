"""
models/vlm.py
=============
Unified multimodal VLM: SatelliteEncoder + VisionProjector + Qwen2.5-0.5B-Instruct.

SOURCE OF TRUTH: Colab notebook cells 34, 93, 101, 117, 122 (SatQueryVLM class).

Pipeline inside generate_response():
    1. SatelliteEncoder(image)          → [B, 640]
    2. VisionProjector(visual_feat)     → [B, 896]
    3. unsqueeze(1)                     → [B, 1, 896]   (one visual token)
    4. tokenize(query)  + embed         → [B, T, 896]
    5. cat([visual_token, text_embeds]) → [B, 1+T, 896] (multimodal sequence)
    6. attention_mask built as ones[1] | text_mask  (Phase 2 fix for warning)
    7. llm.generate(inputs_embeds=…, attention_mask=…) → token ids
    8. tokenizer.decode(…)              → text answer

Language model: Qwen/Qwen2.5-0.5B-Instruct
    — do NOT replace with Qwen2-VL or any other model.
    — hidden_size = 896  (matches projector output_dim)

Generation settings (verbatim from Colab cell 34):
    max_new_tokens=48, do_sample=True, temperature=0.7, top_p=0.9

Phase 2 additions (non-breaking):
    generate_from_visual_features() — accepts pre-computed [B,640] features,
    used by change_detection and optical_sar_fusion tools to avoid re-encoding.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.satellite_encoder import SatelliteEncoder
from models.vision_projector  import VisionProjector

# ── Model identifiers (must match Colab exactly) ────────────────────────────
VISION_CHECKPOINT = "BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.1.1"
LLM_NAME          = "Qwen/Qwen2.5-0.5B-Instruct"


class SatQueryVLM(nn.Module):
    """
    Full multimodal model.

    Parameters
    ----------
    vision_checkpoint : str
        HuggingFace repo id for the BigEarthNet MobileViT-S encoder.
    llm_name : str
        HuggingFace repo id for the causal language model.
    device : str
        'cuda' or 'cpu'.

    Trainable parameters
    --------------------
    Only self.projector — encoder and LLM are frozen.
    """

    def __init__(
        self,
        vision_checkpoint: str = VISION_CHECKPOINT,
        llm_name:          str = LLM_NAME,
        device:            str = "cpu",
    ):
        super().__init__()
        self.device = device

        # 1. Frozen visual encoder (MobileViT-S, BigEarthNet pretrained)
        self.vision_encoder = SatelliteEncoder(vision_checkpoint).to(device)

        # 2. Frozen LLM + tokenizer (Qwen2.5-0.5B-Instruct)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # from Colab

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
        for param in self.llm.parameters():
            param.requires_grad = False

        # 3. Trainable projector (640 → 1024 → llm_hidden_size)
        # Use llm.config.hidden_size to stay correct even if the LLM changes.
        llm_hidden_size = self.llm.config.hidden_size  # 896 for Qwen2.5-0.5B
        self.projector = VisionProjector(
            input_dim=640,
            hidden_dim=1024,
            output_dim=llm_hidden_size,
        ).to(device)

    # ------------------------------------------------------------------
    def _build_multimodal_embeds(
        self,
        satellite_img: torch.Tensor,
        question_text: str,
    ) -> tuple:
        """
        Shared logic used by both forward() and generate_response().

        Returns
        -------
        tuple(multimodal_emb, full_attention_mask)
            multimodal_emb     : [B, 1+T, 896]
            full_attention_mask: [B, 1+T]  — ones for visual token + text mask
        """
        # ── Visual branch ───────────────────────────────────────────────
        vision_features = self.vision_encoder(satellite_img)   # [B, 640]
        projected_emb   = self.projector(vision_features)      # [B, 896]
        projected_emb   = projected_emb.unsqueeze(1)           # [B, 1, 896]

        # ── Text branch ─────────────────────────────────────────────────
        text_inputs = self.tokenizer(
            question_text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        text_emb = self.llm.get_input_embeddings()(text_inputs.input_ids)
        # text_emb: [B, T, 896]

        # ── Build full attention mask ────────────────────────────────────
        # Phase 2 fix: the visual token always attends (mask = 1), then
        # append the text attention mask from the tokenizer.
        # This suppresses the 'attention_mask not set' warning that occurs
        # when pad_token == eos_token and no mask is supplied.
        B = projected_emb.shape[0]
        visual_mask = torch.ones(B, 1, dtype=torch.long, device=self.device)
        full_attention_mask = torch.cat(
            [visual_mask, text_inputs.attention_mask], dim=1
        )  # [B, 1+T]

        # ── Concatenate: [visual | text] ────────────────────────────────
        multimodal_emb = torch.cat([projected_emb, text_emb], dim=1)
        # multimodal_emb: [B, 1+T, 896]

        return multimodal_emb, full_attention_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        satellite_img: torch.Tensor,
        question_text: str,
    ):
        """
        Forward pass (used for training / gradient flow checks).
        Returns (CausalLMOutputWithPast, multimodal_embedding_shape).
        """
        multimodal_emb, full_attention_mask = self._build_multimodal_embeds(
            satellite_img, question_text
        )
        outputs = self.llm(
            inputs_embeds=multimodal_emb,
            attention_mask=full_attention_mask,
        )
        return outputs, multimodal_emb.shape

    # ------------------------------------------------------------------
    def generate_response(
        self,
        satellite_img: torch.Tensor,
        question_text: str,
        max_new_tokens: int = 48,
    ) -> str:
        """
        Autoregressive generation from a raw satellite image.

        Generation settings from Colab cell 34:
            do_sample=True, temperature=0.7, top_p=0.9
        """
        multimodal_emb, full_attention_mask = self._build_multimodal_embeds(
            satellite_img, question_text
        )

        generated_ids = self.llm.generate(
            inputs_embeds=multimodal_emb,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def generate_from_visual_features(
        self,
        visual_features: torch.Tensor,
        question_text: str,
        max_new_tokens: int = 48,
    ) -> str:
        """
        Autoregressive generation from PRE-COMPUTED 640-D visual features.

        Used by specialist tools (change_detection, optical_sar_fusion) that
        compute their own feature vectors (e.g. difference, averaged fusion)
        and need to pass them through the projector → LLM directly, without
        re-running the full image encoder.

        Parameters
        ----------
        visual_features : torch.Tensor
            Shape [B, 640].  Must already be on self.device.
        question_text : str
            Natural-language query.
        max_new_tokens : int
            Max tokens to generate.

        Returns
        -------
        str
            Decoded text answer.
        """
        assert visual_features.ndim == 2 and visual_features.shape[-1] == 640, (
            f"generate_from_visual_features expects [B, 640], "
            f"got {list(visual_features.shape)}"
        )

        # Project pre-computed features → LLM embedding space
        projected_emb = self.projector(visual_features).unsqueeze(1)  # [B, 1, 896]

        # Tokenize query text
        text_inputs = self.tokenizer(
            question_text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        text_emb = self.llm.get_input_embeddings()(text_inputs.input_ids)

        # Build attention mask
        B = projected_emb.shape[0]
        visual_mask = torch.ones(B, 1, dtype=torch.long, device=self.device)
        full_attention_mask = torch.cat(
            [visual_mask, text_inputs.attention_mask], dim=1
        )

        multimodal_emb = torch.cat([projected_emb, text_emb], dim=1)

        generated_ids = self.llm.generate(
            inputs_embeds=multimodal_emb,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
