# -*- coding: utf-8 -*-
"""
tests/test_model.py
====================
Phase 1 validation test for SatQuery AI.

Run from the SatQuery_Prototype/ root:
    python tests/test_model.py

Prerequisites:
    pip install -r requirements.txt
    pip install git+https://github.com/sihechen0530/reben-training-scripts.git

This test verifies the EXACT pipeline from the Colab notebook:
    [1, 12, 120, 120]
        → BigEarthNet MobileViT-S  → [1, 640]
        → VisionProjector           → [1, 896]
        → Qwen2.5-0.5B-Instruct    → non-empty text

PASS/FAIL is printed clearly.  On failure the real exception is raised —
no fake successful output is produced.
"""

import sys
import os
import time

# Force UTF-8 stdout so Unicode characters don't crash cp1252 Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure the project root is on sys.path regardless of cwd ────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── configilm monkey-patch (must happen before any reben_publication import) ─
# Taken verbatim from Colab cells 20, 24, 91, 93 …
from configilm.ConfigILM import ILMConfiguration
if not hasattr(ILMConfiguration, "items"):
    ILMConfiguration.items = lambda self: self.__dict__.items()

import torch
from pipeline.preprocess   import make_dummy_batch
from models.satellite_encoder import SatelliteEncoder, VISION_FEATURE_DIM
from models.vision_projector  import VisionProjector, OUTPUT_DIM as PROJ_OUTPUT_DIM
from models.vlm               import SatQueryVLM, VISION_CHECKPOINT, LLM_NAME

# ── Constants ────────────────────────────────────────────────────────────────
EXPECTED_ENCODER_DIM  = 640
EXPECTED_PROJ_DIM     = 896
EXPECTED_INPUT_SHAPE  = [1, 12, 120, 120]


def separator():
    print("=" * 46)


def run_phase1_test():
    separator()
    print("  SATQUERY AI — PHASE 1 MODEL TEST")
    separator()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}\n")

    # ── Step 1: Create synthetic 12-channel input ────────────────────────
    print("Input:")
    sample = make_dummy_batch(batch_size=1, device=device)
    assert list(sample.shape) == EXPECTED_INPUT_SHAPE, (
        f"Input shape mismatch: got {list(sample.shape)}, "
        f"expected {EXPECTED_INPUT_SHAPE}"
    )
    print(f"  Shape : {list(sample.shape)}")
    print(f"  dtype : {sample.dtype}")

    # ── Step 2: Load encoder and extract visual features ─────────────────
    print("\nVision Encoder:")
    print(f"  Checkpoint : {VISION_CHECKPOINT}")
    t0 = time.perf_counter()

    encoder = SatelliteEncoder(checkpoint_name=VISION_CHECKPOINT)
    encoder = encoder.to(device)
    encoder.eval()

    with torch.no_grad():
        visual_features = encoder(sample)

    encoder_time = time.perf_counter() - t0

    assert visual_features.ndim == 2, (
        f"Encoder output must be 2-D [B, 640], got shape {list(visual_features.shape)}"
    )
    assert visual_features.shape[-1] == EXPECTED_ENCODER_DIM, (
        f"Encoder feature dim = {visual_features.shape[-1]}, "
        f"expected {EXPECTED_ENCODER_DIM}. "
        "If this is 19, the encoder returned classifier logits — "
        "check that forward_features() is being used."
    )
    print(f"  Feature dimension : {visual_features.shape[-1]}   [OK]")
    print(f"  Encoder time      : {encoder_time:.2f}s")

    # ── Step 3: Load projector and project visual features ───────────────
    print("\nVision Projector:")
    print(f"  Architecture : 640 → 1024 → {EXPECTED_PROJ_DIM}")

    projector = VisionProjector(input_dim=640, hidden_dim=1024,
                                output_dim=EXPECTED_PROJ_DIM)
    projector = projector.to(device)

    with torch.no_grad():
        projected = projector(visual_features)

    assert projected.shape[-1] == EXPECTED_PROJ_DIM, (
        f"Projector output dim = {projected.shape[-1]}, "
        f"expected {EXPECTED_PROJ_DIM}."
    )
    print(f"  Projected dimension : {projected.shape[-1]}   [OK]")

    # ── Step 4: Load VLM and generate a text response ────────────────────
    print("\nLanguage Model:")
    print(f"  Model : {LLM_NAME}")
    t1 = time.perf_counter()

    vlm = SatQueryVLM(
        vision_checkpoint=VISION_CHECKPOINT,
        llm_name=LLM_NAME,
        device=device,
    )
    vlm.eval()

    query = "What type of land cover is present in this satellite image?"
    with torch.no_grad():
        response = vlm.generate_response(sample, query, max_new_tokens=48)

    llm_time = time.perf_counter() - t1

    assert isinstance(response, str) and len(response.strip()) > 0, (
        "Language model returned an empty response — generation failed."
    )

    print(f"\nGenerated response:")
    print(f"  {response.strip()}")

    # ── Step 5: Print final report ───────────────────────────────────────
    total_time = encoder_time + llm_time

    separator()
    print("  Status:")
    print(f"    Input shape           : {list(sample.shape)}")
    print(f"    Encoder output dim    : {visual_features.shape[-1]}  (expected 640)  ✓")
    print(f"    Projector output dim  : {projected.shape[-1]}  (expected 896)  ✓")
    print(f"    Response non-empty    : True  ✓")
    print(f"    Encoder latency       : {encoder_time:.2f}s")
    print(f"    LLM latency           : {llm_time:.2f}s")
    print(f"    Total latency         : {total_time:.2f}s")
    print()
    print("  PHASE 1 PASSED")
    separator()


if __name__ == "__main__":
    run_phase1_test()
