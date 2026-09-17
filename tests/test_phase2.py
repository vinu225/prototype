# -*- coding: utf-8 -*-
"""
tests/test_phase2.py
=====================
Phase 2 validation test suite for SatQuery AI specialist tools.

Run from the SatQuery_Prototype/ root:
    python tests/test_phase2.py

All tests use SYNTHETIC tensors — no real datasets required.

Tests
-----
  TEST 1  : Single-image VQA
  TEST 2  : Scene captioning
  TEST 3  : Visual grounding (stub — verifies not_implemented, not fake boxes)
  TEST 4  : Bi-temporal change detection
  TEST 5  : Optical-SAR fusion

Prerequisites (same as Phase 1):
    pip install -r requirements.txt
    pip install git+https://github.com/sihechen0530/reben-training-scripts.git
"""

import sys
import os
import time
import torch

# Force UTF-8 stdout so Unicode characters don't crash cp1252 Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── configilm monkey-patch (must precede any reben_publication import) ──────
from configilm.ConfigILM import ILMConfiguration
if not hasattr(ILMConfiguration, "items"):
    ILMConfiguration.items = lambda self: self.__dict__.items()

from pipeline.preprocess    import make_dummy_batch
from tools.single_vqa       import SingleImageVQA
from tools.captioning       import SceneCaptioning
from tools.grounding        import VisualGrounding
from tools.change_detection import BiTemporalChangeDetection
from tools.optical_sar_fusion import OpticalSARFusion

# ── Helpers ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS   = "PASS ✓"
FAIL   = "FAIL ✗"

results: dict[str, str] = {}


def section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def check(condition: bool, label: str) -> bool:
    status = PASS if condition else FAIL
    print(f"    {label:45s} {status}")
    if not condition:
        print(f"    *** ASSERTION FAILED: {label}")
    return condition


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — Single-image VQA
# ═══════════════════════════════════════════════════════════════════════════
def test_single_vqa():
    section("TEST 1 — Single-image VQA")
    tool = SingleImageVQA(device=DEVICE)
    image = make_dummy_batch(batch_size=1, device=DEVICE)

    result = tool.run(
        image,
        query="What type of land cover is present in this satellite image?"
    )

    ok = all([
        check(result.success,                         "Tool executed successfully"),
        check(result.task == "single_image_vqa",      "Correct task label"),
        check(isinstance(result.answer, str),         "Answer is a string"),
        check(len(result.answer.strip()) > 0,         "Answer is non-empty"),
        check(result.confidence is None,              "Confidence is None (not fabricated)"),
        check(result.confidence_type == "not_available", "Confidence type correct"),
        check(result.processing_time > 0,             "Processing time recorded"),
        check(result.error is None,                   "No error"),
    ])

    status = PASS if ok else FAIL
    results["single_vqa"] = status
    print(f"\n  → Single VQA answer: {result.answer[:120]!r}")
    print(f"  → TEST 1: {status}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 — Scene captioning
# ═══════════════════════════════════════════════════════════════════════════
def test_captioning():
    section("TEST 2 — Scene Captioning")
    tool = SceneCaptioning(device=DEVICE)
    image = make_dummy_batch(batch_size=1, device=DEVICE, seed=7)

    result = tool.run(image)

    ok = all([
        check(result.success,                         "Tool executed successfully"),
        check(result.task == "scene_captioning",      "Correct task label"),
        check(isinstance(result.answer, str),         "Caption is a string"),
        check(len(result.answer.strip()) > 0,         "Caption is non-empty"),
        check(result.confidence is None,              "No fabricated confidence"),
        check(result.processing_time > 0,             "Processing time recorded"),
        check("prototype" in (result.metadata.get("note","")).lower()
              or "note" in result.metadata,           "Prototype note present in metadata"),
    ])

    status = PASS if ok else FAIL
    results["captioning"] = status
    print(f"\n  → Caption: {result.answer[:120]!r}")
    print(f"  → TEST 2: {status}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — Visual grounding (stub — must NOT fabricate boxes)
# ═══════════════════════════════════════════════════════════════════════════
def test_grounding():
    section("TEST 3 — Visual Grounding (stub)")
    tool = VisualGrounding(device=DEVICE)
    image = make_dummy_batch(batch_size=1, device=DEVICE)

    result = tool.run(image, query="Locate water bodies in this image.")

    ok = all([
        check(result.task == "visual_grounding",      "Correct task label"),
        check(not result.success,                     "success=False (not_implemented)"),
        check(result.error == "not_implemented",      "Error is 'not_implemented'"),
        check(result.visual_evidence is not None,     "visual_evidence dict present"),
        check(
            result.visual_evidence.get("boxes") is None,
            "Boxes are None — no fabricated coordinates"
        ),
        check(
            result.visual_evidence.get("status") == "not_implemented",
            "Status clearly marked not_implemented"
        ),
    ])

    status = PASS if ok else FAIL
    results["grounding"] = status
    print(f"\n  → Grounding answer: {result.answer[:80]!r}")
    print(f"  → TEST 3: {status}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Bi-temporal change detection
# ═══════════════════════════════════════════════════════════════════════════
def test_change_detection():
    section("TEST 4 — Bi-temporal Change Detection")
    tool = BiTemporalChangeDetection(device=DEVICE)

    before = make_dummy_batch(batch_size=1, device=DEVICE, seed=1)
    after  = make_dummy_batch(batch_size=1, device=DEVICE, seed=99)  # different seed → different features

    result = tool.run(
        before, after,
        query="What changed between these two satellite observations?"
    )

    ok = all([
        check(result.success,                              "Tool executed successfully"),
        check(result.task == "bitemporal_change_detection","Correct task label"),
        check(isinstance(result.answer, str),              "Answer is a string"),
        check(len(result.answer.strip()) > 0,              "Answer is non-empty"),
        check(result.confidence_type == "heuristic",       "Confidence type is heuristic"),
        check(result.confidence is not None,               "L2 heuristic score present"),
        check(isinstance(result.confidence, float),        "Confidence is a float"),
        check(result.visual_evidence is not None,          "Visual evidence dict present"),
        check(
            "change_magnitude_l2" in result.visual_evidence,
            "L2 magnitude reported"
        ),
        check(
            "change_detected" in result.visual_evidence,
            "change_detected flag present"
        ),
        check(
            result.visual_evidence.get("spatial_change_map") is None,
            "No fabricated spatial map"
        ),
    ])

    status = PASS if ok else FAIL
    results["change_detection"] = status
    mag = result.visual_evidence.get("change_magnitude_l2", "N/A") if result.visual_evidence else "N/A"
    det = result.visual_evidence.get("change_detected",     "N/A") if result.visual_evidence else "N/A"
    print(f"\n  → Change magnitude (L2): {mag}")
    print(f"  → Change detected:       {det}")
    print(f"  → Answer: {result.answer[:120]!r}")
    print(f"  → TEST 4: {status}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5 — Optical-SAR fusion
# ═══════════════════════════════════════════════════════════════════════════
def test_optical_sar_fusion():
    section("TEST 5 — Optical-SAR Fusion")
    tool = OpticalSARFusion(device=DEVICE)

    optical = make_dummy_batch(batch_size=1, device=DEVICE, seed=10)
    sar     = make_dummy_batch(batch_size=1, device=DEVICE, seed=20)

    result = tool.run(
        optical, sar,
        query=(
            "Using both optical and SAR data, "
            "identify the land-cover and any water bodies present."
        ),
    )

    ok = all([
        check(result.success,                         "Tool executed successfully"),
        check(result.task == "optical_sar_fusion",    "Correct task label"),
        check(isinstance(result.answer, str),         "Answer is a string"),
        check(len(result.answer.strip()) > 0,         "Answer is non-empty"),
        check(result.confidence is None,              "No fabricated confidence"),
        check(result.visual_evidence is not None,     "Visual evidence present"),
        check(
            "modality_cosine_similarity" in result.visual_evidence,
            "Cosine similarity reported"
        ),
        check(
            result.visual_evidence.get("fusion_strategy") == "element_wise_average",
            "Fusion strategy documented"
        ),
        check(result.processing_time > 0,             "Processing time recorded"),
    ])

    status = PASS if ok else FAIL
    results["optical_sar_fusion"] = status
    cos = result.visual_evidence.get("modality_cosine_similarity","N/A") if result.visual_evidence else "N/A"
    print(f"\n  → Modality cosine similarity: {cos}")
    print(f"  → Answer: {result.answer[:120]!r}")
    print(f"  → TEST 5: {status}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    total_t0 = time.perf_counter()

    print("=" * 50)
    print("  SATQUERY AI — PHASE 2 TEST SUITE")
    print("=" * 50)
    print(f"\nDevice : {DEVICE}")
    print("Loading shared model pool (one-time cost)…\n")

    test_single_vqa()
    test_captioning()
    test_grounding()
    test_change_detection()
    test_optical_sar_fusion()

    # ── Final report ─────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'=' * 50}")
    print("  PHASE 2 FINAL REPORT")
    print(f"{'=' * 50}")
    all_passed = True
    for test_name, status in results.items():
        print(f"  {test_name:35s} {status}")
        if "FAIL" in status:
            all_passed = False
    print(f"\n  Total suite time: {total_elapsed:.1f}s")
    print()
    if all_passed:
        print("  PHASE 2 PASSED — Awaiting manual approval.")
    else:
        print("  PHASE 2 FAILED — See above for details.")
    print(f"{'=' * 50}\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
