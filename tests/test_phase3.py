# -*- coding: utf-8 -*-
"""
tests/test_phase3.py
=====================
Phase 3 validation test suite — FastAPI backend, routing, registry, validation,
audit logging, and integration tests.

Run from SatQuery_Prototype/ root:
    python tests/test_phase3.py

Design
------
- Uses FastAPI TestClient (no real HTTP server needed).
- Does NOT invoke the VLM for most tests — mocks are used where appropriate.
- TEST 8 and TEST 9 are integration tests that DO invoke the model (optional).
- All 9 tests must PASS before proceeding to Phase 4.
"""

import sys
import os
import io
import json
import time
import uuid
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# configilm monkey-patch (must precede any reben_publication import)
from configilm.ConfigILM import ILMConfiguration
if not hasattr(ILMConfiguration, "items"):
    ILMConfiguration.items = lambda self: self.__dict__.items()

PASS = "PASS [OK]"
FAIL = "FAIL [!!]"
results: dict[str, str] = {}


def section(title: str):
    print(f"\n{'─' * 56}")
    print(f"  {title}")
    print(f"{'─' * 56}")


def check(condition: bool, label: str) -> bool:
    status = PASS if condition else FAIL
    print(f"    {label:50s} {status}")
    if not condition:
        print(f"    *** ASSERTION FAILED: {label}")
    return condition


def make_valid_npy(channels=12, h=120, w=120, seed=0) -> bytes:
    """Produce a valid in-memory .npy file."""
    rng = np.random.default_rng(seed)
    arr = rng.random((channels, h, w), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def make_invalid_npy(channels=3, h=224, w=224) -> bytes:
    arr = np.zeros((channels, h, w), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# =============================================================================
# TEST 1 — Backend import
# =============================================================================
def test_backend_import():
    section("TEST 1 — Backend import")
    try:
        from backend.main import app
        ok = check(app is not None, "FastAPI app imported successfully")
        from backend.router import SatQueryRouter
        ok &= check(SatQueryRouter is not None, "SatQueryRouter imported")
        from backend.schemas import SatQueryResponse, ErrorResponse, HealthResponse
        ok &= check(True, "Pydantic schemas imported")
        from backend.audit import write_audit, generate_session_id
        ok &= check(True, "Audit module imported")
        from backend.tool_registry import get_registry
        ok &= check(True, "Tool registry imported")
    except Exception as exc:
        ok = False
        check(False, f"Import failed: {exc}")

    status = PASS if ok else FAIL
    results["backend_import"] = status
    print(f"\n  -> TEST 1: {status}")


# =============================================================================
# TEST 2 — Health endpoint
# =============================================================================
def test_health_endpoint():
    section("TEST 2 — Health endpoint (GET /health)")
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")

    ok = all([
        check(resp.status_code == 200,         "GET /health returns 200"),
        check(resp.json().get("status") == "ok", "status == 'ok'"),
        check(resp.json().get("phase") == 3,     "phase == 3"),
        check("service" in resp.json(),          "'service' key present"),
    ])

    status = PASS if ok else FAIL
    results["health_endpoint"] = status
    print(f"\n  -> TEST 2: {status}")


# =============================================================================
# TEST 3 — Tool registry
# =============================================================================
def test_tool_registry():
    section("TEST 3 — Tool registry")
    # Use mock to avoid loading the VLM in this unit test
    from unittest.mock import patch, MagicMock

    mock_single = MagicMock()
    mock_caption = MagicMock()
    mock_grounding = MagicMock()
    mock_change = MagicMock()
    mock_fusion = MagicMock()

    mock_registry = {
        "single_vqa":         mock_single,
        "captioning":         mock_caption,
        "grounding":          mock_grounding,
        "change_detection":   mock_change,
        "optical_sar_fusion": mock_fusion,
    }

    with patch("backend.tool_registry._registry", mock_registry):
        from backend.tool_registry import get_registry, get_tool
        reg = get_registry()

        ok = all([
            check("single_vqa"         in reg, "single_vqa registered"),
            check("captioning"         in reg, "captioning registered"),
            check("grounding"          in reg, "grounding registered"),
            check("change_detection"   in reg, "change_detection registered"),
            check("optical_sar_fusion" in reg, "optical_sar_fusion registered"),
            check(len(reg) == 5,               "Exactly 5 tools registered"),
        ])

    status = PASS if ok else FAIL
    results["tool_registry"] = status
    print(f"\n  -> TEST 3: {status}")


# =============================================================================
# TEST 4 — Query routing
# =============================================================================
def test_query_routing():
    section("TEST 4 — Query routing")
    from backend.router import SatQueryRouter

    router = SatQueryRouter()

    cases = [
        ("What type of land cover is present?",   "single_vqa"),
        ("Describe this image.",                   "captioning"),
        ("Where is the road?",                     "grounding"),
        ("What changed between these images?",     "change_detection"),
        ("Compare optical and SAR imagery.",       "optical_sar_fusion"),
        ("Is there vegetation in this image?",     "single_vqa"),
        ("Generate a scene description.",          "captioning"),
        ("Locate the buildings.",                  "grounding"),
        ("Has the area changed?",                  "change_detection"),
        ("Analyze both Sentinel-1 and Sentinel-2.", "optical_sar_fusion"),
    ]

    ok = True
    for query, expected_task in cases:
        try:
            result = router.route(query)
            got = result["task"]
            matched = got == expected_task
            short_q = repr(query)[:47]
            ok &= check(matched, f"{short_q} -> {expected_task}")
            if not matched:
                print(f"      (got {got!r})")
        except Exception as exc:
            ok &= check(False, f"{repr(query)[:45]} -> Exception: {exc}")

    status = PASS if ok else FAIL
    results["query_routing"] = status
    print(f"\n  -> TEST 4: {status}")


# =============================================================================
# TEST 5 — Invalid query handling
# =============================================================================
def test_invalid_query():
    section("TEST 5 — Invalid query handling")
    from backend.router import SatQueryRouter, UnknownQueryError

    router = SatQueryRouter()

    ok = True

    # Empty query
    try:
        router.route("")
        ok &= check(False, "Empty query should raise ValueError")
    except ValueError:
        ok &= check(True, "Empty query raises ValueError")
    except Exception as exc:
        ok &= check(False, f"Wrong exception type for empty query: {type(exc)}")

    # Gibberish query
    try:
        router.route("xkcd fnord asdf qwerty zxcvb")
        ok &= check(False, "Gibberish query should raise UnknownQueryError")
    except UnknownQueryError:
        ok &= check(True, "Gibberish query raises UnknownQueryError")
    except Exception as exc:
        ok &= check(False, f"Wrong exception for gibberish: {type(exc)}: {exc}")

    status = PASS if ok else FAIL
    results["invalid_query"] = status
    print(f"\n  -> TEST 5: {status}")


# =============================================================================
# TEST 6 — Input validation
# =============================================================================
def test_input_validation():
    section("TEST 6 — Input validation")
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False)

    # Valid: [12, 120, 120]
    valid_npy = make_valid_npy(12, 120, 120)
    # Invalid: [3, 224, 224]
    invalid_npy_3ch = make_invalid_npy(3, 224, 224)
    # Invalid: [12, 100, 100]
    invalid_npy_100 = make_invalid_npy(12, 100, 100)

    mock_result = MagicMock()
    mock_result.success        = True
    mock_result.task           = "captioning"
    mock_result.answer         = "Mock caption."
    mock_result.confidence     = None
    mock_result.confidence_type = "not_available"
    mock_result.visual_evidence = None
    mock_result.processing_time = 0.1
    mock_result.model_used     = "mock"
    mock_result.error          = None

    ok = True

    with patch("backend.tool_registry._registry", {
        "captioning": MagicMock(run=MagicMock(return_value=mock_result))
    }):
        # Valid upload
        resp_valid = client.post(
            "/analyse",
            data={"task": "captioning"},
            files=[("files", ("valid.npy", valid_npy, "application/octet-stream"))],
        )
        ok &= check(resp_valid.status_code == 200, "Valid [12,120,120] accepted (200)")

    # Invalid: wrong channels
    resp_3ch = client.post(
        "/analyse",
        data={"task": "captioning"},
        files=[("files", ("invalid.npy", invalid_npy_3ch, "application/octet-stream"))],
    )
    body_3ch = resp_3ch.json()
    ok &= check(resp_3ch.status_code == 200,        "3-channel returns 200 JSON response")
    ok &= check(body_3ch.get("success") is False,   "3-channel: success=False")
    ok &= check("wrong_channels" in body_3ch.get("error", "") or
                "channel" in body_3ch.get("message", "").lower(),
                "3-channel: error reports channel mismatch")

    # Invalid: wrong spatial
    resp_100 = client.post(
        "/analyse",
        data={"task": "captioning"},
        files=[("files", ("wrong_size.npy", invalid_npy_100, "application/octet-stream"))],
    )
    body_100 = resp_100.json()
    ok &= check(resp_100.status_code == 200,       "12ch/100x100 returns 200 JSON response")
    ok &= check(body_100.get("success") is False,  "12ch/100x100: success=False")
    ok &= check("wrong_spatial" in body_100.get("error", "") or
                "120" in body_100.get("message", ""),
                "12ch/100x100: error reports spatial mismatch")

    status = PASS if ok else FAIL
    results["input_validation"] = status
    print(f"\n  -> TEST 6: {status}")


# =============================================================================
# TEST 7 — Audit logging
# =============================================================================
def test_audit_logging():
    section("TEST 7 — Audit logging")
    from backend.audit import write_audit, generate_session_id

    sid = generate_session_id()
    ok = check(sid.startswith("session_"), "generate_session_id produces 'session_...' ID")

    audit_path = write_audit(
        session_id      = sid,
        query           = "Test audit query",
        task            = "single_vqa",
        tool            = "single_vqa",
        model_used      = "test_model",
        input_files     = ["test.npy"],
        input_metadata  = {"shape": [12, 120, 120]},
        routing_reason  = "Unit test",
        processing_time = 0.123,
        confidence      = None,
        confidence_type = "not_available",
        success         = True,
        error           = None,
    )

    ok &= check(audit_path.exists(), f"Audit file created: {audit_path.name}")

    with open(audit_path, encoding="utf-8") as f:
        record = json.load(f)

    ok &= check(record["session_id"]      == sid,          "session_id correct")
    ok &= check(record["task"]            == "single_vqa", "task correct")
    ok &= check(record["success"]         is True,         "success=True")
    ok &= check("timestamp" in record,                     "timestamp present")
    ok &= check("raw_image" not in record,                 "No raw image data in audit log")

    status = PASS if ok else FAIL
    results["audit_logging"] = status
    print(f"\n  -> TEST 7: {status}")


# =============================================================================
# TEST 8 — /query integration (mock VLM — no model invoked)
# =============================================================================
def test_query_integration():
    section("TEST 8 — /query integration (mocked tool)")
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False)

    mock_result = MagicMock()
    mock_result.success         = True
    mock_result.task            = "single_vqa"
    mock_result.answer          = "Agricultural cropland."
    mock_result.confidence      = None
    mock_result.confidence_type = "not_available"
    mock_result.visual_evidence = None
    mock_result.processing_time = 1.23
    mock_result.model_used      = "encoder=mock | llm=mock"
    mock_result.error           = None

    mock_tool = MagicMock()
    mock_tool.run.return_value = mock_result

    valid_npy = make_valid_npy()

    with patch("backend.tool_registry._registry", {"single_vqa": mock_tool}):
        resp = client.post(
            "/query",
            data={"query": "What type of land cover is present?"},
            files=[("files", ("patch.npy", valid_npy, "application/octet-stream"))],
        )

    body = resp.json()
    ok = all([
        check(resp.status_code == 200,                     "/query returns 200"),
        check(body.get("success") is True,                 "success=True"),
        check(body.get("task")    == "single_vqa",         "task == 'single_vqa'"),
        check(isinstance(body.get("answer"), str),         "answer is a string"),
        check(len(body.get("answer", "")) > 0,             "answer is non-empty"),
        check("audit" in body,                             "audit info present"),
        check("processing_time" in body,                   "processing_time present"),
        check(body.get("confidence") is None,              "confidence is None (not fabricated)"),
    ])

    status = PASS if ok else FAIL
    results["query_integration"] = status
    print(f"\n  -> TEST 8: {status}")


# =============================================================================
# TEST 9 — /analyse integration (mock VLM — explicit task)
# =============================================================================
def test_analyse_integration():
    section("TEST 9 — /analyse integration (mocked tool)")
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app, raise_server_exceptions=False)

    mock_result = MagicMock()
    mock_result.success         = True
    mock_result.task            = "change_detection"
    mock_result.answer          = "1. Apparent change present. 2. Vegetation loss."
    mock_result.confidence      = 0.84
    mock_result.confidence_type = "heuristic"
    mock_result.visual_evidence = {"change_detected": True, "change_magnitude_l2": 0.84}
    mock_result.processing_time = 2.5
    mock_result.model_used      = "encoder=mock | llm=mock"
    mock_result.error           = None

    mock_tool = MagicMock()
    mock_tool.run.return_value = mock_result

    before_npy = make_valid_npy(seed=1)
    after_npy  = make_valid_npy(seed=99)

    with patch("backend.tool_registry._registry", {"change_detection": mock_tool}):
        resp = client.post(
            "/analyse",
            data={"task": "change_detection", "query": "What changed?"},
            files=[
                ("files", ("before.npy", before_npy, "application/octet-stream")),
                ("files", ("after.npy",  after_npy,  "application/octet-stream")),
            ],
        )

    body = resp.json()
    ok = all([
        check(resp.status_code == 200,                          "/analyse returns 200"),
        check(body.get("success") is True,                      "success=True"),
        check(body.get("task") == "change_detection",           "task == 'change_detection'"),
        check(body.get("confidence_type") == "heuristic",       "confidence_type == 'heuristic'"),
        check(body.get("confidence") == 0.84,                   "heuristic confidence passed through"),
        check(body.get("visual_evidence") is not None,          "visual_evidence present"),
        check("change_detected" in (body.get("visual_evidence") or {}), "change_detected in evidence"),
        check("audit" in body,                                  "audit info present"),
    ])

    status = PASS if ok else FAIL
    results["analyse_integration"] = status
    print(f"\n  -> TEST 9: {status}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  SATQUERY AI — PHASE 3 TEST SUITE")
    print("=" * 56)
    t_start = time.perf_counter()

    test_backend_import()
    test_health_endpoint()
    test_tool_registry()
    test_query_routing()
    test_invalid_query()
    test_input_validation()
    test_audit_logging()
    test_query_integration()
    test_analyse_integration()

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 56)
    print("  PHASE 3 FINAL REPORT")
    print("=" * 56)
    all_pass = True
    for name, status in results.items():
        print(f"  {name:35s} {status}")
        if status != PASS:
            all_pass = False

    print(f"\n  Total suite time: {elapsed:.1f}s\n")
    if all_pass:
        print("  PHASE 3 PASSED — Awaiting manual approval.")
    else:
        print("  PHASE 3 FAILED — Fix failures above before proceeding.")
    print("=" * 56)
    sys.exit(0 if all_pass else 1)
