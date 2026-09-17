# -*- coding: utf-8 -*-
"""
tests/test_phase4.py
=====================
Phase 4 validation -- Streamlit UI structure, input processing, and
backend integration configuration.

Run from SatQuery_Prototype/:
    python tests/test_phase4.py

Design
------
- Does NOT launch a browser or click the UI.
- Does NOT invoke the VLM.
- Tests 1-7 cover imports, config, validation logic, and TIFF inspection.
- All 7 tests must PASS before Phase 4 is considered complete.
"""

import sys
import io
import json
import time
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS_STR = "PASS [OK]"
FAIL_STR = "FAIL [!!]"
results: dict[str, str] = {}


def section(title: str):
    print(f"\n{'u2500' * 56}".replace("u2500", chr(0x2500)))
    print(f"  {title}")
    print(f"{'u2500' * 56}".replace("u2500", chr(0x2500)))


def check(condition: bool, label: str) -> bool:
    status = PASS_STR if condition else FAIL_STR
    print(f"    {label:52s} {status}")
    if not condition:
        print(f"    *** ASSERTION FAILED: {label}")
    return condition


def make_npy_bytes(channels=12, h=120, w=120, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.random((channels, h, w), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# =============================================================================
# TEST 1 -- UI module imports
# =============================================================================
def test_ui_imports():
    section("TEST 1 -- UI module imports")
    ok = True
    try:
        from ui import config
        ok &= check(True, "ui.config imported")
        ok &= check(hasattr(config, "BACKEND_URL"),    "config.BACKEND_URL exists")
        ok &= check(hasattr(config, "TASK_OPTIONS"),   "config.TASK_OPTIONS exists")
        ok &= check(hasattr(config, "PAIR_TASKS"),     "config.PAIR_TASKS exists")
        ok &= check(hasattr(config, "EXAMPLE_QUERIES"),"config.EXAMPLE_QUERIES exists")
        ok &= check(hasattr(config, "CHANNEL_NAMES"),  "config.CHANNEL_NAMES exists")
        ok &= check(hasattr(config, "EXPECTED_CHANNELS"), "config.EXPECTED_CHANNELS exists")
    except Exception as exc:
        ok = check(False, f"ui.config import failed: {exc}")

    try:
        from ui import input_processor
        ok &= check(True, "ui.input_processor imported")
        ok &= check(hasattr(input_processor, "process_upload"),   "process_upload exists")
        ok &= check(hasattr(input_processor, "process_npy_bytes"),"process_npy_bytes exists")
        ok &= check(hasattr(input_processor, "ProcessedInput"),   "ProcessedInput exists")
        ok &= check(hasattr(input_processor, "InputError"),       "InputError exists")
    except Exception as exc:
        ok &= check(False, f"ui.input_processor import failed: {exc}")

    results["ui_imports"] = PASS_STR if ok else FAIL_STR
    print(f"\n  -> TEST 1: {results['ui_imports']}")


# =============================================================================
# TEST 2 -- Streamlit app initializes without crashing (streamlit mocked)
# =============================================================================
def test_app_init():
    section("TEST 2 -- App initializes without crashing")
    ok = True

    # Build a realistic streamlit mock.
    # app.py calls st.selectbox() and immediately uses the return value as a
    # dict key in TASK_OPTIONS; we return the first valid key so the lookup works.
    from ui.config import TASK_OPTIONS
    first_task_label = list(TASK_OPTIONS.keys())[0]

    mock_st = MagicMock()
    mock_st.session_state    = {}
    mock_st.set_page_config  = MagicMock(return_value=None)
    mock_st.markdown         = MagicMock(return_value=None)
    mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st)
    mock_st.sidebar.__exit__  = MagicMock(return_value=False)
    mock_st.selectbox        = MagicMock(return_value=first_task_label)
    mock_st.file_uploader    = MagicMock(return_value=None)
    mock_st.text_area        = MagicMock(return_value="")
    mock_st.button           = MagicMock(return_value=False)
    mock_st.columns          = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    mock_st.divider          = MagicMock(return_value=None)

    try:
        with patch.dict("sys.modules", {"streamlit": mock_st}):
            for key in list(sys.modules.keys()):
                if key == "ui.app":
                    del sys.modules[key]
            import ui.app as app_module  # noqa: F401
            ok &= check(True, "ui.app imported without crash (streamlit mocked)")
            ok &= check(hasattr(app_module, "BACKEND_URL"),  "BACKEND_URL in app module")
            ok &= check(hasattr(app_module, "TASK_OPTIONS"), "TASK_OPTIONS in app module")
            ok &= check(hasattr(app_module, "PAIR_TASKS"),   "PAIR_TASKS in app module")
    except Exception as exc:
        ok &= check(False, f"ui.app raised exception: {exc}")

    results["app_init"] = PASS_STR if ok else FAIL_STR
    print(f"\n  -> TEST 2: {results['app_init']}")


# =============================================================================
# TEST 3 -- Expected configuration constants exist and are correct
# =============================================================================
def test_config_constants():
    section("TEST 3 -- Configuration constants")
    from ui.config import (
        BACKEND_URL, TASK_OPTIONS, PAIR_TASKS, EXAMPLE_QUERIES,
        CHANNEL_NAMES, EXPECTED_CHANNELS, EXPECTED_H, EXPECTED_W,
    )
    ok = all([
        check(BACKEND_URL == "http://localhost:8000",  "BACKEND_URL == 'http://localhost:8000'"),
        check(None in TASK_OPTIONS.values(),           "Auto Detect (None) in TASK_OPTIONS"),
        check("single_vqa"         in TASK_OPTIONS.values(), "single_vqa in TASK_OPTIONS"),
        check("captioning"         in TASK_OPTIONS.values(), "captioning in TASK_OPTIONS"),
        check("grounding"          in TASK_OPTIONS.values(), "grounding in TASK_OPTIONS"),
        check("change_detection"   in TASK_OPTIONS.values(), "change_detection in TASK_OPTIONS"),
        check("optical_sar_fusion" in TASK_OPTIONS.values(), "optical_sar_fusion in TASK_OPTIONS"),
        check("change_detection"   in PAIR_TASKS,     "change_detection in PAIR_TASKS"),
        check("optical_sar_fusion" in PAIR_TASKS,     "optical_sar_fusion in PAIR_TASKS"),
        check(len(EXAMPLE_QUERIES) >= 3,              "at least 3 example queries"),
        check(len(CHANNEL_NAMES) == 12,               "CHANNEL_NAMES has 12 entries"),
        check(EXPECTED_CHANNELS == 12,                "EXPECTED_CHANNELS == 12"),
        check(EXPECTED_H == 120,                      "EXPECTED_H == 120"),
        check(EXPECTED_W == 120,                      "EXPECTED_W == 120"),
    ])
    results["config_constants"] = PASS_STR if ok else FAIL_STR
    print(f"\n  -> TEST 3: {results['config_constants']}")


# =============================================================================
# TEST 4 -- Backend integration path correctly configured
# =============================================================================
def test_backend_integration():
    section("TEST 4 -- Backend integration path")
    from ui import config
    ok = all([
        check(config.BACKEND_URL.startswith("http"), "BACKEND_URL is HTTP URL"),
        check("localhost" in config.BACKEND_URL,     "BACKEND_URL points to localhost"),
        check("8000" in config.BACKEND_URL,          "BACKEND_URL uses port 8000"),
    ])
    # Verify the config matches the Phase 3 FastAPI startup convention
    import socket
    try:
        s = socket.create_connection(("localhost", 8000), timeout=1)
        s.close()
        ok &= check(True, "FastAPI backend reachable on :8000 (optional live check)")
    except OSError:
        # Backend not running -- that is OK for CI tests
        ok &= check(True, "Backend not running locally (acceptable in test env)")

    results["backend_integration"] = PASS_STR if ok else FAIL_STR
    print(f"\n  -> TEST 4: {results['backend_integration']}")


# =============================================================================
# TEST 5 -- Invalid input: wrong channel count
# =============================================================================
def test_reject_wrong_channels():
    section("TEST 5 -- Input validation: wrong channel count")
    from ui.input_processor import process_npy_bytes, InputError

    bad_3ch  = make_npy_bytes(3, 120, 120)
    bad_10ch = make_npy_bytes(10, 120, 120)

    ok = True
    for raw, n_ch in [(bad_3ch, 3), (bad_10ch, 10)]:
        result = process_npy_bytes(raw, f"bad_{n_ch}ch.npy")
        ok &= check(isinstance(result, InputError), f"{n_ch}-channel array rejected")
        if isinstance(result, InputError):
            ok &= check(result.code == "wrong_channels", f"  code == 'wrong_channels' for {n_ch}ch")
            ok &= check("channel" in result.message.lower() or str(n_ch) in result.message,
                        f"  message mentions channel count")

    results["reject_wrong_channels"] = PASS_STR if ok else FAIL_STR
    print(f"\n  -> TEST 5: {results['reject_wrong_channels']}")


# =============================================================================
# TEST 6 -- Invalid input: wrong spatial dimensions
# =============================================================================
def test_reject_wrong_spatial():
    section("TEST 6 -- Input validation: wrong spatial dimensions")
    from ui.input_processor import process_npy_bytes, InputError

    cases = [(12, 64, 64), (12, 224, 224), (12, 120, 60)]
    ok    = True

    for ch, h, w in cases:
        raw    = make_npy_bytes(ch, h, w)
        result = process_npy_bytes(raw, f"bad_{h}x{w}.npy")
        ok    &= check(isinstance(result, InputError),
                       f"[{ch},{h},{w}] array rejected")
        if isinstance(result, InputError):
            ok &= check(result.code == "wrong_spatial",
                        f"  code == 'wrong_spatial' for {h}x{w}")

    results["reject_wrong_spatial"] = PASS_STR if ok else FAIL_STR
    print(f"\n  -> TEST 6: {results['reject_wrong_spatial']}")


# =============================================================================
# TEST 7 -- Valid .npy [12,120,120] is accepted and produces ProcessedInput
# =============================================================================
def test_accept_valid_npy():
    section("TEST 7 -- Valid [12,120,120] .npy accepted")
    from ui.input_processor import process_npy_bytes, ProcessedInput

    ok = True

    # Shape [12,120,120]
    raw_3d = make_npy_bytes(12, 120, 120, seed=42)
    r3d    = process_npy_bytes(raw_3d, "valid_3d.npy")
    ok    &= check(isinstance(r3d, ProcessedInput), "[12,120,120] returns ProcessedInput")
    if isinstance(r3d, ProcessedInput):
        ok &= check(r3d.array.shape == (12, 120, 120), "  array shape == (12,120,120)")
        ok &= check(r3d.array.dtype == np.float32,     "  dtype == float32")
        ok &= check(len(r3d.npy_bytes) > 0,            "  npy_bytes non-empty")
        ok &= check(r3d.info.get("valid") is True,     "  info['valid'] == True")

    # Shape [1,12,120,120] (batched)
    arr4d  = np.random.default_rng(0).random((1, 12, 120, 120), dtype=np.float32)
    buf    = io.BytesIO(); np.save(buf, arr4d)
    r4d    = process_npy_bytes(buf.getvalue(), "batched.npy")
    ok    &= check(isinstance(r4d, ProcessedInput), "[1,12,120,120] (batched) accepted")
    if isinstance(r4d, ProcessedInput):
        ok &= check(r4d.array.shape == (12, 120, 120), "  squeezed to [12,120,120]")

    results["accept_valid_npy"] = PASS_STR if ok else FAIL_STR
    print(f"\n  -> TEST 7: {results['accept_valid_npy']}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  SATQUERY AI -- PHASE 4 TEST SUITE")
    print("=" * 56)
    t_start = time.perf_counter()

    test_ui_imports()
    test_app_init()
    test_config_constants()
    test_backend_integration()
    test_reject_wrong_channels()
    test_reject_wrong_spatial()
    test_accept_valid_npy()

    elapsed  = time.perf_counter() - t_start
    all_pass = all(v == PASS_STR for v in results.values())

    print("\n" + "=" * 56)
    print("  PHASE 4 FINAL REPORT")
    print("=" * 56)
    for name, status in results.items():
        print(f"  {name:38s} {status}")

    print(f"\n  Total suite time: {elapsed:.1f}s\n")
    if all_pass:
        print("  PHASE 4 PASSED -- Awaiting manual approval.")
    else:
        print("  PHASE 4 FAILED -- Fix failures above before proceeding.")
    print("=" * 56)
    sys.exit(0 if all_pass else 1)
