"""
ui/app.py
==========
SatQuery AI -- Phase 4 Streamlit interface.

This process communicates with the Phase 3 FastAPI backend via HTTP.
No model weights are loaded here.

Start commands
--------------
  # Terminal 1 -- backend:
      python -m uvicorn backend.main:app --reload

  # Terminal 2 -- UI (from SatQuery_Prototype/):
      python -m streamlit run ui/app.py
"""
from __future__ import annotations

import io
import sys
import json
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

# -- Project root on sys.path -------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.config import (
    BACKEND_URL, TASK_OPTIONS, PAIR_TASKS, QUERY_REQUIRED_TASKS,
    EXAMPLE_QUERIES, CHANNEL_NAMES, RGB_PREVIEW_LABEL,
    APP_TITLE, APP_SUBTITLE, GROUNDING_NOT_IMPLEMENTED_MSG,
)
from ui.input_processor import (
    process_upload, make_second_preview, InputError, ProcessedInput,
)

# =============================================================================
# Page configuration (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title = "SatQuery AI",
    page_icon  = ":satellite:",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "Get Help":    None,
        "Report a bug": None,
        "About": (
            "**SatQuery AI** -- Multimodal remote-sensing VQA prototype.\n\n"
            "Phase 4 Streamlit UI fronting the Phase 3 FastAPI backend.\n"
            "Model: BigEarthNet MobileViT-S + Qwen2.5-0.5B-Instruct."
        ),
    },
)

# =============================================================================
# CSS -- premium dark satellite-analytics aesthetic
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Header ---- */
.satquery-header {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.5rem;
}
.satquery-title {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 40%, #a5d6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.satquery-subtitle {
    font-size: 0.95rem;
    color: #8b949e;
    letter-spacing: 0.02em;
    margin-top: 0.25rem;
}
.satquery-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background: #1f3a5f;
    border: 1px solid #2d6eba;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    color: #58a6ff;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

/* ---- Result card ---- */
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-card-header {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #8b949e;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.answer-text {
    font-size: 1.05rem;
    color: #e6edf3;
    line-height: 1.7;
    border-left: 3px solid #1f6feb;
    padding-left: 1rem;
    margin: 0.5rem 0 1.25rem 0;
}
.task-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.75rem;
}
.task-single_vqa        { background:#1f3a5f; color:#58a6ff; border: 1px solid #2d6eba; }
.task-captioning        { background:#1e3a2f; color:#3fb950; border: 1px solid #238636; }
.task-grounding         { background:#3b2d0f; color:#d29922; border: 1px solid #9e6a03; }
.task-change_detection  { background:#2d1f3d; color:#bc8cff; border: 1px solid #6e40c9; }
.task-optical_sar_fusion{ background:#1e2d3b; color:#79c0ff; border: 1px solid #1f6feb; }

/* ---- Evidence card ---- */
.evidence-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
}
.evidence-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.heuristic-warning {
    font-size: 0.78rem;
    color: #d29922;
    margin-top: 0.5rem;
    font-style: italic;
}

/* ---- Landing ---- */
.landing-section {
    text-align: center;
    padding: 3rem 1rem 1rem 1rem;
    color: #8b949e;
}
.landing-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}
.landing-title {
    font-size: 1.2rem;
    color: #e6edf3;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.example-query {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    font-size: 0.88rem;
    color: #8b949e;
    cursor: default;
    transition: border-color 0.2s;
}
.example-query:hover {
    border-color: #58a6ff;
    color: #e6edf3;
}

/* ---- Error / warning boxes ---- */
.error-box {
    background: #2d1f1f;
    border: 1px solid #f85149;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: #f85149;
    font-size: 0.9rem;
}
.warn-box {
    background: #2d280f;
    border: 1px solid #d29922;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: #d29922;
    font-size: 0.9rem;
}
.info-box {
    background: #1f2d3d;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: #79c0ff;
    font-size: 0.9rem;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
.sidebar-section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #58a6ff;
    text-transform: uppercase;
    padding: 0.5rem 0 0.25rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 0.75rem;
}

/* ---- Metric row ---- */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 0.75rem 0;
}
.metric-item {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    min-width: 120px;
}
.metric-label {
    font-size: 0.68rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- File info card ---- */
.file-info-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 0.5rem;
}
.file-info-card code {
    color: #79c0ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

/* Reduce default Streamlit padding */
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session state initialisation
# =============================================================================

def _init_state():
    defaults = {
        "result":      None,   # last SatQueryResponse dict
        "error_msg":   None,   # last human-readable error string
        "input1":      None,   # ProcessedInput for image 1
        "input2":      None,   # ProcessedInput for image 2 (pair tasks)
        "query":       "",
        "task_label":  list(TASK_OPTIONS.keys())[0],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# =============================================================================
# Header
# =============================================================================

st.markdown("""
<div class="satquery-header">
  <div class="satquery-title">SATQUERY AI</div>
  <div class="satquery-subtitle">
      Satellite Intelligence &bull; Multimodal Earth Observation
  </div>
  <div>
    <span class="satquery-badge">Phase 4</span>
    <span class="satquery-badge" style="margin-left:6px;">BigEarthNet + Qwen2.5</span>
    <span class="satquery-badge" style="margin-left:6px;">12-channel Sentinel</span>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:

    # -- Backend status -------------------------------------------------------
    st.markdown('<div class="sidebar-section-header">Backend</div>', unsafe_allow_html=True)
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("status") == "ok":
            st.success(f"API online  \u2014  localhost:8000", icon="\u2705")
        else:
            st.warning("API responded but status not OK.", icon="\u26a0\ufe0f")
    except Exception:
        st.error(
            "FastAPI backend not reachable.\n\n"
            "Start it with:\n"
            "```\npython -m uvicorn backend.main:app --reload\n```",
            icon="\u274c",
        )

    st.divider()

    # -- Data Input -----------------------------------------------------------
    st.markdown('<div class="sidebar-section-header">Data Input</div>', unsafe_allow_html=True)

    task_label = st.selectbox(
        "Analysis task",
        options = list(TASK_OPTIONS.keys()),
        index   = 0,
        key     = "task_label_widget",
        help    = (
            "Auto Detect: enter a natural-language query and the backend "
            "routes it automatically.\n\n"
            "Explicit tasks bypass the router."
        ),
    )
    selected_task: Optional[str] = TASK_OPTIONS[task_label]
    is_pair_task = selected_task in PAIR_TASKS

    # -- File upload(s) -------------------------------------------------------
    if is_pair_task:
        st.caption(f":information_source: **{task_label}** requires 2 images.")

    uploaded_1 = st.file_uploader(
        "Image 1" if is_pair_task else "Satellite image",
        type    = ["npy", "tif", "tiff"],
        key     = "upload_1",
        help    = "Upload a .npy [12, 120, 120] or a compatible 12-band GeoTIFF.",
    )
    uploaded_2 = None
    if is_pair_task:
        uploaded_2 = st.file_uploader(
            "Image 2 (after / SAR)",
            type    = ["npy", "tif", "tiff"],
            key     = "upload_2",
            help    = "Second image for bi-temporal or optical-SAR analysis.",
        )

    # -- Process uploads (validate, preview, cache in session state) ----------
    input1: Optional[ProcessedInput] = None
    input2: Optional[ProcessedInput] = None

    if uploaded_1:
        result1 = process_upload(uploaded_1)
        if isinstance(result1, InputError):
            st.error(f"**{result1.message}**\n\n{result1.detail}", icon="\u26a0\ufe0f")
        else:
            input1 = result1
            st.session_state["input1"] = result1

    if uploaded_2:
        result2 = process_upload(uploaded_2)
        if isinstance(result2, InputError):
            st.error(f"**{result2.message}**\n\n{result2.detail}", icon="\u26a0\ufe0f")
        else:
            input2 = result2
            st.session_state["input2"] = result2

    # -- File info display ---------------------------------------------------
    if input1:
        _info = input1.info
        st.markdown(
            f'<div class="file-info-card">'
            f'<b>{_info.get("filename","")}</b><br>'
            f'Type: <code>{_info.get("file_type","")}</code><br>'
            f'Shape: <code>{_info.get("shape","")}</code> &nbsp; '
            f'dtype: <code>{_info.get("dtype","")}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if input2:
        _info2 = input2.info
        st.markdown(
            f'<div class="file-info-card" style="margin-top:0.4rem;">'
            f'<b>{_info2.get("filename","")}</b><br>'
            f'Type: <code>{_info2.get("file_type","")}</code><br>'
            f'Shape: <code>{_info2.get("shape","")}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # -- Analysis controls ---------------------------------------------------
    st.markdown('<div class="sidebar-section-header">Analysis</div>', unsafe_allow_html=True)

    query_placeholder = (
        "e.g. What type of land cover is visible?" if not is_pair_task
        else "e.g. What changed between these observations?"
    )
    query = st.text_area(
        "Natural-language query",
        value       = "",
        placeholder = query_placeholder,
        height      = 90,
        key         = "query_widget",
        help        = (
            "Required for Single Image VQA and Visual Grounding.\n"
            "Optional for Captioning, Change Detection, Optical-SAR Fusion."
        ),
    )

    # -- Analyse button ------------------------------------------------------
    analyse_clicked = st.button(
        "Analyse",
        type      = "primary",
        use_container_width = True,
        disabled  = (input1 is None),
    )

# =============================================================================
# Analysis execution
# =============================================================================

if analyse_clicked:
    st.session_state["result"]    = None
    st.session_state["error_msg"] = None

    # -- Pre-flight checks ---------------------------------------------------
    if input1 is None:
        st.session_state["error_msg"] = "Please upload a satellite image first."
    elif is_pair_task and input2 is None:
        st.session_state["error_msg"] = f"{task_label} requires two images."
    elif selected_task in QUERY_REQUIRED_TASKS and not query.strip():
        st.session_state["error_msg"] = (
            f"{task_label} requires a natural-language query. "
            "Please enter a question."
        )
    else:
        # -- Build multipart request ----------------------------------------
        with st.spinner("Analysing... (model inference may take 10-60 s on CPU)"):
            try:
                if selected_task is None:
                    # Auto-detect: POST /query
                    files  = [("files", (input1.filename, input1.npy_bytes, "application/octet-stream"))]
                    if is_pair_task and input2:
                        files.append(("files", (input2.filename, input2.npy_bytes, "application/octet-stream")))
                    data   = {"query": query or "Describe this satellite image."}
                    resp   = requests.post(f"{BACKEND_URL}/query", files=files, data=data, timeout=120)
                else:
                    # Explicit task: POST /analyse
                    files  = [("files", (input1.filename, input1.npy_bytes, "application/octet-stream"))]
                    if is_pair_task and input2:
                        files.append(("files", (input2.filename, input2.npy_bytes, "application/octet-stream")))
                    data   = {"task": selected_task}
                    if query.strip():
                        data["query"] = query.strip()
                    resp   = requests.post(f"{BACKEND_URL}/analyse", files=files, data=data, timeout=120)

                resp.raise_for_status()
                body = resp.json()
                st.session_state["result"] = body

            except requests.ConnectionError:
                st.session_state["error_msg"] = (
                    "Cannot reach the backend API at localhost:8000. "
                    "Please start it with:\n\n"
                    "`python -m uvicorn backend.main:app --reload`"
                )
            except requests.Timeout:
                st.session_state["error_msg"] = (
                    "The request timed out (>120 s). "
                    "Model inference on CPU can be slow. "
                    "Try again or reduce load."
                )
            except requests.HTTPError as exc:
                st.session_state["error_msg"] = f"Backend returned an error: {exc}"
            except Exception as exc:
                st.session_state["error_msg"] = (
                    f"Unexpected error communicating with backend: {exc}"
                )


# =============================================================================
# Main content area
# =============================================================================

result    = st.session_state.get("result")
error_msg = st.session_state.get("error_msg")

# -- Error display -----------------------------------------------------------
if error_msg:
    st.error(error_msg, icon="\u274c")

# -- Image previews ----------------------------------------------------------
if input1 and input1.preview:
    preview_cols = [st.columns([1, 2, 1])[1]]
    if input2 and input2.preview:
        preview_cols = st.columns(2)

    with preview_cols[0]:
        st.image(
            input1.preview,
            caption = f"{RGB_PREVIEW_LABEL} \u2014 {input1.filename}",
            use_container_width = True,
        )
    if input2 and input2.preview and len(preview_cols) > 1:
        with preview_cols[1]:
            st.image(
                input2.preview,
                caption = f"{RGB_PREVIEW_LABEL} \u2014 {input2.filename}",
                use_container_width = True,
            )
elif input1 and input1.preview is None:
    st.caption("Preview unavailable for this input format.")

# -- Result card -------------------------------------------------------------
if result:
    task       = result.get("task", "unknown")
    success    = result.get("success", False)
    answer     = result.get("answer", "")
    confidence = result.get("confidence")
    conf_type  = result.get("confidence_type", "not_available")
    proc_time  = result.get("processing_time", 0.0)
    evidence   = result.get("visual_evidence")
    model_used = result.get("model_used", "")

    st.markdown("---")

    if not success:
        err_code = result.get("error", "unknown")
        err_msg  = result.get("message", "The backend reported a failure.")
        st.markdown(
            f'<div class="error-box"><b>Analysis failed</b> ({err_code})<br>{err_msg}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Task badge
        safe_cls = task.replace("-", "_")
        st.markdown(
            f'<span class="task-badge task-{safe_cls}">{task.replace("_", " ").title()}</span>',
            unsafe_allow_html=True,
        )

        # Answer
        st.markdown('<div class="result-card-header">Answer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-text">{answer}</div>',
            unsafe_allow_html=True,
        )

        # Metrics row: processing time + confidence (only if returned)
        col_a, col_b, col_c = st.columns([1, 1, 3])
        with col_a:
            st.metric("Processing time", f"{proc_time:.2f} s")
        with col_b:
            if isinstance(confidence, (int, float)):
                st.metric(
                    f"Confidence ({conf_type})",
                    f"{confidence:.3f}",
                    help=(
                        "heuristic = proxy metric, not calibrated probability\n"
                        if conf_type == "heuristic"
                        else None
                    ),
                )
            else:
                st.metric("Confidence", "N/A", help="This model produces no confidence score.")

        # Visual evidence
        if task == "grounding":
            st.markdown("---")
            st.warning(GROUNDING_NOT_IMPLEMENTED_MSG)

        elif task == "change_detection" and isinstance(evidence, dict):
            st.markdown("---")
            st.markdown('<div class="result-card-header">Change Evidence (heuristic)</div>', unsafe_allow_html=True)
            e_cols = st.columns(3)
            with e_cols[0]:
                changed = evidence.get("change_detected")
                label   = "\u2705 Change detected" if changed else "\u274c No change"
                st.metric("Detected", label)
            with e_cols[1]:
                mag = evidence.get("change_magnitude_l2")
                if mag is not None:
                    st.metric("L2 Magnitude", f"{mag:.4f}")
            st.caption(
                ":warning: This is a **heuristic** feature-level L2 distance, "
                "not a trained change-detection model output."
            )

        elif task == "optical_sar_fusion" and isinstance(evidence, dict):
            sim = evidence.get("modality_similarity")
            if sim is not None:
                st.markdown("---")
                st.markdown('<div class="result-card-header">Fusion Evidence</div>', unsafe_allow_html=True)
                st.metric("Modality Cosine Similarity", f"{sim:.4f}")
                st.caption("Similarity between optical and SAR feature embeddings (heuristic proxy).")

        # Audit info (expandable)
        audit = result.get("audit")
        if audit:
            with st.expander("Session audit record", expanded=False):
                st.markdown(f"**Session ID:** `{audit.get('session_id','')}`")
                st.markdown(f"**Timestamp:** {audit.get('timestamp','')}")
                st.markdown(f"**Model:** {audit.get('model_used','')}")
                st.markdown(f"**Tool:** {audit.get('tool','')}")
                st.markdown(f"**Routing reason:** {audit.get('routing_reason','')}")
                st.markdown(f"**Input files:** {audit.get('input_files','')}")

        if model_used:
            st.caption(f":gear: Model: {model_used}")

# -- Landing (no result yet) -------------------------------------------------
elif not error_msg and input1 is None:
    st.markdown("""
    <div class="landing-section">
      <div class="landing-icon">&#128752;</div>
      <div class="landing-title">
          Upload satellite data and ask a natural-language question<br>
          about the observed scene.
      </div>
      <p style="margin-top:0.5rem;font-size:0.9rem;">
          Supported input: <code>.npy</code> arrays [12, 120, 120] &bull;
          12-band GeoTIFF (120&times;120)
      </p>
      <p style="font-size:0.85rem;margin-top:1.5rem;color:#8b949e;">Example queries:</p>
    </div>
    """, unsafe_allow_html=True)

    # Example query chips
    eq_html = "".join(
        f'<span class="example-query">{q}</span>' for q in EXAMPLE_QUERIES
    )
    st.markdown(
        f'<div style="text-align:center;padding:0 2rem;">{eq_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("""
    <div style="display:flex;gap:2rem;justify-content:center;flex-wrap:wrap;margin-top:1rem;">
        <div style="text-align:center;color:#8b949e;">
            <div style="font-size:1.5rem;">&#128680;</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Single Image VQA</div>
        </div>
        <div style="text-align:center;color:#8b949e;">
            <div style="font-size:1.5rem;">&#128221;</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Scene Captioning</div>
        </div>
        <div style="text-align:center;color:#8b949e;">
            <div style="font-size:1.5rem;">&#128260;</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Change Detection</div>
        </div>
        <div style="text-align:center;color:#8b949e;">
            <div style="font-size:1.5rem;">&#128225;</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Optical-SAR Fusion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:2rem;padding:1rem;
         background:#161b22;border:1px solid #30363d;border-radius:8px;
         max-width:600px;margin-left:auto;margin-right:auto;">
        <span style="font-size:0.8rem;color:#8b949e;">
        <b style="color:#d29922;">&#9888; Important</b><br>
        The model requires exactly <b style="color:#e6edf3;">12 channels</b> at
        <b style="color:#e6edf3;">120&times;120 pixels</b>.
        Arbitrary TIFF files are inspected and an honest explanation is shown
        if preprocessing is needed. No channels are fabricated.
        </span>
    </div>
    """, unsafe_allow_html=True)
