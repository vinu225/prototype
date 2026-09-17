"""
ui/app.py
==========
SatQuery AI -- Multimodal Remote-Sensing Intelligence UI.

Supports:
  1. Pre-packaged Multimodal (.npy, 12 channels) -> Full VLM pipeline (/query or /analyse)
  2. Sentinel-1 SAR Pair (VH + VV GeoTIFFs / .npy) -> SAR Statistical Engine (/sar-analyse)
  3. Quick Demo Presets (Synthetic 12-channel samples from Phase 3 tests) -> Full VLM pipeline

Start commands:
  # Terminal 1 -- backend:
      python -m uvicorn backend.main:app --reload

  # Terminal 2 -- UI (from SatQuery_Prototype/):
      python -m streamlit run ui/app.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.config import (
    BACKEND_URL, TASK_OPTIONS, TASK_PROMPTS, PRESET_OPTIONS,
    PAIR_TASKS, QUERY_REQUIRED_TASKS,
    EXAMPLE_QUERIES, RGB_PREVIEW_LABEL, CHANNEL_NAMES,
    APP_TITLE, APP_SUBTITLE, GROUNDING_NOT_IMPLEMENTED_MSG,
)
from ui.input_processor import (
    process_upload, make_second_preview,
    process_sar_pair, get_demo_preset,
    SARPairInput, InputError, ProcessedInput,
)

# =============================================================================
# Helper: Safe Column & Status Wrapper for testing resilience
# =============================================================================

def _get_cols(spec):
    """Retrieve columns safely regardless of streamlit or unittest mock environment."""
    raw = st.columns(spec)
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


class _DummyStatus:
    def write(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): return False


def _safe_status(label: str):
    """Obtain an st.status context manager or a harmless dummy."""
    try:
        if hasattr(st, "status"):
            return st.status(label, expanded=True)
    except Exception:
        pass
    return _DummyStatus()

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title = "SatQuery AI — Multimodal Satellite Intelligence",
    page_icon  = "🛰️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "About": (
            "**SatQuery AI** -- Multimodal remote-sensing VQA prototype.\n\n"
            "Visual Encoder: BigEarthNet MobileViT-S (12-channel, 120x120).\n"
            "Language Model: Qwen2.5-0.5B-Instruct.\n"
            "SAR Engine: Sentinel-1 C-band Dual-Pol statistical analyser."
        ),
    },
)

# =============================================================================
# CSS Styling (Rich, modern dark theme)
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header */
.satquery-header {
    padding: 1.2rem 0 0.9rem 0;
    border-bottom: 1px solid #30363d;
    margin-bottom: 1.2rem;
}
.satquery-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 50%, #a5d6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
}
.satquery-subtitle {
    font-size: 0.95rem;
    color: #8b949e;
    margin-top: 0.2rem;
}
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 0.4rem 0.3rem 0 0;
    letter-spacing: 0.03em;
}
.badge-sar {
    background: #201705;
    border-color: #9e6a03;
    color: #f0b429;
}
.badge-model {
    background: #102319;
    border-color: #238636;
    color: #3fb950;
}

/* Cards */
.sq-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.1rem;
    margin-bottom: 1rem;
}
.sq-card-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #8b949e;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* Metadata pills */
.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.meta-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
}
.meta-label {
    font-size: 0.68rem;
    color: #8b949e;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.meta-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    color: #e6edf3;
    margin-top: 0.2rem;
}

/* Result styling */
.answer-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    font-size: 1.05rem;
    color: #f0f6fc;
    line-height: 1.7;
    margin: 0.8rem 0 1rem 0;
}
.task-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.8rem;
}
.task-vqa { background: #1f3a5f; color: #58a6ff; border: 1px solid #2d6eba; }
.task-sar { background: #2d1f0f; color: #f0b429; border: 1px solid #9e6a03; }
.task-caption { background: #1e3a2f; color: #3fb950; border: 1px solid #238636; }
.task-change { background: #2d1f3d; color: #bc8cff; border: 1px solid #6e40c9; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

/* Warnings */
.disclaimer-banner {
    background: #1a1400;
    border: 1px solid #9e6a03;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    color: #d29922;
    font-size: 0.85rem;
    line-height: 1.6;
    margin: 0.75rem 0;
}
.disclaimer-banner b { color: #f0b429; }

.block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State Initialization
# =============================================================================

def _init_state():
    defaults = {
        "result":          None,
        "sar_result":      None,
        "error_msg":       None,
        "input1":          None,   # ProcessedInput (12ch)
        "input2":          None,   # ProcessedInput (second 12ch image for pair tasks)
        "sar_pair":        None,   # SARPairInput (VH + VV)
        "query_text":      "",
        "active_preset":   None,
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
  <div class="satquery-title">SatQuery AI — Multimodal Satellite Intelligence</div>
  <div class="satquery-subtitle">Query optical, radar, and multimodal Earth observation data using natural language</div>
  <div>
    <span class="badge badge-model">MobileViT-S + Qwen2.5-0.5B</span>
    <span class="badge">12-Band Multimodal</span>
    <span class="badge badge-sar">Sentinel-1 SAR C-Band</span>
    <span class="badge">Phase 4 Interactive UI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar -- System Specs & Backend Connectivity
# =============================================================================

with st.sidebar:
    st.markdown("### Backend Status")
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if r.status_code == 200 and r.json().get("status") == "ok":
            st.success("FastAPI Online — localhost:8000", icon="✅")
        else:
            st.warning("Backend responding with unknown status", icon="⚠️")
    except Exception:
        st.error(
            "Backend offline.\n\n"
            "Start with:\n```bash\npython -m uvicorn backend.main:app --reload\n```",
            icon="❌",
        )

    st.divider()
    st.markdown("### System Specifications")
    st.markdown(
        "- **Vision Backbone:** MobileViT-S (BigEarthNet)\n"
        "- **Language Model:** Qwen2.5-0.5B-Instruct\n"
        "- **Input Resolution:** 120 × 120 pixels\n"
        "- **Ground Sampling:** 10 m / pixel (S2 / S1)\n"
        "- **Spatial Footprint:** 1.2 km × 1.2 km (1.44 km²)\n"
        "- **Supported Modalities:** Sentinel-2 MSI (10 bands) + Sentinel-1 SAR (VH, VV)"
    )

    st.divider()
    if st.button("🔄 Reset / Clear Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()

# =============================================================================
# Section B: Primary Interaction Area (Top/Center)
# =============================================================================

st.markdown("#### 1. Formulate Intelligence Query")

top_cols = _get_cols([3, 2])
col_q = top_cols[0]
col_task = top_cols[1] if len(top_cols) > 1 else top_cols[0]

# Task Selector Dropdown
ui_task_list = [
    "Auto-Detect from Query (Default)",
    "Flood / Water Detection",
    "Wildfire / Burn Scar",
    "Land Cover Classification",
    "Cloud / Shadow Detection",
    "Crop / Vegetation Health",
    "SAR Radar Analysis (Sentinel-1 Demo)",
    "Scene Captioning",
    "Change Detection",
    "Optical-SAR Fusion",
]

with col_task:
    selected_task_label = st.selectbox(
        "Analysis Task / Router Mode",
        options=ui_task_list,
        index=0,
        help="Select a specialist task or let SatQuery AI auto-detect the intent from your prompt.",
    )

backend_task_name = TASK_OPTIONS.get(selected_task_label)
is_sar_demo_task = (selected_task_label == "SAR Radar Analysis (Sentinel-1 Demo)")
is_pair_task = backend_task_name in PAIR_TASKS

# Default prompt suggestion if user switches task
default_q = ""
if selected_task_label in TASK_PROMPTS:
    default_q = TASK_PROMPTS[selected_task_label]

with col_q:
    saved_query = st.session_state.get("query_text", "")
    query_val = saved_query if (isinstance(saved_query, str) and saved_query.strip()) else default_q
    query_input = st.text_area(
        "Natural-Language Satellite Query",
        value=query_val,
        height=72,
        placeholder="e.g. Assess flood inundation using SAR imagery, or What land cover types are present?",
        help="Enter your natural language question about the satellite imagery.",
    )
    if isinstance(query_input, str):
        st.session_state["query_text"] = query_input

# Suggested query chips
st.markdown("<small style='color:#8b949e;'>Suggested queries: </small>", unsafe_allow_html=True)
q_sample_cols = _get_cols(3)
for idx, eq in enumerate(EXAMPLE_QUERIES[:3]):
    c = q_sample_cols[idx % len(q_sample_cols)]
    with c:
        if st.button(eq, key=f"chip_{idx}", use_container_width=True):
            st.session_state["query_text"] = eq
            st.rerun()

st.divider()

# =============================================================================
# Section C: Data Input Section (Clean Upload Options)
# =============================================================================

st.markdown("#### 2. Satellite Data Input")

input_option = st.radio(
    "Choose Input Option",
    options=[
        "Option 1: Multimodal (.npy, 12 channels)",
        "Option 2: Sentinel-1 SAR Pair (VH + VV GeoTIFFs)",
        "Option 3: Quick Demo Presets (Synthetic 12-ch)",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

option_str = str(input_option)

# -----------------------------------------------------------------------------
# Option 1: 12-Channel Multimodal (.npy or 12-band GeoTIFF)
# -----------------------------------------------------------------------------
if "Option 1" in option_str:
    st.caption("Upload full 12-channel Sentinel-2 + Sentinel-1 composite array `[12, 120, 120]`.")
    opt1_cols = _get_cols(2 if is_pair_task else [2, 1])
    with opt1_cols[0]:
        up_file1 = st.file_uploader(
            "Observation 1 (or Primary Observation)" if is_pair_task else "Multimodal Satellite File (.npy / .tif)",
            type=["npy", "tif", "tiff"],
            key="upload_opt1_file1",
        )
    up_file2 = None
    if is_pair_task and len(opt1_cols) > 1:
        with opt1_cols[1]:
            up_file2 = st.file_uploader(
                "Observation 2 (Temporal / Second Observation)",
                type=["npy", "tif", "tiff"],
                key="upload_opt1_file2",
            )

    if up_file1:
        r1 = process_upload(up_file1)
        if isinstance(r1, InputError):
            st.error(f"**{r1.message}**\n\n{r1.detail}", icon="⚠️")
        else:
            st.session_state["input1"] = r1
            st.session_state["sar_pair"] = None

    if up_file2:
        r2 = process_upload(up_file2)
        if isinstance(r2, InputError):
            st.error(f"**{r2.message}**\n\n{r2.detail}", icon="⚠️")
        else:
            st.session_state["input2"] = r2

# -----------------------------------------------------------------------------
# Option 2: Sentinel-1 SAR Pair (VH + VV GeoTIFFs)
# -----------------------------------------------------------------------------
elif "Option 2" in option_str:
    st.caption("Upload separate single-band Sentinel-1 VH and VV polarisations (`.tif` GeoTIFF or 2-D `.npy`).")
    sar_cols = _get_cols(2)
    with sar_cols[0]:
        vh_up = st.file_uploader(
            "Sentinel-1 VH Polarisation (.tif / .npy)",
            type=["tif", "tiff", "npy"],
            key="upload_opt2_vh",
        )
    with sar_cols[1 if len(sar_cols) > 1 else 0]:
        vv_up = st.file_uploader(
            "Sentinel-1 VV Polarisation (.tif / .npy)",
            type=["tif", "tiff", "npy"],
            key="upload_opt2_vv",
        )

    if vh_up and vv_up:
        sar_res = process_sar_pair(vh_up, vv_up)
        if isinstance(sar_res, InputError):
            st.error(f"**{sar_res.message}**\n\n{sar_res.detail}", icon="⚠️")
        else:
            st.session_state["sar_pair"] = sar_res
            st.session_state["input1"] = None
            st.session_state["input2"] = None

# -----------------------------------------------------------------------------
# Option 3: Quick Demo Presets (Synthetic 12-channel)
# -----------------------------------------------------------------------------
else:
    st.caption("Load curated synthetic 12-channel benchmarks for instant demonstration without file uploads.")
    p_cols = _get_cols(3)
    presets_def = [
        ("🌾 Agricultural Scene", "agriculture", "High NIR & Red Edge, moderate SAR backscatter, dense vegetation."),
        ("🌊 Wetland / Water Scene", "wetland", "High Green/Blue, strong NIR absorption, low specular SAR backscatter."),
        ("🏢 Urban / Built-up Scene", "urban", "High SWIR reflectance, high VH/VV double-bounce SAR return."),
    ]
    for idx, (label, key, desc) in enumerate(presets_def):
        c = p_cols[idx % len(p_cols)]
        with c:
            st.markdown(f"**{label}**")
            st.caption(desc)
            if st.button(f"Load {label.split()[1]} Preset", key=f"btn_preset_{key}", use_container_width=True):
                preset_data = get_demo_preset(key)
                st.session_state["input1"] = preset_data
                st.session_state["input2"] = None
                st.session_state["sar_pair"] = None
                st.session_state["active_preset"] = key
                st.success(f"Loaded {label} (12 channels, 120×120)", icon="✅")

# -----------------------------------------------------------------------------
# File Metadata Card (Shown whenever an input is loaded)
# -----------------------------------------------------------------------------
active_input = st.session_state.get("input1")
active_sar = st.session_state.get("sar_pair")

if active_input:
    info = active_input.info
    st.markdown(f"""
    <div class="sq-card">
      <div class="sq-card-header">Loaded Dataset Metadata — {info.get('filename')}</div>
      <div class="meta-grid">
        <div class="meta-item"><div class="meta-label">Format / Source</div><div class="meta-val">{info.get('file_type')}</div></div>
        <div class="meta-item"><div class="meta-label">Dimensions</div><div class="meta-val">{info.get('shape')} px</div></div>
        <div class="meta-item"><div class="meta-label">Bands</div><div class="meta-val">12 Channels (10 Optical + 2 SAR)</div></div>
        <div class="meta-item"><div class="meta-label">Ground Footprint</div><div class="meta-val">1.2 km × 1.2 km (1.44 km²)</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif active_sar:
    info = active_sar.info
    st.markdown(f"""
    <div class="sq-card">
      <div class="sq-card-header">Loaded SAR Pair Metadata — Dual Polarisation</div>
      <div class="meta-grid">
        <div class="meta-item"><div class="meta-label">VH File</div><div class="meta-val">{info.get('vh_filename')} ({info.get('vh_shape')})</div></div>
        <div class="meta-item"><div class="meta-label">VV File</div><div class="meta-val">{info.get('vv_filename')} ({info.get('vv_shape')})</div></div>
        <div class="meta-item"><div class="meta-label">Polarisation / Mode</div><div class="meta-val">Sentinel-1 C-Band (VH + VV)</div></div>
        <div class="meta-item"><div class="meta-label">Spatial Resolution</div><div class="meta-val">{info.get('spatial')} px (10m GSD)</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =============================================================================
# Section D: Run Analysis Button & Dynamic Status
# =============================================================================

can_run = (active_input is not None) or (active_sar is not None)
run_clicked = st.button(
    "🚀 Run Analysis / Execute Query",
    type="primary",
    use_container_width=True,
    disabled=not can_run,
    help="Execute the multimodal satellite analysis pipeline." if can_run else "Please upload data or select a preset first.",
)

# Execution Logic
if run_clicked:
    st.session_state["result"] = None
    st.session_state["sar_result"] = None
    st.session_state["error_msg"] = None

    # Routing determination
    should_run_sar_engine = is_sar_demo_task or (active_sar is not None)

    # 1. Sentinel-1 SAR Engine Route
    if should_run_sar_engine:
        if active_sar is None:
            st.session_state["error_msg"] = "Please upload both Sentinel-1 VH and VV TIFF files in Option 2."
        else:
            with _safe_status("Executing Sentinel-1 SAR Analysis...") as status_box:
                try:
                    status_box.write("📡 Reading C-band dual-polarisation arrays...")
                    time.sleep(0.3)
                    status_box.write("📊 Computing backscatter statistics, cross-ratio, and correlation...")

                    files = [
                        ("files", (f"{Path(active_sar.vh_filename).stem}.npy", active_sar.vh_npy_bytes, "application/octet-stream")),
                        ("files", (f"{Path(active_sar.vv_filename).stem}.npy", active_sar.vv_npy_bytes, "application/octet-stream")),
                    ]
                    resp = requests.post(f"{BACKEND_URL}/sar-analyse", files=files, timeout=30)
                    resp.raise_for_status()

                    status_box.write("🧠 Synthesizing heuristic surface roughness & moisture interpretation...")
                    time.sleep(0.2)
                    status_box.update(label="SAR Analysis Complete!", state="complete")
                    st.session_state["sar_result"] = resp.json()

                except requests.ConnectionError:
                    status_box.update(label="Backend Unreachable", state="error")
                    st.session_state["error_msg"] = f"Cannot reach backend at {BACKEND_URL}. Ensure uvicorn is running."
                except Exception as exc:
                    status_box.update(label="Analysis Failed", state="error")
                    st.session_state["error_msg"] = f"SAR Analysis failed: {exc}"

    # 2. Full 12-Channel Multimodal Route
    else:
        q_str = str(st.session_state.get("query_text", "")).strip()
        if active_input is None:
            st.session_state["error_msg"] = "Please upload a 12-channel satellite image or choose a demo preset."
        elif is_pair_task and st.session_state.get("input2") is None:
            st.session_state["error_msg"] = f"Task '{selected_task_label}' requires two input images."
        elif backend_task_name in QUERY_REQUIRED_TASKS and not q_str:
            st.session_state["error_msg"] = f"Task '{selected_task_label}' requires a natural language query."
        else:
            with _safe_status("Executing SatQuery AI Pipeline...") as status_box:
                try:
                    status_box.write("🛰️ Extracting multimodal patch tokens via MobileViT-S visual backbone...")
                    files = [("files", (active_input.filename, active_input.npy_bytes, "application/octet-stream"))]
                    if is_pair_task and st.session_state.get("input2"):
                        inp2 = st.session_state["input2"]
                        files.append(("files", (inp2.filename, inp2.npy_bytes, "application/octet-stream")))

                    time.sleep(0.4)
                    status_box.write("🔀 Routing to domain specialist tool...")

                    query_to_send = q_str or "Describe this satellite scene."
                    if backend_task_name is None:
                        resp = requests.post(
                            f"{BACKEND_URL}/query",
                            files=files,
                            data={"query": query_to_send},
                            timeout=120,
                        )
                    else:
                        data = {"task": backend_task_name, "query": query_to_send}
                        resp = requests.post(f"{BACKEND_URL}/analyse", files=files, data=data, timeout=120)

                    resp.raise_for_status()
                    status_box.write("✨ Synthesizing intelligence response with Qwen2.5-0.5B...")
                    time.sleep(0.3)
                    status_box.update(label="SatQuery Analysis Complete!", state="complete")
                    st.session_state["result"] = resp.json()

                except requests.ConnectionError:
                    status_box.update(label="Backend Unreachable", state="error")
                    st.session_state["error_msg"] = f"Cannot reach backend at {BACKEND_URL}. Ensure uvicorn is running."
                except Exception as exc:
                    status_box.update(label="Inference Failed", state="error")
                    st.session_state["error_msg"] = f"Pipeline execution error: {exc}"

# Display Errors if any
error_msg = st.session_state.get("error_msg")
if error_msg:
    st.error(error_msg, icon="❌")

# =============================================================================
# Section E: Results Area (Two-Column Presentation)
# =============================================================================

res_12ch = st.session_state.get("result")
res_sar  = st.session_state.get("sar_result")

if res_12ch or res_sar:
    st.markdown("### 3. Intelligence Analysis Results")
    res_cols = _get_cols([1, 1])
    res_col_left = res_cols[0]
    res_col_right = res_cols[1] if len(res_cols) > 1 else res_cols[0]

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Visual Preview
    # -------------------------------------------------------------------------
    with res_col_left:
        st.markdown('<div class="sq-card-header">Visual Evidence & Preview</div>', unsafe_allow_html=True)

        # 12-channel preview
        if active_input and active_input.preview:
            st.image(
                active_input.preview,
                caption=f"True-Color RGB Composite (B04-B03-B02 stretched) — {active_input.filename}",
                use_container_width=True,
            )
            if st.session_state.get("input2") and st.session_state["input2"].preview:
                st.image(
                    st.session_state["input2"].preview,
                    caption=f"Temporal Observation 2 — {st.session_state['input2'].filename}",
                    use_container_width=True,
                )

        # SAR preview
        elif active_sar:
            if active_sar.composite_preview:
                st.image(
                    active_sar.composite_preview,
                    caption="False-Color Dual-Pol Composite (R=VH, G=VV, B=VH/VV Ratio)",
                    use_container_width=True,
                )
            sar_sub_cols = _get_cols(2)
            with sar_sub_cols[0]:
                if active_sar.vh_preview:
                    st.image(active_sar.vh_preview, caption="VH Polarisation (Grayscale)", use_container_width=True)
            with sar_sub_cols[1 if len(sar_sub_cols) > 1 else 0]:
                if active_sar.vv_preview:
                    st.image(active_sar.vv_preview, caption="VV Polarisation (Grayscale)", use_container_width=True)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Model Intelligence Response
    # -------------------------------------------------------------------------
    with res_col_right:
        st.markdown('<div class="sq-card-header">Model Intelligence Output</div>', unsafe_allow_html=True)

        # 1. Handling 12-Channel Result
        if res_12ch:
            success = res_12ch.get("success", False)
            task_resp = res_12ch.get("task", "unknown")
            answer = res_12ch.get("answer", "")
            confidence = res_12ch.get("confidence")
            conf_type = res_12ch.get("confidence_type", "not_available")
            proc_time = res_12ch.get("processing_time", 0.0)

            if not success:
                st.error(f"Analysis Failed: {res_12ch.get('message', res_12ch.get('error'))}", icon="❌")
            else:
                st.markdown(
                    f'<span class="task-pill task-vqa">Task: {task_resp.replace("_", " ").title()}</span>',
                    unsafe_allow_html=True,
                )

                # Key Metrics Row
                m_cols = _get_cols(3)
                m_cols[0].metric("Processing Time", f"{proc_time:.2f} s")
                if isinstance(confidence, (int, float)):
                    m_cols[1 if len(m_cols) > 1 else 0].metric("Confidence", f"{confidence:.3f}", help=f"Type: {conf_type}")
                else:
                    m_cols[1 if len(m_cols) > 1 else 0].metric("Confidence", "N/A", help="No model logits produced.")

                # Estimate spectral indices from array if available
                if active_input and active_input.array is not None:
                    arr = active_input.array
                    nir, red, green = arr[3], arr[2], arr[1]
                    ndvi = float(np.mean((nir - red) / (nir + red + 1e-6)))
                    m_cols[2 if len(m_cols) > 2 else 0].metric("NDVI Proxy", f"{ndvi:.2f}")

                # Natural language answer
                st.markdown("**Intelligence Finding:**")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                if task_resp == "grounding":
                    st.warning(GROUNDING_NOT_IMPLEMENTED_MSG)

        # 2. Handling SAR Result
        elif res_sar:
            success = res_sar.get("success", False)
            proc_time = res_sar.get("processing_time", 0.0)
            interp = res_sar.get("interpretation", "")
            corr = res_sar.get("correlation")
            vh_s = res_sar.get("vh_stats", {})
            vv_s = res_sar.get("vv_stats", {})
            rat_s = res_sar.get("ratio_stats", {})

            if not success:
                st.error(f"SAR Analysis Failed: {res_sar.get('error')}", icon="❌")
            else:
                st.markdown(
                    '<span class="task-pill task-sar">Specialist: SAR Statistical Engine (No VLM)</span>',
                    unsafe_allow_html=True,
                )

                # Key Metrics
                sm_cols = _get_cols(3)
                sm_cols[0].metric("Latency", f"{proc_time:.3f} s")
                if corr is not None:
                    sm_cols[1 if len(sm_cols) > 1 else 0].metric("VH/VV Correlation", f"{corr:.3f}")
                if rat_s and rat_s.get("mean") is not None:
                    sm_cols[2 if len(sm_cols) > 2 else 0].metric("Mean VH/VV Ratio", f"{rat_s['mean']:.3f}")

                # Channel backscatter metrics
                st.markdown("**Polarisation Backscatter Stats:**")
                sc_cols = _get_cols(2)
                with sc_cols[0]:
                    st.markdown(f"""
                    <div class="meta-item">
                      <div class="meta-label">VH Polarisation</div>
                      <div class="meta-val">Mean: {vh_s.get('mean', 0):.4f} &nbsp;|&nbsp; Std: {vh_s.get('std', 0):.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with sc_cols[1 if len(sc_cols) > 1 else 0]:
                    st.markdown(f"""
                    <div class="meta-item">
                      <div class="meta-label">VV Polarisation</div>
                      <div class="meta-val">Mean: {vv_s.get('mean', 0):.4f} &nbsp;|&nbsp; Std: {vv_s.get('std', 0):.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Heuristic interpretation
                st.markdown("**Heuristic Interpretation:**")
                st.markdown(f'<div class="answer-box">{interp}</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="disclaimer-banner">
                  <b>⚠️ Sentinel-1 SAR Demonstration Notice:</b><br>
                  {res_sar.get('warning', 'SAR-only statistical evaluation mode.')}
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # Section F: Technical Details Collapsible (Accordion)
    # =========================================================================
    with st.expander("🛠️ Technical Details & System Audit Record", expanded=False):
        t_cols = _get_cols(2)
        with t_cols[0]:
            st.markdown("**Band Mapping (BigEarthNet 12-Channel Standard):**")
            st.code(
                "0: B02 (Blue, 490nm)\n"
                "1: B03 (Green, 560nm)\n"
                "2: B04 (Red, 665nm)\n"
                "3: B08 (NIR, 842nm)\n"
                "4: B05 (Red Edge 1, 705nm)\n"
                "5: B06 (Red Edge 2, 740nm)\n"
                "6: B07 (Red Edge 3, 783nm)\n"
                "7: B11 (SWIR 1, 1610nm)\n"
                "8: B12 (SWIR 2, 2190nm)\n"
                "9: B8A (Narrow NIR, 865nm)\n"
                "10: VH (Sentinel-1 SAR cross-polarisation)\n"
                "11: VV (Sentinel-1 SAR co-polarisation)",
                language="yaml",
            )
        with t_cols[1 if len(t_cols) > 1 else 0]:
            st.markdown("**Raw JSON Response Payload:**")
            st.json(res_12ch or res_sar)
