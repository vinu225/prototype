# SatQuery AI

**Multimodal remote-sensing Visual Question Answering prototype**

---

## 1. Project Name
**SatQuery AI** — A clean, phased implementation of a multimodal model that
answers natural-language questions about multispectral satellite imagery.

---

## 2. Current Objective
Wire the existing working Colab pipeline into standalone Python modules so the
full `[1,12,120,120] → encoder → projector → LLM → text` flow can be executed
locally without Jupyter, Colab, or any Colab-specific path hacks.

---

## 3. Phase 0 — Status ✅
Project scaffolding created:

```
SatQuery_Prototype/
├── app.py                      ← top-level entry point (placeholder)
├── requirements.txt            ← Phase 1 dependencies
├── README.md                   ← this file
├── .env.example                ← HF_TOKEN placeholder
│
├── models/                     ← neural-network modules
│   ├── __init__.py
│   ├── satellite_encoder.py    ← BigEarthNet MobileViT-S wrapper
│   ├── vision_projector.py     ← 640 → 1024 → 896 MLP
│   └── vlm.py                  ← Qwen2.5-0.5B-Instruct wrapper
│
├── pipeline/                   ← data flow
│   ├── __init__.py
│   ├── preprocess.py           ← numpy → [B,12,120,120] tensor
│   └── inference.py            ← analyze(image, query) → dict
│
├── agent/                      ← Phase 2+ placeholder
│   ├── __init__.py
│   ├── router.py
│   └── registry.py
│
├── tools/                      ← Phase 2+ placeholder
│   ├── __init__.py
│   └── single_vqa.py
│
├── ui/                         ← Phase 2+ placeholder
│   └── components.py
│
├── sample_data/
│   ├── single/                 ← place .npy files here
│   └── pairs/
│
├── logs/
└── tests/
    └── test_model.py           ← Phase 1 validation test
```

---

## 4. Phase 1 — Status ✅ (implemented, pending hardware run)
All Phase 1 model modules are implemented and faithfully reproduce the Colab
pipeline:

| Stage | Module | Status |
|-------|--------|--------|
| Preprocess | `pipeline/preprocess.py` | ✅ |
| Vision Encoder | `models/satellite_encoder.py` | ✅ |
| Vision Projector | `models/vision_projector.py` | ✅ |
| Language Model | `models/vlm.py` | ✅ |
| Inference | `pipeline/inference.py` | ✅ |
| Test | `tests/test_model.py` | ✅ |

**NOT implemented yet**: change detection, grounding, optical-SAR fusion,
agentic orchestration, Streamlit/Gradio UI, FastAPI.

---

## 5. Model Pipeline

```
12-channel Sentinel input  [B, 12, 120, 120]
              ↓
  BigEarthNet MobileViT-S
  (BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.1.1)
  — forward_features() → global-avg-pool —
              ↓
    640-D visual feature  [B, 640]
              ↓
   Vision Projector  (640 → Linear → 1024 → GELU → Linear → 896)
              ↓
    896-D token embedding  [B, 1, 896]
              ↓
  [visual token | text embeddings] → inputs_embeds
              ↓
  Qwen/Qwen2.5-0.5B-Instruct
  (AutoModelForCausalLM, frozen)
              ↓
       text response
```

**Channel order** (must be preserved, matches Colab):
`B02, B03, B04, B08, B05, B06, B07, B11, B12, B8A, VH, VV`

---

## 6. Installation

```bash
# 1. Create a Python 3.11 environment
python3.11 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install reben encoder package (not on PyPI)
pip install git+https://github.com/sihechen0530/reben-training-scripts.git

# 4. (Optional) set HuggingFace token for gated models
cp .env.example .env
# edit .env and add your HF_TOKEN
```

---

## 7. Run the Phase 1 Test

```bash
python tests/test_model.py
```

Expected output:

```
========================================
SATQUERY AI — PHASE 1 MODEL TEST
========================================
Input:
  Shape : [1, 12, 120, 120]
  dtype : float32

Vision Encoder:
  BigEarthNet MobileViT-S
  Feature dimension : 640   ✓

Vision Projector:
  640 → 1024 → 896
  Projected dimension : 896   ✓

Language Model:
  Qwen2.5-0.5B-Instruct

Generated response:
  <non-empty text>

Status:
  PHASE 1 PASSED
========================================
```

---

## 8. Awaiting

- [x] Phase 1 manual hardware approval — **APPROVED**
- [x] Phase 2 implemented — **APPROVED**
- [x] Phase 3 implemented — **9/9 TESTS PASSED — APPROVED**
- [x] Phase 4 implemented — **7/7 TESTS PASSED**
- [ ] Phase 4 manual approval — awaiting

---

## Phase 2 — Specialist Remote-Sensing Tools

### Available Specialist Tools

| Tool | Class | Status | Confidence |
|---|---|---|---|
| Single-image VQA | `SingleImageVQA` | ✅ Implemented | not_available |
| Scene Captioning | `SceneCaptioning` | ✅ Implemented | not_available |
| Visual Grounding | `VisualGrounding` | ⚠️ Stub (not_implemented) | not_available |
| Bi-temporal Change | `BiTemporalChangeDetection` | ✅ Implemented (prototype) | heuristic L2 |
| Optical-SAR Fusion | `OpticalSARFusion` | ✅ Implemented (prototype) | not_available |

### Shared Model Pool

All tools share a **single loaded instance** of the VLM via `models/model_pool.py`.
The ~988 MB of Qwen2.5-0.5B weights and the MobileViT-S encoder are loaded exactly
once per process, regardless of how many tools are invoked.

### Tool Input Requirements

All image inputs must be:
- `torch.Tensor` or `np.ndarray`
- Shape: `[C, H, W]` (single) or `[B, C, H, W]` (batched)
- `C = 12`, `H = W = 120`

Channel order: `B02 B03 B04 B08 B05 B06 B07 B11 B12 B8A VH VV`

### Prototype Method Notes (be honest)

**Single VQA / Captioning**: Uses the Phase 1 pipeline (projector not yet trained
on BigEarthNet QA pairs). Output quality improves after Phase 3 projector training.

**Change Detection**: Uses **feature-level L2 difference** between two 640-D
encoder outputs. This is a heuristic — it is NOT a trained Siamese change-detection
model (e.g. BIT-CD, ChangeMamba). The L2 magnitude is labelled `confidence_type=heuristic`.
No pixel-level change map is produced.

**Optical-SAR Fusion**: Uses **element-wise feature averaging** of two independently
encoded 640-D representations. Both images must be 12-channel (encoder constraint).
This is NOT a trained cross-modal attention fusion model.

**Visual Grounding**: Returns `success=False`, `boxes=None`, `status=not_implemented`.
No bounding boxes are fabricated. A future phase will integrate a dedicated grounding
model (e.g. Grounding DINO, OWL-ViT).

### Run Phase 2 Tests

```bash
# Run Phase 1 first to confirm it still passes
python tests/test_model.py

# Run Phase 2 tool tests
python tests/test_phase2.py
```

### Attention Mask Fix (Phase 2)

The Phase 1 warning:
```
The attention mask is not set and cannot be inferred from input
because pad token is same as eos token.
```
was resolved by explicitly building `full_attention_mask = [ones_for_visual_token | text_mask]`
and passing it to every `llm.generate()` and `llm()` call in `models/vlm.py`.


---

## Phase 4 — Streamlit UI

### Overview

A professional satellite-analysis web interface built with Streamlit.
The UI communicates with the Phase 3 FastAPI backend via HTTP —
**no model weights are loaded inside Streamlit**.

### Start Commands

Run both processes from `SatQuery_Prototype/`:

```bash
# Terminal 1 — FastAPI backend
python -m uvicorn backend.main:app --reload

# Terminal 2 — Streamlit UI
python -m streamlit run ui/app.py
```

UI opens at `http://localhost:8501`
API docs at `http://localhost:8000/docs`

### Upload Workflow

1. Upload one or two satellite files via the sidebar (`Data Input` section).
2. The UI validates the file immediately and shows shape / dtype information.
3. An RGB preview is displayed using channels B04/B03/B02 with p2–p98 stretch.
4. If the file is invalid, a clear human-readable error is shown — no crash.

### Natural-Language Query Workflow

1. Select **Auto Detect** (default) or an explicit task in the sidebar.
2. Type a natural-language question in the `Analysis` section.
3. Click **Analyse**.
4. The UI posts to `POST /query` (auto) or `POST /analyse` (explicit).
5. Results are displayed: task type, answer, processing time, confidence (if any),
   and visual evidence (if returned by the backend).

### Task Selection

| UI Label | Backend task | Images required |
|---|---|---|
| Auto Detect | `/query` endpoint (router decides) | 1 |
| Single Image VQA | `single_vqa` | 1 |
| Scene Captioning | `captioning` | 1 |
| Visual Grounding | `grounding` | 1 |
| Change Detection | `change_detection` | 2 |
| Optical-SAR Fusion | `optical_sar_fusion` | 2 |

### Backend Integration

```
Streamlit UI
  ↓ requests.post (HTTP multipart)
    FastAPI (backend/main.py)   ← already running on :8000
      ↓ SatQueryRouter / ToolRegistry / VLM
```

All validated images are re-serialized as `.npy` bytes and posted as
`multipart/form-data` to the existing `/query` or `/analyse` endpoints.

### Visualization

- **`.npy` files**: RGB preview using channels B04 (R), B03 (G), B02 (B)
  with percentile-stretched contrast. Clearly labelled as a preview.
- **GeoTIFF files**: inspected for band count and spatial dimensions;
  if compatible, pixel data read and visualized; otherwise an explanation is shown.
- **If preview unavailable**: `"Preview unavailable for this input format."` is shown;
  the application does not crash.

### Supported Input

The model requires **exactly**:

```
12 channels  (B02, B03, B04, B08, B05, B06, B07, B11, B12, B8A, VH, VV)
120 × 120 pixels spatial dimension
```

### TIFF Handling

The UI inspects TIFFs using `rasterio` (preferred) with PIL fallback.

| TIFF type | Behaviour |
|---|---|
| 1-band SAR (VV or VH only) | Explains that 11 more bands are required; shows preprocessing instructions |
| 2-band (VV + VH only) | Explains that 10 Sentinel-2 optical bands are missing |
| 12-band, 120×120 | Loaded and used directly |
| Wrong spatial size | Shows crop/resample instructions |
| Unreadable | Shows parse error message |

**No channels are fabricated. No silent reshaping occurs.**

### Honest Capability Report

| Capability | Status |
|---|---|
| Confidence scores | Only shown if returned by backend (never invented) |
| Bounding boxes (grounding) | `not_implemented` — clear UI message, no fake boxes |
| Change maps | Not produced — heuristic L2 feature distance only |
| Land-cover percentages | Not produced by this prototype |
| Pixel-level segmentation | Not implemented |

### UI Files

```
ui/
├── __init__.py          ← package init
├── config.py            ← constants (BACKEND_URL, TASK_OPTIONS, …) — no streamlit
├── input_processor.py   ← file validation / preview (no streamlit)
└── app.py               ← Streamlit rendering layer
```

### Run Phase 4 Tests

```bash
python tests/test_phase4.py
```
