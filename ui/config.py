"""
ui/config.py
=============
SatQuery AI -- Phase 4 UI constants.

No streamlit imports here -- this module is importable by tests
and by any script that needs the configuration values.
"""
from __future__ import annotations

# -- Backend --------------------------------------------------------------------

BACKEND_URL: str = "http://localhost:8000"

# -- Task options ---------------------------------------------------------------

#: Mapping from display label -> backend task name (None = auto-detect via /query)
TASK_OPTIONS: dict[str, "str | None"] = {
    "Auto Detect":         None,
    "Single Image VQA":    "single_vqa",
    "Scene Captioning":    "captioning",
    "Visual Grounding":    "grounding",
    "Change Detection":    "change_detection",
    "Optical-SAR Fusion":  "optical_sar_fusion",
}

#: Tasks that require exactly two input images
PAIR_TASKS: set = {"change_detection", "optical_sar_fusion"}

#: Tasks that require a non-empty query string
QUERY_REQUIRED_TASKS: set = {"single_vqa", "grounding"}

#: Example queries shown on the landing page
EXAMPLE_QUERIES: list = [
    "What type of land cover is visible?",
    "Describe the scene.",
    "Is there any significant change between these observations?",
    "What features are visible in the satellite data?",
    "Compare the optical and SAR observations.",
    "Is there vegetation in this image?",
    "What is the dominant land use in this area?",
]

# -- Encoder constraints --------------------------------------------------------

EXPECTED_CHANNELS: int = 12
EXPECTED_H:        int = 120
EXPECTED_W:        int = 120

#: Channel order: must match BigEarthNet MobileViT-S training data
CHANNEL_NAMES: list = [
    "B02", "B03", "B04", "B08",
    "B05", "B06", "B07", "B11",
    "B12", "B8A", "VH",  "VV",
]

#: Channels used for RGB preview (B04=idx2, B03=idx1, B02=idx0)
RGB_PREVIEW_INDICES: tuple = (2, 1, 0)
RGB_PREVIEW_LABEL:   str = "RGB preview - B04 (R) / B03 (G) / B02 (B)"

# -- UI text -------------------------------------------------------------------

APP_TITLE    = "SATQUERY AI"
APP_SUBTITLE = "Natural-language intelligence for Earth observation data"
PAGE_ICON    = "satellite"

GROUNDING_NOT_IMPLEMENTED_MSG = (
    "Visual grounding is not implemented in the current prototype. "
    "The backend returns success=False / boxes=None for this task. "
    "A future phase will integrate a dedicated grounding model. "
    "No bounding-box coordinates have been fabricated."
)
