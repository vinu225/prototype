"""
ui/input_processor.py
======================
File-handling, validation, and visualization utilities for the SatQuery AI
Streamlit interface.

Responsibilities
----------------
- Detect file type (.npy / .tif/.tiff)
- Validate NumPy arrays against encoder requirements [12, 120, 120]
- Inspect TIFF metadata and give honest channel/dimension reports
- Generate safe RGB preview images (B04/B03/B02 percentile-stretched)
- Re-serialize validated arrays to .npy bytes for multipart upload

NO model inference or streamlit calls happen here.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ui.config import (
    EXPECTED_CHANNELS,
    EXPECTED_H,
    EXPECTED_W,
    CHANNEL_NAMES,
    RGB_PREVIEW_INDICES,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ProcessedInput:
    """Successfully validated input ready for backend submission."""
    filename:    str
    array:       np.ndarray   # shape [12, 120, 120] float32
    npy_bytes:   bytes        # re-serialized for multipart POST
    preview:     Optional[object]  # PIL.Image or None
    info:        dict = field(default_factory=dict)


@dataclass
class InputError:
    """Structured validation failure with machine code + human message."""
    code:    str
    message: str
    detail:  str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_upload(uploaded_file) -> "ProcessedInput | InputError":
    """
    Validate an uploaded Streamlit file and return ProcessedInput or InputError.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The file object from st.file_uploader.

    Returns
    -------
    ProcessedInput on success, InputError on any validation failure.
    """
    name = uploaded_file.name
    ext  = Path(name).suffix.lower()
    raw  = uploaded_file.read()

    if ext == ".npy":
        return _process_npy(name, raw)
    elif ext in {".tif", ".tiff"}:
        return _process_tiff(name, raw)
    else:
        return InputError(
            code    = "unsupported_format",
            message = f"Unsupported file type \u2018{ext}\u2019.",
            detail  = (
                "Only .npy (NumPy array) and .tif / .tiff (GeoTIFF) files are accepted. "
                "The model requires a 12-channel 120\u00d7120 NumPy array."
            ),
        )


def process_npy_bytes(raw: bytes, filename: str = "input.npy") -> "ProcessedInput | InputError":
    """Validate raw .npy bytes directly (used in tests without a Streamlit context)."""
    return _process_npy(filename, raw)


def make_second_preview(arr: np.ndarray) -> Optional[object]:
    """Public alias: RGB preview for the second image in a pair."""
    return _make_rgb_preview(arr)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_npy(filename: str, raw: bytes) -> "ProcessedInput | InputError":
    """Load and strictly validate a .npy file."""
    try:
        arr = np.load(io.BytesIO(raw))
    except Exception as exc:
        return InputError(
            code    = "invalid_file",
            message = "Could not parse the file as a NumPy array.",
            detail  = str(exc),
        )

    # Normalise: accept [C,H,W] or [B,C,H,W] (batch size must be 1)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            return InputError(
                code    = "invalid_shape",
                message = f"Batched array must have batch size 1, got {arr.shape[0]}.",
                detail  = f"Shape received: {list(arr.shape)}",
            )
        arr = arr[0]

    if arr.ndim != 3:
        return InputError(
            code    = "invalid_shape",
            message = f"Expected 3-D [C,H,W] or 4-D [1,C,H,W] array, got {arr.ndim}-D.",
            detail  = f"Shape received: {list(arr.shape)}",
        )

    C, H, W = arr.shape

    if C != EXPECTED_CHANNELS:
        dir_msg = "Too few \u2014 missing bands." if C < EXPECTED_CHANNELS else "Too many \u2014 extra bands not supported."
        return InputError(
            code    = "wrong_channels",
            message = f"Expected exactly {EXPECTED_CHANNELS} channels, got {C}. {dir_msg}",
            detail  = (
                f"Required channel order: {', '.join(CHANNEL_NAMES)}.\n"
                "Arbitrary channel mappings are not silently remapped."
            ),
        )

    if H != EXPECTED_H or W != EXPECTED_W:
        return InputError(
            code    = "wrong_spatial",
            message = f"Expected 120\u00d7120 spatial dimensions, got {H}\u00d7{W}.",
            detail  = "The MobileViT-S encoder requires exactly 120\u00d7120 patches.",
        )

    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
        return InputError(
            code    = "wrong_dtype",
            message = f"Array dtype must be numeric (float or integer), got {arr.dtype}.",
        )

    arr_f32   = arr.astype(np.float32)
    npy_bytes = _to_npy_bytes(arr_f32)
    preview   = _make_rgb_preview(arr_f32)

    return ProcessedInput(
        filename  = filename,
        array     = arr_f32,
        npy_bytes = npy_bytes,
        preview   = preview,
        info      = {
            "filename":  filename,
            "file_type": "NumPy array (.npy)",
            "shape":     list(arr_f32.shape),
            "dtype":     str(arr_f32.dtype),
            "channels":  CHANNEL_NAMES,
            "valid":     True,
        },
    )


def _process_tiff(filename: str, raw: bytes) -> "ProcessedInput | InputError":
    """
    Inspect a GeoTIFF and either produce a ProcessedInput or return a
    human-readable InputError explaining preprocessing requirements.

    A Sentinel-1 single-polarisation TIFF (1 band) cannot automatically
    become a 12-channel composite -- this is made explicit, not silently ignored.
    """
    meta = _inspect_tiff(raw, filename)
    if isinstance(meta, InputError):
        return meta

    bands, H, W = meta["bands"], meta["height"], meta["width"]
    driver = meta.get("driver", "unknown")

    # -- Wrong channel count -------------------------------------------------
    if bands != EXPECTED_CHANNELS:
        if bands == 1:
            detail = (
                f"Detected: 1 band, {H}\u00d7{W} pixels (driver: {driver}).\n\n"
                "This is likely a single Sentinel-1 SAR polarisation (VV or VH). "
                "The model requires all 12 channels: "
                f"{', '.join(CHANNEL_NAMES)}.\n\n"
                "To use this file: combine it with the remaining 11 bands into "
                "a [12, 120, 120] NumPy composite and save as .npy."
            )
        elif bands == 2:
            detail = (
                f"Detected: 2 bands (possibly VV + VH), {H}\u00d7{W} pixels.\n\n"
                "The model needs 12 channels including 10 Sentinel-2 optical bands. "
                "Preprocessing (channel assembly) is required before this TIFF "
                "can be used as model input."
            )
        else:
            detail = (
                f"Detected: {bands} band(s), {H}\u00d7{W} pixels (driver: {driver}).\n\n"
                f"Required channel order: {', '.join(CHANNEL_NAMES)}.\n"
                "Assemble all 12 bands into a [12, 120, 120] .npy array first."
            )
        return InputError(
            code    = "tiff_wrong_channels",
            message = f"This TIFF has {bands} band(s) \u2014 model requires exactly 12.",
            detail  = detail,
        )

    # -- Wrong spatial size --------------------------------------------------
    if H != EXPECTED_H or W != EXPECTED_W:
        return InputError(
            code    = "tiff_wrong_spatial",
            message = f"This TIFF is {H}\u00d7{W} pixels \u2014 model requires exactly 120\u00d7120.",
            detail  = (
                "Crop or resample to 120\u00d7120 pixels and save as a [12, 120, 120] .npy array."
            ),
        )

    # -- Compatible: 12-band, 120x120 ----------------------------------------
    arr = meta.get("array")
    if arr is None:
        return InputError(
            code    = "tiff_load_failed",
            message = "Could not read pixel data from this TIFF.",
            detail  = "Try converting to a NumPy .npy array and re-uploading.",
        )

    arr_f32   = arr.astype(np.float32)
    npy_bytes = _to_npy_bytes(arr_f32)
    preview   = _make_rgb_preview(arr_f32)

    return ProcessedInput(
        filename  = filename,
        array     = arr_f32,
        npy_bytes = npy_bytes,
        preview   = preview,
        info      = {
            "filename":  filename,
            "file_type": f"GeoTIFF (.tif) \u2014 driver: {driver}",
            "shape":     list(arr_f32.shape),
            "dtype":     str(arr_f32.dtype),
            "channels":  CHANNEL_NAMES,
            "crs":       meta.get("crs"),
            "valid":     True,
        },
    )


def _inspect_tiff(raw: bytes, filename: str) -> "dict | InputError":
    """
    Read TIFF metadata (band count, dimensions, optional pixel data).
    Tries rasterio first (best for GeoTIFF), falls back to PIL.
    """
    # -- rasterio (preferred) ------------------------------------------------
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(raw) as mf:
            with mf.open() as ds:
                bands  = ds.count
                height = ds.height
                width  = ds.width
                crs    = str(ds.crs) if ds.crs else None
                arr    = None
                if bands == EXPECTED_CHANNELS and height == EXPECTED_H and width == EXPECTED_W:
                    arr = ds.read().astype(np.float32)  # [C, H, W]
        return {"bands": bands, "height": height, "width": width,
                "crs": crs, "array": arr, "driver": "rasterio"}
    except ImportError:
        pass
    except Exception as exc:
        _LOG.warning("rasterio failed for %s: %s", filename, exc)

    # -- PIL fallback --------------------------------------------------------
    try:
        from PIL import Image
        img    = Image.open(io.BytesIO(raw))
        W, H   = img.size
        bands  = len(img.getbands())
        arr    = None
        if bands == EXPECTED_CHANNELS and H == EXPECTED_H and W == EXPECTED_W:
            a = np.array(img, dtype=np.float32)
            if a.ndim == 2:
                a = a[np.newaxis, ...]        # [1,H,W]
            elif a.ndim == 3:
                a = a.transpose(2, 0, 1)      # [C,H,W]
            arr = a
        return {"bands": bands, "height": H, "width": W,
                "crs": None, "array": arr, "driver": "PIL"}
    except Exception as exc:
        return InputError(
            code    = "tiff_parse_error",
            message = "Could not read this TIFF file.",
            detail  = str(exc),
        )


def _to_npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to in-memory .npy bytes."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _make_rgb_preview(arr: np.ndarray) -> Optional[object]:
    """
    Create a PIL RGB Image from a [12, 120, 120] float32 array.

    Uses channels B04 (index 2), B03 (index 1), B02 (index 0) as R, G, B
    with a p2-p98 percentile stretch for visual clarity.
    Returns None on any error (never raises).
    """
    try:
        from PIL import Image

        def stretch(band: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(band, 2), np.percentile(band, 98)
            if hi <= lo:
                return np.zeros_like(band, dtype=np.uint8)
            return (np.clip((band - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

        r, g, b = [stretch(arr[i]) for i in RGB_PREVIEW_INDICES]
        return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")
    except Exception as exc:
        _LOG.warning("RGB preview generation failed: %s", exc)
        return None
