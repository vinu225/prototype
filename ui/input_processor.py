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


def get_demo_preset(preset_key: str = "agriculture") -> ProcessedInput:
    """
    Generate a synthetic 12-channel [12, 120, 120] sample with realistic spectral profiles
    for quick demonstration (derived from Phase 3 test patterns).
    """
    seed_map = {"agriculture": 42, "wetland": 101, "urban": 202}
    seed = seed_map.get(preset_key, 42)
    rng = np.random.default_rng(seed)

    # Base noise
    arr = rng.uniform(0.02, 0.08, size=(EXPECTED_CHANNELS, EXPECTED_H, EXPECTED_W)).astype(np.float32)

    # Spectral band adjustments:
    # 0: B02 (Blue), 1: B03 (Green), 2: B04 (Red), 3: B08 (NIR),
    # 4: B05, 5: B06, 6: B07, 7: B11 (SWIR1), 8: B12 (SWIR2), 9: B8A, 10: VH, 11: VV
    if preset_key == "agriculture":
        arr[0] *= 0.6   # Low blue
        arr[1] *= 1.4   # Green peak
        arr[2] *= 0.5   # Chlorophyll absorption in red
        arr[3] = rng.uniform(0.35, 0.55, size=(EXPECTED_H, EXPECTED_W))  # High NIR
        arr[4:7] = rng.uniform(0.20, 0.35, size=(3, EXPECTED_H, EXPECTED_W)) # Red edge
        arr[10] = rng.uniform(0.05, 0.12, size=(EXPECTED_H, EXPECTED_W)) # VH
        arr[11] = rng.uniform(0.12, 0.22, size=(EXPECTED_H, EXPECTED_W)) # VV
        desc = "Agricultural / Vegetation Scene"
    elif preset_key == "wetland":
        arr[0] = rng.uniform(0.25, 0.35, size=(EXPECTED_H, EXPECTED_W))  # High blue/green
        arr[1] = rng.uniform(0.20, 0.30, size=(EXPECTED_H, EXPECTED_W))
        arr[2] = rng.uniform(0.10, 0.15, size=(EXPECTED_H, EXPECTED_W))
        arr[3] = rng.uniform(0.02, 0.06, size=(EXPECTED_H, EXPECTED_W))  # Water absorbs NIR
        arr[7:9] = rng.uniform(0.01, 0.04, size=(2, EXPECTED_H, EXPECTED_W)) # Water absorbs SWIR
        arr[10] = rng.uniform(0.005, 0.02, size=(EXPECTED_H, EXPECTED_W)) # Low SAR
        arr[11] = rng.uniform(0.01, 0.04, size=(EXPECTED_H, EXPECTED_W))
        desc = "Water / Wetland Scene"
    else:  # urban
        arr[0:3] = rng.uniform(0.15, 0.25, size=(3, EXPECTED_H, EXPECTED_W)) # Neutral gray
        arr[3] = rng.uniform(0.12, 0.20, size=(EXPECTED_H, EXPECTED_W))
        arr[7:9] = rng.uniform(0.22, 0.35, size=(2, EXPECTED_H, EXPECTED_W)) # High SWIR
        arr[10] = rng.uniform(0.15, 0.30, size=(EXPECTED_H, EXPECTED_W)) # Strong SAR double-bounce
        arr[11] = rng.uniform(0.25, 0.45, size=(EXPECTED_H, EXPECTED_W))
        desc = "Urban / Built-up Scene"

    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    npy_bytes = _to_npy_bytes(arr)
    preview = _make_rgb_preview(arr)

    return ProcessedInput(
        filename  = f"demo_{preset_key}.npy",
        array     = arr,
        npy_bytes = npy_bytes,
        preview   = preview,
        info      = {
            "filename":  f"demo_{preset_key}.npy",
            "file_type": f"Preset \u2014 {desc}",
            "shape":     list(arr.shape),
            "dtype":     str(arr.dtype),
            "channels":  CHANNEL_NAMES,
            "valid":     True,
            "preset":    True,
        },
    )


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

    Uses a NamedTemporaryFile with .tif extension as the primary approach so
    rasterio can detect the GTiff driver via the file extension.  Falls back to
    rasterio.MemoryFile with a filename hint, then PIL.
    """
    import tempfile as _tmp
    import os as _os

    # -- Primary: NamedTemporaryFile (.tif) --- most reliable for GeoTIFF ----
    try:
        import rasterio
        tmp_path: Optional[str] = None
        with _tmp.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
            tf.write(raw)
            tmp_path = tf.name
        try:
            with rasterio.open(tmp_path) as ds:
                bands  = ds.count
                height = ds.height
                width  = ds.width
                crs    = str(ds.crs) if ds.crs else None
                arr    = None
                if bands == EXPECTED_CHANNELS and height == EXPECTED_H and width == EXPECTED_W:
                    arr = ds.read().astype(np.float32)  # [C, H, W]
            return {"bands": bands, "height": height, "width": width,
                    "crs": crs, "array": arr, "driver": "rasterio"}
        finally:
            try: _os.unlink(tmp_path)
            except OSError: pass
    except ImportError:
        pass
    except Exception as exc:
        _LOG.warning("rasterio (NamedTemporaryFile) failed for %s: %s", filename, exc)

    # -- Fallback: rasterio MemoryFile with filename hint --------------------
    try:
        import rasterio
        from rasterio.io import MemoryFile
        # Pass filename so rasterio picks GTiff driver via extension
        with MemoryFile(raw, filename=filename) as mf:
            with mf.open() as ds:
                bands  = ds.count
                height = ds.height
                width  = ds.width
                crs    = str(ds.crs) if ds.crs else None
                arr    = None
                if bands == EXPECTED_CHANNELS and height == EXPECTED_H and width == EXPECTED_W:
                    arr = ds.read().astype(np.float32)
        return {"bands": bands, "height": height, "width": width,
                "crs": crs, "array": arr, "driver": "rasterio-memfile"}
    except ImportError:
        pass
    except Exception as exc:
        _LOG.warning("rasterio (MemoryFile) failed for %s: %s", filename, exc)

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


# =============================================================================
# SAR Demo Mode -- Sentinel-1 VH/VV pair support
# =============================================================================

@dataclass
class SARPairInput:
    """
    Validated Sentinel-1 VH + VV pair for SAR demo mode.

    IMPORTANT: input_mode = "sentinel1_sar_pair"
    This is NOT compatible with the 12-channel multimodal model.
    model_compatible is always False.
    """
    vh_filename:  str
    vv_filename:  str
    vh_array:     np.ndarray   # [H, W] float32
    vv_array:     np.ndarray   # [H, W] float32
    vh_npy_bytes: bytes        # for POST to /sar-analyse
    vv_npy_bytes: bytes
    vh_preview:   Optional[object]  # PIL Image (grayscale L)
    vv_preview:   Optional[object]
    composite_preview: Optional[object]  # PIL Image (false-color RGB)
    info:         dict = field(default_factory=dict)
    input_mode:   str  = "sentinel1_sar_pair"


def process_sar_pair(
    vh_file,
    vv_file,
) -> "SARPairInput | InputError":
    """
    Process a Sentinel-1 VH + VV file pair for SAR demo mode.

    Parameters
    ----------
    vh_file : streamlit UploadedFile  (VH polarisation)
    vv_file : streamlit UploadedFile  (VV polarisation)

    Returns
    -------
    SARPairInput on success, InputError on any validation failure.
    """
    vh_name = vh_file.name
    vv_name = vv_file.name
    vh_raw  = vh_file.read()
    vv_raw  = vv_file.read()

    vh_result = _load_single_band(vh_name, vh_raw)
    if isinstance(vh_result, InputError):
        return InputError(
            code    = vh_result.code,
            message = f"VH file: {vh_result.message}",
            detail  = vh_result.detail,
        )

    vv_result = _load_single_band(vv_name, vv_raw)
    if isinstance(vv_result, InputError):
        return InputError(
            code    = vv_result.code,
            message = f"VV file: {vv_result.message}",
            detail  = vv_result.detail,
        )

    vh_arr, vh_H, vh_W = vh_result
    vv_arr, vv_H, vv_W = vv_result

    if vh_H != vv_H or vh_W != vv_W:
        return InputError(
            code    = "sar_dimension_mismatch",
            message = (
                f"VH ({vh_H}\u00d7{vh_W}) and VV ({vv_H}\u00d7{vv_W}) "
                f"have different spatial dimensions."
            ),
            detail  = "Both VH and VV must have identical pixel dimensions.",
        )

    vh_bytes = _to_npy_bytes(vh_arr)
    vv_bytes = _to_npy_bytes(vv_arr)

    vh_preview  = _make_sar_preview(vh_arr, "VH")
    vv_preview  = _make_sar_preview(vv_arr, "VV")
    composite   = _make_sar_composite(vh_arr, vv_arr)

    spatial_note = (
        f" Note: {vh_H}\u00d7{vh_W} (not 120\u00d7120 \u2014 SAR stats still computed)."
        if (vh_H != 120 or vh_W != 120) else ""
    )

    return SARPairInput(
        vh_filename       = vh_name,
        vv_filename       = vv_name,
        vh_array          = vh_arr,
        vv_array          = vv_arr,
        vh_npy_bytes      = vh_bytes,
        vv_npy_bytes      = vv_bytes,
        vh_preview        = vh_preview,
        vv_preview        = vv_preview,
        composite_preview = composite,
        info = {
            "vh_filename":     vh_name,
            "vv_filename":     vv_name,
            "vh_shape":        list(vh_arr.shape),
            "vv_shape":        list(vv_arr.shape),
            "spatial":         f"{vh_H}\u00d7{vh_W}" + spatial_note,
            "bands":           "VH + VV only (Sentinel-1 C-band)",
            "mode":            "sentinel1_sar_pair",
            "model_compatible": False,
        },
        input_mode = "sentinel1_sar_pair",
    )


def process_sar_pair_bytes(
    vh_raw:   bytes,
    vv_raw:   bytes,
    vh_name:  str = "vh.npy",
    vv_name:  str = "vv.npy",
) -> "SARPairInput | InputError":
    """
    Process SAR pair from raw bytes directly (used by tests without Streamlit).
    """
    vh_result = _load_single_band(vh_name, vh_raw)
    if isinstance(vh_result, InputError):
        return InputError(vh_result.code, f"VH: {vh_result.message}", vh_result.detail)

    vv_result = _load_single_band(vv_name, vv_raw)
    if isinstance(vv_result, InputError):
        return InputError(vv_result.code, f"VV: {vv_result.message}", vv_result.detail)

    vh_arr, vh_H, vh_W = vh_result
    vv_arr, vv_H, vv_W = vv_result

    if vh_H != vv_H or vh_W != vv_W:
        return InputError(
            code    = "sar_dimension_mismatch",
            message = f"VH ({vh_H}\u00d7{vh_W}) and VV ({vv_H}\u00d7{vv_W}) dimensions differ.",
            detail  = "Both must be identical.",
        )

    return SARPairInput(
        vh_filename       = vh_name,
        vv_filename       = vv_name,
        vh_array          = vh_arr,
        vv_array          = vv_arr,
        vh_npy_bytes      = _to_npy_bytes(vh_arr),
        vv_npy_bytes      = _to_npy_bytes(vv_arr),
        vh_preview        = _make_sar_preview(vh_arr, "VH"),
        vv_preview        = _make_sar_preview(vv_arr, "VV"),
        composite_preview = _make_sar_composite(vh_arr, vv_arr),
        info = {
            "spatial":         f"{vh_H}\u00d7{vh_W}",
            "mode":            "sentinel1_sar_pair",
            "model_compatible": False,
        },
        input_mode = "sentinel1_sar_pair",
    )


# ---------------------------------------------------------------------------
# SAR internal helpers
# ---------------------------------------------------------------------------

def _load_single_band(filename: str, raw: bytes):
    """
    Load a single-band TIFF or 1/2-D .npy file.
    Returns (np.ndarray [H,W] float32, H, W) or InputError.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".npy":
        try:
            arr = np.load(io.BytesIO(raw))
        except Exception as exc:
            return InputError("invalid_file", f"Cannot parse {filename} as .npy.", str(exc))
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 2:
            return InputError(
                "invalid_shape",
                f"{filename}: expected 2-D or 1-band array, got shape {list(arr.shape)}.",
                "SAR files must be single-polarisation (2-D or [1,H,W] arrays).",
            )
        arr_f32 = arr.astype(np.float32)
        H, W = arr_f32.shape
        return arr_f32, H, W

    elif ext in {".tif", ".tiff"}:
        import tempfile as _tmp2
        import os as _os2
        # -- Primary: NamedTemporaryFile (.tif) — rasterio detects GTiff ------
        try:
            import rasterio
            _validation_err = None
            _result         = None
            with _tmp2.NamedTemporaryFile(suffix=".tif", delete=False) as _tf:
                _tf.write(raw)
                _tpath = _tf.name
            try:
                with rasterio.open(_tpath) as ds:
                    if ds.count != 1:
                        _validation_err = InputError(
                            "sar_wrong_bands",
                            f"{filename}: expected 1-band SAR TIFF, got {ds.count} bands.",
                            "Each SAR file (VH / VV) must be a single-polarisation TIFF.",
                        )
                    else:
                        _arr = ds.read(1).astype(np.float32)
                        _result = (_arr, ds.height, ds.width)
            finally:
                try: _os2.unlink(_tpath)
                except OSError: pass
            if _validation_err:
                return _validation_err
            if _result:
                return _result
        except ImportError:
            pass
        except Exception as exc:
            _LOG.warning("rasterio (NamedTemporaryFile) SAR failed for %s: %s", filename, exc)

        # -- Fallback: rasterio MemoryFile with filename hint -----------------
        try:
            import rasterio
            from rasterio.io import MemoryFile
            with MemoryFile(raw, filename=filename) as mf:
                with mf.open() as ds:
                    if ds.count != 1:
                        return InputError(
                            "sar_wrong_bands",
                            f"{filename}: expected 1-band SAR TIFF, got {ds.count} bands.",
                            "Each SAR file (VH / VV) must be a single-polarisation TIFF.",
                        )
                    arr = ds.read(1).astype(np.float32)
                    H, W = ds.height, ds.width
            return arr, H, W
        except ImportError:
            pass
        except Exception as exc:
            _LOG.warning("rasterio (MemoryFile) SAR failed for %s: %s", filename, exc)

        # -- PIL fallback -----------------------------------------------------
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(raw))
            W, H = img.size
            bands = len(img.getbands())
            if bands != 1:
                return InputError(
                    "sar_wrong_bands",
                    f"{filename}: PIL detected {bands} bands, expected 1.",
                    "SAR files must be single-polarisation.",
                )
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return arr, H, W
        except Exception as exc:
            return InputError("tiff_parse_error", f"Cannot read {filename}.", str(exc))

    else:
        return InputError(
            "unsupported_format",
            f"Unsupported format {ext!r} for SAR file.",
            "SAR files must be .tif/.tiff (single-band GeoTIFF) or .npy (2-D array).",
        )


def _make_sar_preview(arr: np.ndarray, label: str = "") -> Optional[object]:
    """
    Grayscale PIL preview for a single SAR band [H,W] with p2-p98 stretch.
    Returns None on any error. Never raises.
    """
    try:
        from PIL import Image
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        if hi <= lo:
            stretched = np.zeros_like(arr, dtype=np.uint8)
        else:
            stretched = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(stretched, "L")
    except Exception as exc:
        _LOG.warning("SAR preview failed for %s: %s", label, exc)
        return None


def _make_sar_composite(vh: np.ndarray, vv: np.ndarray) -> Optional[object]:
    """
    False-color VH/VV composite: R=VH, G=VV, B=VH/VV ratio (all stretched).
    Clearly a false-color visualization -- not RGB optical imagery.
    Returns None on any error.
    """
    try:
        from PIL import Image

        def stretch(band: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(band, 2), np.percentile(band, 98)
            if hi <= lo:
                return np.zeros_like(band, dtype=np.uint8)
            return (np.clip((band - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

        safe_vv = np.where(np.abs(vv) > 1e-12, vv, 1e-12)
        ratio   = np.clip(vh / safe_vv, 0, 5)   # clamp extreme ratios

        r = stretch(vh)
        g = stretch(vv)
        b = stretch(ratio)
        rgb = np.stack([r, g, b], axis=-1)
        return Image.fromarray(rgb, "RGB")
    except Exception as exc:
        _LOG.warning("SAR composite preview failed: %s", exc)
        return None
