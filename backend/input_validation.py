"""
backend/input_validation.py
============================
Strict input validation for satellite file uploads in SatQuery AI.

The MobileViT-S encoder expects:
    [B, 12, 120, 120]  float32

Rules (enforced — no silent reshaping)
--------------------------------------
  - Exactly 12 channels required
  - Spatial size must be 120 × 120
  - Numeric dtype required (float or integer)
  - Minimum 1 file, maximum 2 files
  - Supported extensions: .npy (NumPy array)

Invalid input → raises InputValidationError with a clear message.
"""

from __future__ import annotations

import io
import logging
from typing import Literal

import numpy as np
from fastapi import UploadFile

_LOG = logging.getLogger(__name__)

EXPECTED_CHANNELS = 12
EXPECTED_H        = 120
EXPECTED_W        = 120

SUPPORTED_EXTENSIONS = {".npy"}


class InputValidationError(Exception):
    """Raised when uploaded file fails validation."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code    = code
        self.message = message


async def _read_npy(upload: UploadFile) -> np.ndarray:
    """Read an uploaded .npy file into a numpy array."""
    raw = await upload.read()
    try:
        arr = np.load(io.BytesIO(raw))
    except Exception as exc:
        raise InputValidationError(
            "invalid_file",
            f"Could not read {upload.filename!r} as a NumPy array: {exc}",
        ) from exc
    return arr


def _validate_array_shape(arr: np.ndarray, filename: str) -> None:
    """Validate shape and dtype of a loaded numpy array."""
    # Accept [C,H,W] or [B,C,H,W]
    if arr.ndim == 3:
        C, H, W = arr.shape
    elif arr.ndim == 4:
        _, C, H, W = arr.shape
    else:
        raise InputValidationError(
            "invalid_shape",
            f"{filename!r}: Expected 3-D [C,H,W] or 4-D [B,C,H,W] array, "
            f"got {arr.ndim}-D array with shape {list(arr.shape)}.",
        )

    if C != EXPECTED_CHANNELS:
        raise InputValidationError(
            "wrong_channels",
            f"{filename!r}: Expected exactly {EXPECTED_CHANNELS} channels "
            f"(12-band Sentinel-1+2 composite), got {C}. "
            f"{'Too few — missing bands.' if C < EXPECTED_CHANNELS else 'Too many — extra bands not supported.'}",
        )

    if H != EXPECTED_H or W != EXPECTED_W:
        raise InputValidationError(
            "wrong_spatial",
            f"{filename!r}: Expected spatial size {EXPECTED_H}×{EXPECTED_W}, "
            f"got {H}×{W}. The MobileViT-S encoder requires exactly 120×120 patches.",
        )

    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
        raise InputValidationError(
            "wrong_dtype",
            f"{filename!r}: Array dtype must be numeric (float or integer), got {arr.dtype}.",
        )


def _check_extension(filename: str | None) -> None:
    """Check that the uploaded file has a supported extension."""
    if not filename:
        raise InputValidationError("missing_filename", "Uploaded file has no filename.")
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise InputValidationError(
            "unsupported_format",
            f"Unsupported file format {ext!r}. Only .npy (NumPy array) files are accepted.",
        )


async def validate_single_image(files: list[UploadFile]) -> tuple[np.ndarray, dict]:
    """
    Validate exactly one satellite image upload.

    Returns
    -------
    (numpy_array, metadata_dict)
    """
    if len(files) == 0:
        raise InputValidationError("missing_file", "At least one satellite image file is required.")
    if len(files) > 1:
        raise InputValidationError(
            "too_many_files",
            f"This task requires exactly 1 satellite image, got {len(files)}.",
        )
    upload = files[0]
    _check_extension(upload.filename)
    arr = await _read_npy(upload)
    _validate_array_shape(arr, upload.filename)

    shape = list(arr.shape)
    meta  = {"filename": upload.filename, "shape": shape, "dtype": str(arr.dtype)}
    _LOG.debug("Validated single image: %s", meta)
    return arr, meta


async def validate_image_pair(files: list[UploadFile]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Validate exactly two satellite images (for change detection).

    Returns
    -------
    (before_array, after_array, metadata_dict)
    """
    if len(files) == 0:
        raise InputValidationError("missing_file", "Two satellite image files are required (before and after).")
    if len(files) != 2:
        raise InputValidationError(
            "wrong_file_count",
            f"Change detection requires exactly 2 images (before + after), got {len(files)}.",
        )

    results = []
    for upload in files:
        _check_extension(upload.filename)
        arr = await _read_npy(upload)
        _validate_array_shape(arr, upload.filename)
        results.append(arr)

    meta = {
        "file_0": {"filename": files[0].filename, "shape": list(results[0].shape)},
        "file_1": {"filename": files[1].filename, "shape": list(results[1].shape)},
    }
    return results[0], results[1], meta


async def validate_optical_sar_pair(files: list[UploadFile]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Validate exactly two satellite images (optical + SAR) for fusion.

    Returns
    -------
    (optical_array, sar_array, metadata_dict)
    """
    if len(files) != 2:
        raise InputValidationError(
            "wrong_file_count",
            f"Optical-SAR fusion requires exactly 2 images (optical + SAR), got {len(files)}.",
        )
    return await validate_image_pair(files)   # same dimension rules apply


def get_input_metadata(arr: np.ndarray, filename: str) -> dict:
    """Build a metadata dict for audit logging (no raw data)."""
    return {"filename": filename, "shape": list(arr.shape), "dtype": str(arr.dtype)}
