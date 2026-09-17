"""
pipeline/preprocess.py
=======================
Converts raw satellite data (numpy arrays or torch tensors) into the exact
[B, 12, 120, 120] float32 tensor expected by SatelliteEncoder.

SOURCE OF TRUTH: Colab cells 31 (get_dummy_batch), 64 (SentinelDataset).

Rules (enforced strictly):
  - Exactly 12 channels required — no silent resizing or channel dropping.
  - Spatial size must be exactly 120×120 — no silent resizing.
  - dtype must be float32 — int inputs are cast, others rejected.
  - Batch dimension is added if a 3-D array [C, H, W] is passed.
  - Invalid inputs raise ValueError with a clear message.

Channel order (from Colab README cell 37):
  Index  Band
  -----  ----
    0    B02  (Blue)
    1    B03  (Green)
    2    B04  (Red)
    3    B08  (NIR-Wide)
    4    B05  (RedEdge-1)
    5    B06  (RedEdge-2)
    6    B07  (RedEdge-3)
    7    B11  (SWIR-1)
    8    B12  (SWIR-2)
    9    B8A  (NIR-Narrow)
   10    VH   (Radar dual-pol)
   11    VV   (Radar co-pol)
"""

import numpy as np
import torch

EXPECTED_CHANNELS = 12
EXPECTED_H        = 120
EXPECTED_W        = 120


def preprocess_image(
    image,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert a raw satellite patch into a model-ready tensor.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Accepted shapes:
          [C, H, W]       — single patch, batch dim will be added
          [B, C, H, W]    — already batched
        Must have C == 12, H == 120, W == 120.

    device : str
        Target device for the output tensor ('cpu' or 'cuda').

    Returns
    -------
    torch.Tensor
        Shape [B, 12, 120, 120], dtype float32, on `device`.

    Raises
    ------
    ValueError
        If channel count != 12, or spatial dims != 120×120,
        or ndim is unexpected.
    TypeError
        If dtype is not castable to float32.
    """
    # ── Convert numpy → torch ──────────────────────────────────────────
    if isinstance(image, np.ndarray):
        # Only int and float dtypes are accepted.
        if not np.issubdtype(image.dtype, np.floating) and \
           not np.issubdtype(image.dtype, np.integer):
            raise TypeError(
                f"numpy array dtype must be floating or integer, got {image.dtype}"
            )
        image = torch.from_numpy(image.astype(np.float32))
    elif isinstance(image, torch.Tensor):
        if image.dtype not in (torch.float16, torch.float32, torch.float64,
                               torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f"Unsupported tensor dtype: {image.dtype}")
        image = image.float()
    else:
        raise TypeError(
            f"image must be np.ndarray or torch.Tensor, got {type(image)}"
        )

    # ── Add batch dimension if needed ──────────────────────────────────
    if image.ndim == 3:
        image = image.unsqueeze(0)   # [C, H, W] → [1, C, H, W]
    elif image.ndim == 4:
        pass                          # already [B, C, H, W]
    else:
        raise ValueError(
            f"Expected 3-D [C,H,W] or 4-D [B,C,H,W] input, "
            f"got {image.ndim}-D tensor with shape {list(image.shape)}"
        )

    # ── Validate channels ───────────────────────────────────────────────
    C = image.shape[1]
    if C != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected exactly {EXPECTED_CHANNELS} channels "
            f"(B02 B03 B04 B08 B05 B06 B07 B11 B12 B8A VH VV), "
            f"got {C}. "
            f"{'Too few — missing bands.' if C < EXPECTED_CHANNELS else 'Too many — extra bands not supported.'}"
        )

    # ── Validate spatial dims ───────────────────────────────────────────
    H, W = image.shape[2], image.shape[3]
    if H != EXPECTED_H or W != EXPECTED_W:
        raise ValueError(
            f"Expected spatial size {EXPECTED_H}×{EXPECTED_W}, "
            f"got {H}×{W}. "
            "Do not silently resize — the MobileViT-S encoder expects exactly 120×120."
        )

    return image.to(device)


def make_dummy_batch(
    batch_size: int = 1,
    device:     str = "cpu",
    seed:       int = 42,
) -> torch.Tensor:
    """
    Create a random 12-channel batch for unit tests / sanity checks.
    Equivalent to Colab's get_dummy_batch().

    Returns
    -------
    torch.Tensor
        Shape [batch_size, 12, 120, 120], dtype float32.
    """
    torch.manual_seed(seed)
    return torch.randn(batch_size, EXPECTED_CHANNELS, EXPECTED_H, EXPECTED_W).to(device)
