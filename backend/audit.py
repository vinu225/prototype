"""
backend/audit.py
=================
Audit logger for SatQuery AI Phase 3.

Every successful (and failed) tool execution writes a structured JSON record
to audit_logs/<session_id>.json.

Schema
------
{
    "session_id":     str,
    "timestamp":      str (ISO-8601),
    "query":          str | null,
    "task":           str,
    "tool":           str,
    "model_used":     str,
    "input_files":    list[str],       ← filenames only, NOT raw image bytes
    "input_metadata": dict,
    "routing_reason": str,
    "processing_time": float,
    "confidence":     float | null,
    "confidence_type": str,
    "success":        bool,
    "error":          str | null
}

IMPORTANT:
- Raw satellite image data is NEVER written to audit logs.
- HuggingFace tokens and secrets are NEVER included.
- Audit logs are append-safe (one file per session).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOG = logging.getLogger(__name__)

# Audit log directory (relative to project root, created automatically)
_AUDIT_DIR = Path(__file__).parent.parent / "audit_logs"


def _ensure_audit_dir() -> Path:
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return _AUDIT_DIR


def generate_session_id() -> str:
    """Generate a unique session ID for this request."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"session_{ts}_{uid}"


def write_audit(
    session_id:      str,
    query:           str | None,
    task:            str,
    tool:            str,
    model_used:      str,
    input_files:     list[str],
    input_metadata:  dict,
    routing_reason:  str,
    processing_time: float,
    confidence:      float | None,
    confidence_type: str,
    success:         bool,
    error:           str | None = None,
) -> Path:
    """
    Write one structured audit record.

    Parameters
    ----------
    session_id      : Unique session identifier for this request.
    query           : User's natural-language query (may be None for /analyse).
    task            : Selected task name (e.g. 'single_vqa').
    tool            : Tool class name used (e.g. 'single_vqa').
    model_used      : Model identifier string from ToolResult.
    input_files     : List of uploaded filenames (no raw bytes).
    input_metadata  : Dict with shape/dtype info (no raw data).
    routing_reason  : Human-readable routing explanation.
    processing_time : Wall-clock seconds for tool execution.
    confidence      : Confidence score or None.
    confidence_type : 'calibrated' | 'heuristic' | 'not_available'.
    success         : Whether the tool executed successfully.
    error           : Error message if success=False.

    Returns
    -------
    Path to the written audit log file.
    """
    audit_dir = _ensure_audit_dir()
    record = {
        "session_id":      session_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "query":           query,
        "task":            task,
        "tool":            tool,
        "model_used":      model_used,
        "input_files":     input_files,
        "input_metadata":  input_metadata,
        "routing_reason":  routing_reason,
        "processing_time": round(processing_time, 3),
        "confidence":      confidence,
        "confidence_type": confidence_type,
        "success":         success,
        "error":           error,
    }

    log_path = audit_dir / f"{session_id}.json"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        _LOG.error("Failed to write audit log %s: %s", log_path, exc)

    return log_path
