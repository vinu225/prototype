"""
backend/schemas.py
===================
Pydantic request/response schemas for SatQuery AI Phase 3 API.

All schemas are JSON-serializable.
Confidence semantics preserved from Phase 2 ToolResult:
  - confidence=None         → model produces no score
  - confidence_type=heuristic → proxy metric (e.g. L2 distance)
  - confidence_type=calibrated → proper scoring model (not used in prototype)
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── Audit sub-schema ─────────────────────────────────────────────────────────

class AuditInfo(BaseModel):
    session_id:    str
    timestamp:     str
    model_used:    str
    tool:          str
    routing_reason: str
    input_files:   list[str]
    query:         str | None = None


# ── Bounding box (grounding stub) ─────────────────────────────────────────────

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str | None = None


# ── Tool response (success) ───────────────────────────────────────────────────

class SatQueryResponse(BaseModel):
    success:          bool
    task:             str
    query:            str | None = None
    answer:           str
    confidence:       float | None = None
    confidence_type:  str   = "not_available"   # calibrated | heuristic | not_available
    visual_evidence:  dict[str, Any] | None = None
    processing_time:  float = 0.0
    audit:            AuditInfo | None = None
    model_used:       str = ""

    class Config:
        # Allow any extra fields from ToolResult.metadata to pass through
        extra = "allow"


# ── Error response ────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    success: bool  = False
    error:   str              # machine-readable error code
    message: str              # human-readable description

    # Optional extras for debugging without exposing tracebacks
    task:    str | None = None
    query:   str | None = None


# ── Tool discovery schemas ────────────────────────────────────────────────────

class ToolInfo(BaseModel):
    name:        str
    description: str
    implemented: bool = True
    input_type:  str  = "single_image"   # single_image | image_pair | optical_sar_pair


# ── Health response ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = "ok"
    service: str = "SatQuery AI"
    phase:   int = 3


# ── Routing result (internal use, also serializable) ─────────────────────────

class RoutingResult(BaseModel):
    task:           str
    routing_reason: str
    required_inputs: str          # "single_image" | "image_pair" | "optical_sar_pair"
    tool_name:      str
