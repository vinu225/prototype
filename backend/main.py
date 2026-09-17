"""
backend/main.py
================
SatQuery AI — Phase 3 FastAPI application.

Start with:
    python -m uvicorn backend.main:app --reload

Endpoints
---------
  GET  /            — Welcome message
  GET  /health      — Service health check
  GET  /tools       — Discover available specialist tools
  POST /query       — NL query + satellite image(s) → auto-routed analysis
  POST /analyse     — Explicit task + satellite image(s) → specialist execution

Design notes
------------
- All specialist tools share a single VLM via models/model_pool.py.
- No external LLM APIs (Gemini, OpenAI, etc.) are used.
- No PostgreSQL, vector databases, or authentication.
- Python tracebacks are never exposed through the API response.
- All uploaded satellite images are NumPy .npy arrays [C,H,W] or [B,C,H,W].
- Input must be 12 channels, 120×120 spatial — no silent reshaping.
"""

from __future__ import annotations

import logging
import sys
import io
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from backend.audit          import generate_session_id, write_audit
from backend.input_validation import (
    InputValidationError,
    validate_single_image,
    validate_image_pair,
    validate_optical_sar_pair,
)
from backend.router import SatQueryRouter, RouterError, UnknownQueryError, AmbiguousQueryError
from backend.schemas import (
    AuditInfo,
    ErrorResponse,
    HealthResponse,
    SatQueryResponse,
    ToolInfo,
)
from backend.tool_registry import get_registry, get_tool

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_LOG = logging.getLogger(__name__)

# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SatQuery AI",
    description = (
        "Remote-sensing multimodal AI — Phase 3 API.\n\n"
        "Upload 12-channel (120×120) satellite .npy arrays and ask "
        "natural-language questions. The backend automatically routes your "
        "query to the correct specialist tool."
    ),
    version     = "0.3.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Single shared router instance
_router = SatQueryRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_error(code: str, message: str, status_code: int = 400,
                task: str | None = None, query: str | None = None) -> JSONResponse:
    body = ErrorResponse(success=False, error=code, message=message,
                         task=task, query=query)
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _tool_result_to_response(
    result,
    query:           str | None,
    session_id:      str,
    routing_reason:  str,
    input_files:     list[str],
    input_metadata:  dict,
) -> SatQueryResponse:
    """Convert a ToolResult to a SatQueryResponse and write an audit record."""
    audit_info = AuditInfo(
        session_id     = session_id,
        timestamp      = __import__("datetime").datetime.utcnow().isoformat() + "Z",
        model_used     = result.model_used,
        tool           = result.task,
        routing_reason = routing_reason,
        input_files    = input_files,
        query          = query,
    )
    write_audit(
        session_id      = session_id,
        query           = query,
        task            = result.task,
        tool            = result.task,
        model_used      = result.model_used,
        input_files     = input_files,
        input_metadata  = input_metadata,
        routing_reason  = routing_reason,
        processing_time = result.processing_time,
        confidence      = result.confidence,
        confidence_type = result.confidence_type,
        success         = result.success,
        error           = result.error,
    )
    return SatQueryResponse(
        success          = result.success,
        task             = result.task,
        query            = query,
        answer           = result.answer,
        confidence       = result.confidence,
        confidence_type  = result.confidence_type,
        visual_evidence  = result.visual_evidence,
        processing_time  = result.processing_time,
        audit            = audit_info,
        model_used       = result.model_used,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "SatQuery AI",
        "phase":   3,
        "docs":    "/docs",
        "health":  "/health",
        "tools":   "/tools",
    }


@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health():
    """Return service health status."""
    return HealthResponse(status="ok", service="SatQuery AI", phase=3)


@app.get("/tools", response_model=list[ToolInfo], tags=["Service"])
def list_tools():
    """
    Discover available specialist tools.

    **grounding** is listed but currently returns `not_implemented` — no
    fabricated bounding-box coordinates are produced.
    """
    return [ToolInfo(**t) for t in _router.list_tasks()]


@app.post(
    "/query",
    response_model = SatQueryResponse,
    tags           = ["Analysis"],
    summary        = "Auto-route NL query to correct specialist tool",
    description    = (
        "Upload one or more **12-channel 120×120 satellite .npy** files and a "
        "natural-language question.  The backend automatically routes the query "
        "to the correct specialist tool (single_vqa, captioning, grounding, "
        "change_detection, or optical_sar_fusion).\n\n"
        "**File format**: NumPy `.npy` array with shape `[12, 120, 120]` or "
        "`[B, 12, 120, 120]`."
    ),
)
async def query_endpoint(
    query: str = Form(..., description="Natural-language question about the satellite image(s)."),
    files: List[UploadFile] = File(..., description="12-channel satellite .npy file(s)."),
):
    """
    Accepts a natural-language query and one or more satellite images.
    Routes automatically to the correct specialist tool.
    """
    session_id = generate_session_id()
    _LOG.info("[%s] /query — %r, files=%s", session_id, query, [f.filename for f in files])

    # ── Route the query ───────────────────────────────────────────────────────
    try:
        routing = _router.route(query)
    except (UnknownQueryError, AmbiguousQueryError) as exc:
        return _make_error("unknown_task", str(exc), query=query)
    except ValueError as exc:
        return _make_error("invalid_query", str(exc), query=query)

    task           = routing["task"]
    routing_reason = routing["routing_reason"]
    required       = routing["required_inputs"]
    _LOG.info("[%s] routed → %s", session_id, task)

    # ── Validate + load files ─────────────────────────────────────────────────
    try:
        if required == "single_image":
            arr, meta = await validate_single_image(files)
            arrs      = [arr]
        elif required == "image_pair":
            before, after, meta = await validate_image_pair(files)
            arrs = [before, after]
        elif required == "optical_sar_pair":
            optical, sar, meta = await validate_optical_sar_pair(files)
            arrs = [optical, sar]
        else:
            return _make_error("internal_error", f"Unknown required_inputs: {required!r}")
    except InputValidationError as exc:
        return _make_error(exc.code, exc.message, status_code=200, query=query)

    input_files    = [f.filename for f in files]
    input_metadata = meta

    # ── Execute tool ──────────────────────────────────────────────────────────
    try:
        tool = get_tool(task)
    except KeyError as exc:
        return _make_error("unknown_task", str(exc), task=task, query=query)

    try:
        if task == "single_vqa":
            result = tool.run(arrs[0], query=query)
        elif task == "captioning":
            result = tool.run(arrs[0], custom_prompt=query)
        elif task == "grounding":
            result = tool.run(arrs[0], query=query)
        elif task == "change_detection":
            result = tool.run(arrs[0], arrs[1] if len(arrs) > 1 else arrs[0], query=query)
        elif task == "optical_sar_fusion":
            result = tool.run(arrs[0], arrs[1] if len(arrs) > 1 else arrs[0], query=query)
        else:
            return _make_error("unknown_task", f"No handler for task: {task!r}")
    except Exception as exc:
        _LOG.exception("[%s] Tool %r raised exception", session_id, task)
        return _make_error("tool_failure",
                           f"The specialist tool encountered an unexpected error. "
                           f"Check server logs for details.",
                           task=task, query=query)

    response = _tool_result_to_response(
        result, query, session_id, routing_reason, input_files, input_metadata
    )
    return JSONResponse(status_code=200, content=response.model_dump())


@app.post(
    "/analyse",
    response_model = SatQueryResponse,
    tags           = ["Analysis"],
    summary        = "Explicit task-directed satellite image analysis",
    description    = (
        "Upload one or more **12-channel 120×120 satellite .npy** files and "
        "specify an explicit `task`.  Supported tasks:\n\n"
        "- `single_vqa` — requires 1 file + query\n"
        "- `captioning` — requires 1 file\n"
        "- `grounding` — requires 1 file + query *(not implemented)*\n"
        "- `change_detection` — requires 2 files\n"
        "- `optical_sar_fusion` — requires 2 files\n\n"
        "**File format**: NumPy `.npy` array with shape `[12, 120, 120]` or "
        "`[B, 12, 120, 120]`."
    ),
)
async def analyse_endpoint(
    task: str = Form(..., description=(
        "Explicit task name: single_vqa | captioning | grounding | "
        "change_detection | optical_sar_fusion"
    )),
    files: List[UploadFile] = File(..., description="12-channel satellite .npy file(s)."),
    query: str | None = Form(None, description="Optional natural-language query (required for vqa/grounding)."),
):
    """
    Explicit task-directed endpoint.  No automatic routing — task is specified directly.
    """
    session_id = generate_session_id()
    _LOG.info("[%s] /analyse — task=%r, query=%r, files=%s",
              session_id, task, query, [f.filename for f in files])

    # ── Validate task name ────────────────────────────────────────────────────
    valid_tasks = {"single_vqa", "captioning", "grounding", "change_detection", "optical_sar_fusion"}
    if task not in valid_tasks:
        return _make_error(
            "unknown_task",
            f"Unknown task {task!r}. Valid tasks: {sorted(valid_tasks)}",
            task=task,
        )

    routing_reason = f"Explicit task selected by caller: {task!r}"

    # ── Validate + load files ─────────────────────────────────────────────────
    pair_tasks   = {"change_detection", "optical_sar_fusion"}
    single_tasks = {"single_vqa", "captioning", "grounding"}

    try:
        if task in single_tasks:
            arr, meta = await validate_single_image(files)
            arrs      = [arr]
        else:
            before, after, meta = await validate_image_pair(files)
            arrs = [before, after]
    except InputValidationError as exc:
        return _make_error(exc.code, exc.message, status_code=200, task=task, query=query)

    input_files    = [f.filename for f in files]
    input_metadata = meta

    # ── Execute tool ──────────────────────────────────────────────────────────
    try:
        tool = get_tool(task)
    except KeyError as exc:
        return _make_error("unknown_task", str(exc), task=task)

    try:
        if task == "single_vqa":
            if not query:
                return _make_error("missing_query",
                                   "single_vqa requires a 'query' field.",
                                   task=task)
            result = tool.run(arrs[0], query=query)
        elif task == "captioning":
            result = tool.run(arrs[0], custom_prompt=query)
        elif task == "grounding":
            if not query:
                return _make_error("missing_query",
                                   "grounding requires a 'query' field.",
                                   task=task)
            result = tool.run(arrs[0], query=query)
        elif task == "change_detection":
            result = tool.run(arrs[0], arrs[1], query=query or "What changed between these two observations?")
        elif task == "optical_sar_fusion":
            result = tool.run(arrs[0], arrs[1], query=query or "Using both modalities, describe the land cover.")
        else:
            return _make_error("unknown_task", f"No handler for task: {task!r}")
    except Exception as exc:
        _LOG.exception("[%s] Tool %r raised exception", session_id, task)
        return _make_error("tool_failure",
                           "The specialist tool encountered an unexpected error. "
                           "Check server logs for details.",
                           task=task, query=query)

    response = _tool_result_to_response(
        result, query, session_id, routing_reason, input_files, input_metadata
    )
    return JSONResponse(status_code=200, content=response.model_dump())
