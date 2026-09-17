"""
tools/base.py
==============
Common interface and result dataclass shared by all specialist tools.

Every specialist tool must:
  1. Inherit from BaseSpecialistTool
  2. Implement run() returning a ToolResult

ToolResult is a plain dataclass — JSON-serializable, no torch tensors.

Confidence policy
-----------------
  confidence_type = "calibrated"   → output of a proper scoring model
  confidence_type = "heuristic"    → computed from a proxy metric
                                     (e.g. L2 feature distance)
  confidence_type = "not_available" → model produces no confidence at all

NEVER mark heuristic values as calibrated.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class ToolResult:
    """
    Structured, serializable result returned by every specialist tool.

    Fields
    ------
    task             : human-readable name of the task performed
    success          : True if the tool completed without errors
    answer           : primary text answer / summary
    confidence       : float in [0, 1] or None
    confidence_type  : 'calibrated' | 'heuristic' | 'not_available'
    visual_evidence  : dict of extra visual/spatial info (boxes, maps, etc.)
                       or None if not applicable
    metadata         : arbitrary key-value pairs (shapes, model names, etc.)
    model_used       : identifier of the model(s) used
    processing_time  : wall-clock seconds for the tool's run() call
    error            : error message if success=False, else None
    """
    task:             str
    success:          bool
    answer:           str
    confidence:       float | None
    confidence_type:  str                        # calibrated|heuristic|not_available
    visual_evidence:  dict | None
    metadata:         dict = field(default_factory=dict)
    model_used:       str  = ""
    processing_time:  float = 0.0
    error:            str | None = None

    def to_dict(self) -> dict:
        """Return a plain dict (safe for JSON serialisation)."""
        return {
            "task":             self.task,
            "success":          self.success,
            "answer":           self.answer,
            "confidence":       self.confidence,
            "confidence_type":  self.confidence_type,
            "visual_evidence":  self.visual_evidence,
            "metadata":         self.metadata,
            "model_used":       self.model_used,
            "processing_time":  round(self.processing_time, 3),
            "error":            self.error,
        }


import re

def clean_tool_output(text: str, query: str = "") -> str:
    """
    Lightweight post-processing to strip obvious generation artifacts
    (e.g., multiple-choice options, meta-instructional preamble)
    without aggressively rewriting the model's actual answer.
    """
    if not text:
        return ""

    query_lower = query.lower()
    user_wanted_options = any(k in query_lower for k in ["option", "choice", "multiple choice", "a)", "b)"])
    user_wanted_calculation = any(k in query_lower for k in ["calculate", "formula", "compute", "percentage"])

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Check options / multiple-choice artifacts if user did not request them
        if not user_wanted_options:
            if re.match(r"^(options\s+are\s*:?|options\s*:?)$", stripped, re.IGNORECASE):
                continue
            if re.match(r"^[A-Da-d]\)\s*.*", stripped) and not any(k in stripped.lower() for k in ["answer", "result", "observation", "change", "feature"]):
                continue
            if re.match(r"^[1-4]\.\s+(desert|forest|wetland|grassland|urban|water|agriculture).*", stripped, re.IGNORECASE):
                continue

        # Check meta-instructional artifacts
        if not user_wanted_calculation:
            if re.match(r"^(also,\s+)?calculate\s+(the\s+)?.*percentage.*", stripped, re.IGNORECASE):
                continue

        if re.match(r"^(to\s+(?:determine|identify|analyze|find)|first,\s+we\s+need\s+to|finally,\s+provide\s+a\s+report).*", stripped, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result if result else text.strip()


class BaseSpecialistTool(ABC):
    """Abstract base that every specialist tool must implement."""

    #: Human-readable name shown in reports
    TOOL_NAME: str = "base"

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> ToolResult:
        """
        Execute the tool and return a ToolResult.

        Implementations must:
        - Catch internal errors and return success=False with an error message
          rather than letting exceptions propagate unchecked.
        - Never fabricate confidence values or visual evidence.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} tool_name={self.TOOL_NAME!r}>"
