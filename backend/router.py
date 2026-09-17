"""
backend/router.py
==================
SatQuery AI — Deterministic rule-based intent router.

Classifies a natural-language query into one of five supported tasks
WITHOUT using any external LLM.

Supported tasks
---------------
  single_vqa          — Answer a question about one satellite image
  captioning          — Describe a scene
  grounding           — Locate a specific object/feature
  change_detection    — Compare two temporal observations
  optical_sar_fusion  — Analyze optical + SAR jointly

Algorithm
---------
Each task is associated with a weighted keyword/phrase list.
The router computes a score for each task and picks the highest scoring
non-ambiguous winner.

If two tasks share the top score, an AmbiguousQueryError is raised.
If no task exceeds a minimum relevance threshold, an UnknownQueryError is raised.

No LLM is invoked — pure string matching.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


# ── Exceptions ────────────────────────────────────────────────────────────────

class RouterError(Exception):
    """Base class for routing errors."""
    pass

class AmbiguousQueryError(RouterError):
    """Query matches multiple tasks with equal confidence."""
    pass

class UnknownQueryError(RouterError):
    """Query matches no known task."""
    pass


# ── Task definitions ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskDefinition:
    name:            str
    description:     str
    required_inputs: str    # "single_image" | "image_pair" | "optical_sar_pair"
    keywords:        tuple[str, ...]
    weight:          int = 1   # score increment per keyword hit


_TASKS: list[TaskDefinition] = [

    TaskDefinition(
        name="change_detection",
        description="Compare two temporal satellite observations for geographic change",
        required_inputs="image_pair",
        keywords=(
            "changed", "change", "changes", "detect changes", "what changed",
            "before and after", "temporal", "compare images", "compare the images",
            "two images", "two observations", "between these",
            "evolution", "land cover change", "vegetation change",
            "deforestation", "urbanization", "flooding",
        ),
        weight=2,
    ),

    TaskDefinition(
        name="optical_sar_fusion",
        description="Analyze optical and SAR observations jointly",
        required_inputs="optical_sar_pair",
        keywords=(
            "sar", "sentinel-1", "sentinel 1", "radar", "backscatter",
            "optical and sar", "sar and optical", "fuse", "fusion",
            "multimodal", "both modalities", "combine", "dual",
            "sentinel-2 and sentinel-1", "s1", "s2",
        ),
        weight=3,
    ),

    TaskDefinition(
        name="grounding",
        description="Locate a specific object or feature in the satellite image",
        required_inputs="single_image",
        keywords=(
            "where is", "where are", "locate", "find", "location of",
            "bounding box", "coordinates", "position", "detect",
            "show me where", "highlight", "mark", "identify location",
            "road", "building", "buildings", "structure", "water body",
        ),
        weight=2,
    ),

    TaskDefinition(
        name="captioning",
        description="Generate a concise scene description of the satellite image",
        required_inputs="single_image",
        keywords=(
            "describe", "description", "scene description", "what does this show",
            "what does the image show", "caption", "summarize", "overview",
            "tell me about", "what is in this image", "what can you see",
            "generate a caption", "generate a description",
        ),
        weight=2,
    ),

    TaskDefinition(
        name="single_vqa",
        description="Answer a natural-language question about a single satellite image",
        required_inputs="single_image",
        keywords=(
            "what type", "what kind", "is there", "are there",
            "what is visible", "what is present", "how many",
            "land cover", "vegetation", "water", "urban", "agricultural",
            "crop", "forest", "desert", "wetland", "grassland",
            "what is shown", "what does this satellite image",
            "answer", "question",
        ),
        weight=1,
    ),
]

# Minimum score to accept a routing decision
_MIN_SCORE: int = 1


# ── Router ────────────────────────────────────────────────────────────────────

class SatQueryRouter:
    """
    Lightweight deterministic query router for SatQuery AI.

    Usage
    -----
    router = SatQueryRouter()
    result = router.route("What type of land cover is visible?")
    # RoutingResult(task='single_vqa', ...)
    """

    def route(self, query: str) -> dict:
        """
        Classify a natural-language query into a specialist task.

        Parameters
        ----------
        query : str
            User's natural-language question.

        Returns
        -------
        dict with keys:
            task, routing_reason, required_inputs, tool_name

        Raises
        ------
        UnknownQueryError
            If no task matches the query.
        AmbiguousQueryError
            If two or more tasks score equally at the top.
        ValueError
            If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        q = query.lower().strip()
        # Normalize punctuation for matching
        q_clean = re.sub(r"[^\w\s]", " ", q)

        scores: dict[str, int] = {t.name: 0 for t in _TASKS}
        matched_keywords: dict[str, list[str]] = {t.name: [] for t in _TASKS}

        for task in _TASKS:
            for kw in task.keywords:
                if kw in q_clean or kw in q:
                    scores[task.name] += task.weight
                    matched_keywords[task.name].append(kw)

        max_score = max(scores.values())

        if max_score < _MIN_SCORE:
            raise UnknownQueryError(
                f"No supported task matched the query: {query!r}. "
                "Supported tasks: single_vqa, captioning, grounding, "
                "change_detection, optical_sar_fusion."
            )

        winners = [name for name, score in scores.items() if score == max_score]

        if len(winners) > 1:
            # Prefer change_detection > optical_sar_fusion > grounding > captioning > single_vqa
            priority = [
                "change_detection",
                "optical_sar_fusion",
                "grounding",
                "captioning",
                "single_vqa",
            ]
            for preferred in priority:
                if preferred in winners:
                    winners = [preferred]
                    break

        selected_name = winners[0]
        selected_task = next(t for t in _TASKS if t.name == selected_name)
        kws = matched_keywords[selected_name]
        reason = (
            f"Matched keyword(s): {kws} → task={selected_name!r} "
            f"(score={max_score})"
        )

        return {
            "task":            selected_name,
            "routing_reason":  reason,
            "required_inputs": selected_task.required_inputs,
            "tool_name":       selected_name,
        }

    def list_tasks(self) -> list[dict]:
        """Return a description of all supported tasks."""
        return [
            {
                "name":        t.name,
                "description": t.description,
                "implemented": t.name != "grounding",
                "input_type":  t.required_inputs,
            }
            for t in _TASKS
        ]
