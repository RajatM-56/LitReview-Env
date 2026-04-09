"""Deterministic graders for each task difficulty level.

Each grader takes the agent's parsed response and the ground truth,
then returns a score in [0, 1] along with a detailed breakdown.

Design principles:
- Reward correct content even when paraphrased differently
- Use recall-oriented metrics (don't penalize extra good info)
- Provide smooth gradients for partial credit
- Detect hallucination and format errors
"""

from __future__ import annotations

from typing import Any

from litreview_env.schemas import TaskDifficulty
from litreview_env.utils import (
    keyword_overlap,
    list_coverage,
    semantic_similarity,
    word_recall,
    check_paper_ids_grounded,
)


# ---------------------------------------------------------------------------
# Easy grader: single-paper extraction
# ---------------------------------------------------------------------------

_EASY_FIELDS = ["problem", "method", "dataset_or_domain", "key_finding", "limitation", "confidence"]
_EASY_FIELD_WEIGHTS = {
    "problem": 0.20,
    "method": 0.20,
    "dataset_or_domain": 0.15,
    "key_finding": 0.25,
    "limitation": 0.15,
    "confidence": 0.05,
}


def grade_easy(response: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Grade an easy (single-paper extraction) task.

    Scores:
    - Format validity: are all required fields present?
    - Per-field similarity using blended metrics
    - Bonus for keyword recall (captures meaning despite paraphrasing)

    Returns:
        A dict with 'score' (float) and 'breakdown' (dict).
    """
    breakdown: dict[str, Any] = {}

    # Check format: are all required fields present?
    present_fields = [f for f in _EASY_FIELDS if f in response and response[f]]
    missing_fields = [f for f in _EASY_FIELDS if f not in response or not response[f]]
    format_score = len(present_fields) / len(_EASY_FIELDS)
    breakdown["format_score"] = round(format_score, 3)
    breakdown["missing_fields"] = missing_fields

    # Per-field similarity
    field_scores: dict[str, float] = {}
    weighted_sum = 0.0
    for field in _EASY_FIELDS:
        pred = str(response.get(field, "")).strip()
        ref = str(ground_truth.get(field, "")).strip()
        if not ref:
            field_scores[field] = 1.0 if not pred else 0.5
        elif field == "confidence":
            # Exact match for confidence level
            field_scores[field] = 1.0 if pred.lower() == ref.lower() else 0.0
        else:
            # Blend of multiple metrics for robustness
            sim = semantic_similarity(pred, ref)
            kw = keyword_overlap(pred, ref)
            wr = word_recall(pred, ref)
            # Weight keyword/recall more than character similarity
            field_scores[field] = 0.35 * sim + 0.35 * kw + 0.30 * wr
        weighted_sum += field_scores[field] * _EASY_FIELD_WEIGHTS[field]

    breakdown["field_scores"] = {k: round(v, 3) for k, v in field_scores.items()}
    breakdown["content_score"] = round(weighted_sum, 3)

    # Final score blends format and content
    raw_score = 0.10 * format_score + 0.90 * weighted_sum
    score = max(0.001, min(0.999, raw_score))
    breakdown["final_score"] = round(score, 3)

    return {"score": round(score, 4), "breakdown": breakdown}


# ---------------------------------------------------------------------------
# Medium grader: multi-paper comparison
# ---------------------------------------------------------------------------

_MEDIUM_FIELDS = ["shared_theme", "methodological_differences", "result_differences", "tradeoffs"]
_MEDIUM_FIELD_WEIGHTS = {
    "shared_theme": 0.20,
    "methodological_differences": 0.30,
    "result_differences": 0.25,
    "tradeoffs": 0.25,
}


def grade_medium(response: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Grade a medium (comparison) task.

    Scores:
    - Format validity
    - shared_theme similarity
    - List coverage for methodological_differences, result_differences, tradeoffs

    Returns:
        A dict with 'score' (float) and 'breakdown' (dict).
    """
    breakdown: dict[str, Any] = {}

    # Format check
    present = [f for f in _MEDIUM_FIELDS if f in response and response[f]]
    format_score = len(present) / len(_MEDIUM_FIELDS)
    breakdown["format_score"] = round(format_score, 3)

    field_scores: dict[str, float] = {}

    # shared_theme: string similarity
    pred_theme = str(response.get("shared_theme", ""))
    ref_theme = str(ground_truth.get("shared_theme", ""))
    theme_sim = semantic_similarity(pred_theme, ref_theme)
    theme_kw = keyword_overlap(pred_theme, ref_theme)
    theme_wr = word_recall(pred_theme, ref_theme)
    field_scores["shared_theme"] = 0.35 * theme_sim + 0.35 * theme_kw + 0.30 * theme_wr

    # List fields: coverage score with lower thresholds
    for field in ["methodological_differences", "result_differences", "tradeoffs"]:
        pred_list = response.get(field, [])
        ref_list = ground_truth.get(field, [])
        if isinstance(pred_list, str):
            pred_list = [pred_list]
        if isinstance(ref_list, str):
            ref_list = [ref_list]
        if not isinstance(pred_list, list):
            pred_list = []
        if not isinstance(ref_list, list):
            ref_list = []
        field_scores[field] = list_coverage(
            [str(x) for x in pred_list],
            [str(x) for x in ref_list],
            threshold=0.25,  # Lower threshold for paraphrased content
        )

    breakdown["field_scores"] = {k: round(v, 3) for k, v in field_scores.items()}

    weighted_sum = sum(
        field_scores[f] * _MEDIUM_FIELD_WEIGHTS[f] for f in _MEDIUM_FIELDS
    )
    breakdown["content_score"] = round(weighted_sum, 3)

    # Paper ID grounding bonus
    full_text = str(response)
    valid_ids = ground_truth.get("paper_ids", [])
    grounding, hallucinated = check_paper_ids_grounded(full_text, valid_ids)
    breakdown["grounding_score"] = round(grounding, 3)
    breakdown["hallucinated_ids"] = hallucinated

    raw_score = 0.10 * format_score + 0.80 * weighted_sum + 0.10 * grounding
    score = max(0.001, min(0.999, raw_score))
    breakdown["final_score"] = round(score, 3)

    return {"score": round(score, 4), "breakdown": breakdown}


# ---------------------------------------------------------------------------
# Hard grader: full literature review synthesis
# ---------------------------------------------------------------------------

_HARD_FIELDS = [
    "overview", "themes", "evidence_points", "contradictions",
    "limitations", "research_gaps", "future_directions",
]
_HARD_FIELD_WEIGHTS = {
    "overview": 0.10,
    "themes": 0.15,
    "evidence_points": 0.20,
    "contradictions": 0.10,
    "limitations": 0.15,
    "research_gaps": 0.15,
    "future_directions": 0.15,
}


def grade_hard(response: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
    """Grade a hard (synthesis) task.

    Scores:
    - Format validity (all 7 sections present)
    - overview similarity
    - List coverage for all list fields
    - Paper ID grounding

    Returns:
        A dict with 'score' (float) and 'breakdown' (dict).
    """
    breakdown: dict[str, Any] = {}

    # Format
    present = [f for f in _HARD_FIELDS if f in response and response[f]]
    format_score = len(present) / len(_HARD_FIELDS)
    breakdown["format_score"] = round(format_score, 3)

    field_scores: dict[str, float] = {}

    # overview: string similarity blend
    pred_overview = str(response.get("overview", ""))
    ref_overview = str(ground_truth.get("overview", ""))
    ov_sim = semantic_similarity(pred_overview, ref_overview)
    ov_kw = keyword_overlap(pred_overview, ref_overview)
    ov_wr = word_recall(pred_overview, ref_overview)
    field_scores["overview"] = 0.35 * ov_sim + 0.35 * ov_kw + 0.30 * ov_wr

    # List fields
    for field in _HARD_FIELDS:
        if field == "overview":
            continue
        pred_list = response.get(field, [])
        ref_list = ground_truth.get(field, [])
        if isinstance(pred_list, str):
            pred_list = [pred_list]
        if isinstance(ref_list, str):
            ref_list = [ref_list]
        if not isinstance(pred_list, list):
            pred_list = []
        if not isinstance(ref_list, list):
            ref_list = []

        # Lower thresholds for harder matching
        threshold = 0.20 if field == "evidence_points" else 0.25
        field_scores[field] = list_coverage(
            [str(x) for x in pred_list],
            [str(x) for x in ref_list],
            threshold=threshold,
        )

    breakdown["field_scores"] = {k: round(v, 3) for k, v in field_scores.items()}

    weighted_sum = sum(
        field_scores[f] * _HARD_FIELD_WEIGHTS[f] for f in _HARD_FIELDS
    )
    breakdown["content_score"] = round(weighted_sum, 3)

    # Grounding check
    full_text = str(response)
    valid_ids = ground_truth.get("paper_ids", [])
    grounding, hallucinated = check_paper_ids_grounded(full_text, valid_ids)
    breakdown["grounding_score"] = round(grounding, 3)
    breakdown["hallucinated_ids"] = hallucinated

    raw_score = 0.10 * format_score + 0.75 * weighted_sum + 0.15 * grounding
    score = max(0.001, min(0.999, raw_score))
    breakdown["final_score"] = round(score, 3)

    return {"score": round(score, 4), "breakdown": breakdown}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def grade(
    difficulty: TaskDifficulty,
    response: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch grading to the appropriate difficulty-specific grader.

    Args:
        difficulty: The task difficulty level.
        response: The agent's parsed response dict.
        ground_truth: The ground truth annotation dict.

    Returns:
        A dict with 'score' and 'breakdown'.
    """
    graders = {
        TaskDifficulty.EASY: grade_easy,
        TaskDifficulty.MEDIUM: grade_medium,
        TaskDifficulty.HARD: grade_hard,
    }
    grader_fn = graders[difficulty]
    return grader_fn(response, ground_truth)
