"""Reward shaping for LitReview-Env.

Provides continuous reward signals that:
- Reward partial progress on intermediate steps
- Penalize invalid JSON, hallucinations, repetition, and no-ops
- Scale from 0.0 to 1.0 for the final submission score
"""

from __future__ import annotations

from typing import Any

from litreview_env.schemas import ActionType, LitReviewState, TaskDifficulty
from litreview_env.graders import grade
from litreview_env.utils import safe_parse_json


# ---------------------------------------------------------------------------
# Penalty constants
# ---------------------------------------------------------------------------

PENALTY_INVALID_JSON = -0.15
PENALTY_NOP = -0.10
PENALTY_REPEATED_NOP = -0.20
PENALTY_EMPTY_SUBMIT = -0.10
PENALTY_HALLUCINATED_IDS = -0.05  # Per hallucinated ID
HINT_COST = -0.05  # Cost per hint request

# Bonus constants
BONUS_FIRST_VALID_SUBMIT = 0.05
BONUS_IMPROVEMENT = 0.02


def compute_reward(
    action_type: ActionType,
    content: str,
    difficulty: TaskDifficulty,
    ground_truth: dict[str, Any],
    state: LitReviewState,
    valid_paper_ids: list[str],
) -> tuple[float, str, dict[str, Any]]:
    """Compute the reward for a single step.

    Args:
        action_type: The type of action taken.
        content: The content of the action (for SUBMIT actions).
        difficulty: The current task difficulty.
        ground_truth: The ground truth annotations.
        state: The current episode state.
        valid_paper_ids: Paper IDs in the current corpus.

    Returns:
        Tuple of (reward, feedback_message, grading_info).
    """
    info: dict[str, Any] = {}

    # --- NOP action ---
    if action_type == ActionType.NOP:
        if state.nop_count >= 2:
            return PENALTY_REPEATED_NOP, "Repeated no-op actions are penalized. Please submit a response.", info
        return PENALTY_NOP, "No-op action taken. Use 'submit' to provide your analysis.", info

    # --- HINT action ---
    if action_type == ActionType.REQUEST_HINT:
        return HINT_COST, _generate_hint(difficulty, state.hint_count), info

    # --- SUBMIT action ---
    assert action_type == ActionType.SUBMIT

    # Empty submission
    if not content or not content.strip():
        return PENALTY_EMPTY_SUBMIT, "Empty submission. Please provide a structured JSON response.", info

    # Parse JSON
    parsed = safe_parse_json(content)
    if parsed is None:
        return PENALTY_INVALID_JSON, (
            "Could not parse your response as valid JSON. "
            "Please return a well-formed JSON object with the required fields."
        ), info

    # Grade the response
    grading_result = grade(difficulty, parsed, ground_truth)
    score = grading_result["score"]
    breakdown = grading_result["breakdown"]
    info["grading"] = grading_result

    # Base reward from grading score (0 to 1)
    reward = score

    # Bonus for first valid submission
    if state.submission_count == 0:
        reward += BONUS_FIRST_VALID_SUBMIT
        info["first_submit_bonus"] = True

    # Bonus for improvement over previous best
    if score > state.best_score and state.submission_count > 0:
        reward += BONUS_IMPROVEMENT
        info["improvement_bonus"] = True

    # Hallucination penalty
    hallucinated = breakdown.get("hallucinated_ids", [])
    if hallucinated:
        penalty = PENALTY_HALLUCINATED_IDS * len(hallucinated)
        reward += penalty
        info["hallucination_penalty"] = penalty

    # Clamp to [-1, 1]
    reward = max(-1.0, min(1.0, reward))

    # Build feedback message
    feedback = _build_feedback(score, breakdown, difficulty)

    return round(reward, 4), feedback, info


def _generate_hint(difficulty: TaskDifficulty, hint_count: int) -> str:
    """Generate a progressive hint based on difficulty and how many hints used."""
    hints = {
        TaskDifficulty.EASY: [
            "Hint: Make sure your JSON includes all six required fields: problem, method, dataset_or_domain, key_finding, limitation, and confidence.",
            "Hint: Focus on what the abstract explicitly states. The 'problem' should describe what challenge the paper addresses.",
            "Hint: For 'confidence', use one of: high, medium, or low. Base it on how clearly the abstract states the information.",
        ],
        TaskDifficulty.MEDIUM: [
            "Hint: Your response needs four keys: shared_theme, methodological_differences, result_differences, and tradeoffs. Each except shared_theme should be a list.",
            "Hint: For methodological_differences, describe how each paper's approach is distinct. Reference paper IDs.",
            "Hint: Think about what practical tradeoffs exist between the approaches. What does each method gain or sacrifice?",
        ],
        TaskDifficulty.HARD: [
            "Hint: Your review needs seven sections: overview, themes, evidence_points, contradictions, limitations, research_gaps, and future_directions.",
            "Hint: evidence_points should reference specific paper IDs and cite concrete results or claims from the abstracts.",
            "Hint: research_gaps should identify what is NOT covered or underexplored across the papers, not just restate limitations.",
            "Hint: Look for tensions between papers. Do any papers disagree or approach the problem in contradictory ways?",
        ],
    }
    task_hints = hints.get(difficulty, hints[TaskDifficulty.EASY])
    idx = min(hint_count, len(task_hints) - 1)
    return task_hints[idx]


def _build_feedback(score: float, breakdown: dict[str, Any], difficulty: TaskDifficulty) -> str:
    """Build human-readable feedback from the grading breakdown."""
    parts = [f"Score: {score:.2f}/1.00"]

    format_score = breakdown.get("format_score", 0)
    parts.append(f"Format: {format_score:.0%} of required fields present")

    missing = breakdown.get("missing_fields", [])
    if missing:
        parts.append(f"Missing fields: {', '.join(missing)}")

    content_score = breakdown.get("content_score", 0)
    parts.append(f"Content quality: {content_score:.2f}")

    field_scores = breakdown.get("field_scores", {})
    if field_scores:
        low_fields = [f for f, s in field_scores.items() if s < 0.4]
        if low_fields:
            parts.append(f"Weak areas: {', '.join(low_fields)}")

    grounding = breakdown.get("grounding_score")
    if grounding is not None and grounding < 1.0:
        parts.append(f"Grounding: {grounding:.0%} (some references may be incorrect)")

    hallucinated = breakdown.get("hallucinated_ids", [])
    if hallucinated:
        parts.append(f"Hallucinated paper IDs: {', '.join(hallucinated)}")

    return " | ".join(parts)
