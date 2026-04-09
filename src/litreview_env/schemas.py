"""Pydantic v2 schemas for LitReview-Env actions, observations, state, and grading."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskDifficulty(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, enum.Enum):
    """The set of actions an agent can take."""
    SUBMIT = "submit"          # Submit a structured answer
    REQUEST_HINT = "request_hint"  # Ask for a hint (costs reward)
    NOP = "nop"                # No-operation (penalized)


# ---------------------------------------------------------------------------
# Paper / corpus models
# ---------------------------------------------------------------------------

class PaperAbstract(BaseModel):
    """A single paper abstract provided to the agent."""
    paper_id: str = Field(..., description="Unique identifier for the paper")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Author list")
    year: int = Field(..., description="Publication year")
    venue: str = Field(default="", description="Publication venue")
    abstract: str = Field(..., description="Full abstract text")
    domain: str = Field(default="", description="Research domain or topic area")


# ---------------------------------------------------------------------------
# Ground-truth annotation models (used internally for grading)
# ---------------------------------------------------------------------------

class EasyGroundTruth(BaseModel):
    """Ground truth for an easy (single-paper extraction) task."""
    problem: str
    method: str
    dataset_or_domain: str
    key_finding: str
    limitation: str
    confidence: str  # e.g. "high", "medium", "low"


class MediumGroundTruth(BaseModel):
    """Ground truth for a medium (comparison) task."""
    shared_theme: str
    methodological_differences: list[str]
    result_differences: list[str]
    tradeoffs: list[str]
    paper_ids: list[str]


class HardGroundTruth(BaseModel):
    """Ground truth for a hard (synthesis) task."""
    overview: str
    themes: list[str]
    evidence_points: list[str]
    contradictions: list[str]
    limitations: list[str]
    research_gaps: list[str]
    future_directions: list[str]
    paper_ids: list[str]


# ---------------------------------------------------------------------------
# Task instance
# ---------------------------------------------------------------------------

class TaskInstance(BaseModel):
    """A single graded task instance loaded from the data directory."""
    task_id: str
    difficulty: TaskDifficulty
    instruction: str
    papers: list[PaperAbstract]
    ground_truth: dict[str, Any]  # Validated later per difficulty
    max_steps: int = Field(default=5)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class LitReviewAction(BaseModel):
    """Action submitted by the agent."""
    action_type: ActionType = Field(
        default=ActionType.SUBMIT,
        description="Type of action to take",
    )
    content: str = Field(
        default="",
        description="JSON-formatted structured response for SUBMIT, or free text for other actions",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class LitReviewObservation(BaseModel):
    """Observation returned to the agent after each step or reset."""
    task_id: str = Field(default="")
    difficulty: str = Field(default="")
    instruction: str = Field(default="")
    papers: list[PaperAbstract] = Field(default_factory=list)
    feedback: str = Field(default="")
    hint: str = Field(default="")
    step_number: int = Field(default=0)
    max_steps: int = Field(default=5)
    done: bool = Field(default=False)


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Result returned from step(), combining observation + reward + done."""
    observation: LitReviewObservation
    reward: float = Field(default=0.001, gt=0.0, lt=1.0)
    done: bool = Field(default=False)
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class LitReviewState(BaseModel):
    """Tracks episode-level metadata."""
    episode_id: str = Field(default="")
    task_id: str = Field(default="")
    difficulty: str = Field(default="")
    step_count: int = Field(default=0)
    max_steps: int = Field(default=5)
    done: bool = Field(default=False)
    cumulative_reward: float = Field(default=0.0)
    last_action_type: str = Field(default="")
    hint_count: int = Field(default=0)
    submission_count: int = Field(default=0)
    nop_count: int = Field(default=0)
    best_score: float = Field(default=0.0)
