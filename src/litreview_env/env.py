"""LitReview-Env: Core environment implementation.

Implements the OpenEnv interface: reset(), step(action), state().
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from litreview_env.schemas import (
    ActionType,
    LitReviewAction,
    LitReviewObservation,
    LitReviewState,
    StepResult,
    TaskDifficulty,
    TaskInstance,
)
from litreview_env.tasks import TaskManager
from litreview_env.rewards import compute_reward


class LitReviewEnvironment:
    """OpenEnv-compatible environment for academic literature review benchmarking.

    The environment presents the agent with paper abstracts and a task instruction.
    The agent must produce a structured, evidence-grounded analysis.

    Supports three difficulty levels:
    - easy: Extract key information from a single paper abstract
    - medium: Compare and contrast three related papers
    - hard: Synthesize a full literature review from eight papers

    Methods:
        reset(task_id=None, difficulty=None) -> LitReviewObservation
        step(action) -> StepResult
        state -> LitReviewState
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the environment.

        Args:
            seed: Random seed for reproducible task selection.
        """
        self._task_manager = TaskManager(seed=seed)
        self._state = LitReviewState()
        self._current_task: Optional[TaskInstance] = None
        self._history: list[dict[str, Any]] = []

    def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> LitReviewObservation:
        """Initialize a new episode.

        Args:
            task_id: Specific task to load. If None, a random task is selected.
            difficulty: Difficulty filter ('easy', 'medium', 'hard').
                       Only used when task_id is None.

        Returns:
            Initial observation with task instruction and paper abstracts.
        """
        # Select task
        if task_id:
            task = self._task_manager.get_task(task_id)
            if task is None:
                raise ValueError(f"Task '{task_id}' not found. Available: {self._task_manager.list_tasks()}")
        else:
            diff = TaskDifficulty(difficulty) if difficulty else None
            task = self._task_manager.get_random_task(diff)

        self._current_task = task
        self._history = []

        # Initialize state
        self._state = LitReviewState(
            episode_id=str(uuid.uuid4()),
            task_id=task.task_id,
            difficulty=task.difficulty.value,
            step_count=0,
            max_steps=task.max_steps,
            done=False,
            cumulative_reward=0.0,
            last_action_type="",
            hint_count=0,
            submission_count=0,
            nop_count=0,
            best_score=0.0,
        )

        return LitReviewObservation(
            task_id=task.task_id,
            difficulty=task.difficulty.value,
            instruction=task.instruction,
            papers=task.papers,
            feedback="Environment reset. Read the abstracts and submit your structured analysis.",
            hint="",
            step_number=0,
            max_steps=task.max_steps,
            done=False,
        )

    def step(self, action: LitReviewAction) -> StepResult:
        """Execute an action and return the result.

        Args:
            action: The action to execute.

        Returns:
            StepResult with observation, reward, done flag, and info.

        Raises:
            RuntimeError: If the environment has not been reset or the episode is done.
        """
        if self._current_task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        task = self._current_task

        # Update state counters
        self._state.step_count += 1
        self._state.last_action_type = action.action_type.value

        if action.action_type == ActionType.NOP:
            self._state.nop_count += 1
        elif action.action_type == ActionType.REQUEST_HINT:
            self._state.hint_count += 1
        elif action.action_type == ActionType.SUBMIT:
            self._state.submission_count += 1

        # Compute reward
        valid_paper_ids = [p.paper_id for p in task.papers]
        reward, feedback, info = compute_reward(
            action_type=action.action_type,
            content=action.content,
            difficulty=task.difficulty,
            ground_truth=task.ground_truth,
            state=self._state,
            valid_paper_ids=valid_paper_ids,
        )

        # Update best score from grading
        grading = info.get("grading", {})
        current_score = grading.get("score", 0.0)
        if current_score > self._state.best_score:
            self._state.best_score = current_score

        # Update cumulative reward
        self._state.cumulative_reward += reward

        # Check termination
        done = False
        if self._state.step_count >= self._state.max_steps:
            done = True
            feedback += " | Episode ended: maximum steps reached."
        elif action.action_type == ActionType.SUBMIT and current_score >= 0.95:
            done = True
            feedback += " | Excellent! Near-perfect score achieved."

        self._state.done = done

        # Build hint text
        hint = ""
        if action.action_type == ActionType.REQUEST_HINT:
            hint = feedback
            feedback = f"Hint requested (cost: {reward})"

        # Record history
        self._history.append({
            "step": self._state.step_count,
            "action_type": action.action_type.value,
            "reward": reward,
            "score": current_score,
            "done": done,
        })

        observation = LitReviewObservation(
            task_id=task.task_id,
            difficulty=task.difficulty.value,
            instruction=task.instruction,
            papers=task.papers,
            feedback=feedback,
            hint=hint,
            step_number=self._state.step_count,
            max_steps=self._state.max_steps,
            done=done,
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info={
                "grading": grading,
                "cumulative_reward": round(self._state.cumulative_reward, 4),
                "best_score": round(self._state.best_score, 4),
                "history": self._history,
                **{k: v for k, v in info.items() if k != "grading"},
            },
        )

    @property
    def state(self) -> LitReviewState:
        """Access current episode-level metadata."""
        return self._state

    def list_tasks(self, difficulty: Optional[str] = None) -> list[str]:
        """List available task IDs."""
        diff = TaskDifficulty(difficulty) if difficulty else None
        return self._task_manager.list_tasks(diff)

    @property
    def task_count(self) -> int:
        """Total number of tasks."""
        return self._task_manager.task_count
