"""Task management for LitReview-Env.

Handles task selection, cycling, and validation.
"""

from __future__ import annotations

import random
from typing import Optional

from litreview_env.data_loader import load_tasks, load_task_by_id, list_task_ids
from litreview_env.schemas import TaskDifficulty, TaskInstance


class TaskManager:
    """Manages task loading and selection for the environment."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._tasks: dict[str, TaskInstance] = {}
        self._rng = random.Random(seed)
        self._load_all()

    def _load_all(self) -> None:
        """Load all tasks from disk."""
        all_tasks = load_tasks()
        for task in all_tasks:
            self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[TaskInstance]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_random_task(self, difficulty: Optional[TaskDifficulty] = None) -> TaskInstance:
        """Get a random task, optionally filtered by difficulty.

        Args:
            difficulty: If provided, only select from this difficulty.

        Returns:
            A randomly selected TaskInstance.

        Raises:
            ValueError: If no tasks are available.
        """
        candidates = list(self._tasks.values())
        if difficulty:
            candidates = [t for t in candidates if t.difficulty == difficulty]
        if not candidates:
            raise ValueError(f"No tasks available for difficulty={difficulty}")
        return self._rng.choice(candidates)

    def list_tasks(self, difficulty: Optional[TaskDifficulty] = None) -> list[str]:
        """List all task IDs, optionally filtered by difficulty."""
        tasks = list(self._tasks.values())
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]
        return [t.task_id for t in tasks]

    @property
    def task_count(self) -> int:
        """Total number of loaded tasks."""
        return len(self._tasks)

    def difficulties(self) -> list[str]:
        """Return all available difficulty levels."""
        return sorted(set(t.difficulty.value for t in self._tasks.values()))
