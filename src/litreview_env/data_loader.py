"""Data loader for LitReview-Env task instances."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from litreview_env.schemas import TaskDifficulty, TaskInstance


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _resolve_data_dir() -> Path:
    """Resolve the data directory, checking env var override first."""
    env_override = os.environ.get("LITREVIEW_DATA_DIR")
    if env_override:
        return Path(env_override)
    return _DATA_DIR


def load_tasks(difficulty: Optional[TaskDifficulty] = None) -> list[TaskInstance]:
    """Load task instances, optionally filtered by difficulty.

    Args:
        difficulty: If provided, only load tasks of this difficulty.

    Returns:
        A list of TaskInstance objects.
    """
    data_dir = _resolve_data_dir()
    tasks: list[TaskInstance] = []

    difficulties = [difficulty] if difficulty else list(TaskDifficulty)

    for diff in difficulties:
        task_file = data_dir / diff.value / "tasks.json"
        if not task_file.exists():
            continue
        with open(task_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            tasks.append(TaskInstance(**item))

    return tasks


def load_task_by_id(task_id: str) -> Optional[TaskInstance]:
    """Load a specific task by its ID.

    Args:
        task_id: The unique task identifier.

    Returns:
        The matching TaskInstance, or None if not found.
    """
    all_tasks = load_tasks()
    for task in all_tasks:
        if task.task_id == task_id:
            return task
    return None


def list_task_ids(difficulty: Optional[TaskDifficulty] = None) -> list[str]:
    """Return all available task IDs, optionally filtered by difficulty."""
    return [t.task_id for t in load_tasks(difficulty)]
