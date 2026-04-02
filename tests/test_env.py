"""Tests for the LitReview-Env environment."""

from __future__ import annotations

import json
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litreview_env.env import LitReviewEnvironment
from litreview_env.schemas import LitReviewAction, ActionType, TaskDifficulty


@pytest.fixture
def env():
    """Create a fresh environment for each test."""
    return LitReviewEnvironment(seed=42)


class TestReset:
    """Tests for the reset() method."""

    def test_reset_random(self, env: LitReviewEnvironment):
        obs = env.reset()
        assert obs.task_id
        assert obs.instruction
        assert len(obs.papers) > 0
        assert not obs.done
        assert obs.step_number == 0

    def test_reset_by_task_id(self, env: LitReviewEnvironment):
        obs = env.reset(task_id="easy_001")
        assert obs.task_id == "easy_001"
        assert obs.difficulty == "easy"
        assert len(obs.papers) == 1

    def test_reset_by_difficulty(self, env: LitReviewEnvironment):
        obs = env.reset(difficulty="medium")
        assert obs.difficulty == "medium"
        assert len(obs.papers) == 3

    def test_reset_hard(self, env: LitReviewEnvironment):
        obs = env.reset(difficulty="hard")
        assert obs.difficulty == "hard"
        assert len(obs.papers) == 8

    def test_reset_invalid_task(self, env: LitReviewEnvironment):
        with pytest.raises(ValueError, match="not found"):
            env.reset(task_id="nonexistent_999")

    def test_reset_clears_state(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        env.step(LitReviewAction(action_type=ActionType.NOP))
        assert env.state.step_count == 1

        env.reset(task_id="easy_001")
        assert env.state.step_count == 0
        assert env.state.cumulative_reward == 0.0
        assert not env.state.done


class TestStep:
    """Tests for the step() method."""

    def test_step_without_reset_raises(self, env: LitReviewEnvironment):
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(LitReviewAction(action_type=ActionType.NOP))

    def test_nop_penalty(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        result = env.step(LitReviewAction(action_type=ActionType.NOP))
        assert result.reward < 0
        assert env.state.nop_count == 1

    def test_hint_cost(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        result = env.step(LitReviewAction(action_type=ActionType.REQUEST_HINT))
        assert result.reward < 0
        assert env.state.hint_count == 1

    def test_invalid_json_penalty(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        result = env.step(LitReviewAction(
            action_type=ActionType.SUBMIT,
            content="not json at all",
        ))
        assert result.reward < 0

    def test_empty_submit_penalty(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        result = env.step(LitReviewAction(
            action_type=ActionType.SUBMIT,
            content="",
        ))
        assert result.reward < 0

    def test_valid_submit_positive(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        content = json.dumps({
            "problem": "Sequence transduction relies on complex RNNs and CNNs",
            "method": "Transformer with self-attention only",
            "dataset_or_domain": "Machine translation WMT 2014",
            "key_finding": "State-of-the-art BLEU scores with faster training",
            "limitation": "Limited evaluation beyond translation",
            "confidence": "high",
        })
        result = env.step(LitReviewAction(
            action_type=ActionType.SUBMIT,
            content=content,
        ))
        assert result.reward > 0
        assert result.info.get("grading", {}).get("score", 0) > 0

    def test_ground_truth_high_score(self, env: LitReviewEnvironment):
        """Submitting the exact ground truth should score very high."""
        from litreview_env.data_loader import load_task_by_id
        task = load_task_by_id("easy_001")
        env.reset(task_id="easy_001")
        result = env.step(LitReviewAction(
            action_type=ActionType.SUBMIT,
            content=json.dumps(task.ground_truth),
        ))
        score = result.info.get("grading", {}).get("score", 0)
        assert score > 0.8, f"Ground truth score should be high, got {score}"

    def test_step_after_done(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        # Exhaust all steps
        for _ in range(env.state.max_steps):
            env.step(LitReviewAction(action_type=ActionType.NOP))
        assert env.state.done
        with pytest.raises(RuntimeError, match="done"):
            env.step(LitReviewAction(action_type=ActionType.NOP))

    def test_max_steps_termination(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        max_steps = env.state.max_steps
        for i in range(max_steps):
            result = env.step(LitReviewAction(action_type=ActionType.NOP))
        assert result.done

    def test_step_count_increments(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        for i in range(1, 3):
            env.step(LitReviewAction(action_type=ActionType.NOP))
            assert env.state.step_count == i


class TestState:
    """Tests for the state property."""

    def test_initial_state(self, env: LitReviewEnvironment):
        assert env.state.episode_id == ""
        assert env.state.step_count == 0

    def test_state_after_reset(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        assert env.state.episode_id
        assert env.state.task_id == "easy_001"
        assert env.state.difficulty == "easy"
        assert env.state.step_count == 0
        assert not env.state.done

    def test_cumulative_reward(self, env: LitReviewEnvironment):
        env.reset(task_id="easy_001")
        r1 = env.step(LitReviewAction(action_type=ActionType.NOP))
        r2 = env.step(LitReviewAction(action_type=ActionType.NOP))
        expected = r1.reward + r2.reward
        assert abs(env.state.cumulative_reward - expected) < 0.001


class TestListTasks:
    """Tests for task listing."""

    def test_list_all(self, env: LitReviewEnvironment):
        tasks = env.list_tasks()
        assert len(tasks) >= 6  # 3 easy + 2 medium + 1 hard

    def test_list_by_difficulty(self, env: LitReviewEnvironment):
        easy = env.list_tasks("easy")
        medium = env.list_tasks("medium")
        hard = env.list_tasks("hard")
        assert len(easy) >= 3
        assert len(medium) >= 2
        assert len(hard) >= 1
        assert all("easy" in t for t in easy)
        assert all("medium" in t for t in medium)
        assert all("hard" in t for t in hard)
