"""Tests for the baseline inference script (without actual API calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestBaselineHelpers:
    """Test helper functions from the baseline script."""

    def test_build_system_prompt_easy(self):
        from baseline_inference import build_system_prompt
        prompt = build_system_prompt("easy")
        assert "problem" in prompt
        assert "method" in prompt
        assert "JSON" in prompt

    def test_build_system_prompt_medium(self):
        from baseline_inference import build_system_prompt
        prompt = build_system_prompt("medium")
        assert "shared_theme" in prompt
        assert "methodological_differences" in prompt

    def test_build_system_prompt_hard(self):
        from baseline_inference import build_system_prompt
        prompt = build_system_prompt("hard")
        assert "overview" in prompt
        assert "research_gaps" in prompt

    def test_build_user_prompt(self):
        from baseline_inference import build_user_prompt
        from litreview_env.env import LitReviewEnvironment

        env = LitReviewEnvironment(seed=42)
        obs = env.reset(task_id="easy_001")
        prompt = build_user_prompt(obs)

        assert "P001" in prompt
        assert "Attention" in prompt
        assert "Transformer" in prompt


class TestBaselineRun:
    """Test the task runner with a mock OpenAI client."""

    def test_run_task_mock(self):
        from baseline_inference import run_task
        from litreview_env.env import LitReviewEnvironment

        env = LitReviewEnvironment(seed=42)

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "problem": "Training sequence models is slow",
            "method": "Transformer with attention",
            "dataset_or_domain": "Machine translation",
            "key_finding": "Better BLEU scores",
            "limitation": "Limited scope",
            "confidence": "high",
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        result = run_task(env, mock_client, "mock-model", "easy_001")

        assert result["task_id"] == "easy_001"
        assert result["score"] > 0
        assert result["difficulty"] == "easy"
        assert "elapsed_seconds" in result

    def test_run_task_empty_response(self):
        from baseline_inference import run_task
        from litreview_env.env import LitReviewEnvironment

        env = LitReviewEnvironment(seed=42)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        result = run_task(env, mock_client, "mock-model", "easy_001")
        assert result["score"] == 0 or result["reward"] <= 0
