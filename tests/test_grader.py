"""Tests for the grading system."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litreview_env.graders import grade_easy, grade_medium, grade_hard, grade
from litreview_env.schemas import TaskDifficulty
from litreview_env.data_loader import load_tasks


class TestEasyGrader:
    """Tests for the easy difficulty grader."""

    def _gt(self):
        return {
            "problem": "Training deep networks is hard",
            "method": "Residual learning with skip connections",
            "dataset_or_domain": "ImageNet and CIFAR-10",
            "key_finding": "152-layer networks achieve 3.57% error",
            "limitation": "Focused on classification only",
            "confidence": "high",
        }

    def test_perfect_match(self):
        gt = self._gt()
        result = grade_easy(gt, gt)
        assert result["score"] > 0.9

    def test_missing_fields(self):
        gt = self._gt()
        partial = {"problem": "Training deep networks is hard"}
        result = grade_easy(partial, gt)
        assert result["score"] < 0.5
        assert len(result["breakdown"]["missing_fields"]) > 0

    def test_empty_response(self):
        gt = self._gt()
        result = grade_easy({}, gt)
        assert result["score"] < 0.2

    def test_wrong_confidence(self):
        gt = self._gt()
        response = gt.copy()
        response["confidence"] = "low"
        result = grade_easy(response, gt)
        # Should be close to perfect except for confidence
        assert result["score"] > 0.8

    def test_partial_content(self):
        gt = self._gt()
        response = {
            "problem": "Training very deep neural networks",
            "method": "Skip connections",
            "dataset_or_domain": "ImageNet",
            "key_finding": "Deep residual nets work well",
            "limitation": "Only classification",
            "confidence": "high",
        }
        result = grade_easy(response, gt)
        assert 0.3 < result["score"] < 1.0


class TestMediumGrader:
    """Tests for the medium difficulty grader."""

    def _gt(self):
        return {
            "shared_theme": "Language model pre-training for NLP tasks",
            "methodological_differences": [
                "GPT-2 uses autoregressive LM",
                "RoBERTa uses optimized BERT training",
                "XLNet uses permutation-based LM",
            ],
            "result_differences": [
                "GPT-2 zero-shot benchmarks",
                "RoBERTa GLUE and SQuAD SOTA",
                "XLNet outperforms BERT on 20 tasks",
            ],
            "tradeoffs": [
                "GPT-2 no fine-tuning needed but lower performance",
                "RoBERTa simple but compute-heavy",
                "XLNet richer but more complex",
            ],
            "paper_ids": ["P004", "P005", "P006"],
        }

    def test_perfect_match(self):
        gt = self._gt()
        result = grade_medium(gt, gt)
        assert result["score"] > 0.85

    def test_empty_lists(self):
        gt = self._gt()
        response = {
            "shared_theme": "Pre-training",
            "methodological_differences": [],
            "result_differences": [],
            "tradeoffs": [],
        }
        result = grade_medium(response, gt)
        assert result["score"] < 0.4

    def test_hallucinated_ids(self):
        gt = self._gt()
        response = gt.copy()
        response["methodological_differences"] = [
            "P004 uses autoregressive LM",
            "P099 uses fictional approach",  # Hallucinated ID
        ]
        result = grade_medium(response, gt)
        assert result["breakdown"]["hallucinated_ids"]


class TestHardGrader:
    """Tests for the hard difficulty grader."""

    def test_loads_and_grades(self):
        """Verify hard task can load and grade."""
        tasks = load_tasks(TaskDifficulty.HARD)
        assert len(tasks) >= 1
        task = tasks[0]
        # Grade with ground truth
        result = grade_hard(task.ground_truth, task.ground_truth)
        assert result["score"] > 0.8

    def test_empty_response(self):
        tasks = load_tasks(TaskDifficulty.HARD)
        task = tasks[0]
        result = grade_hard({}, task.ground_truth)
        assert result["score"] < 0.2


class TestGradeDispatcher:
    """Tests for the grade() dispatcher."""

    def test_easy_dispatch(self):
        gt = {"problem": "X", "method": "Y", "dataset_or_domain": "Z",
              "key_finding": "A", "limitation": "B", "confidence": "high"}
        result = grade(TaskDifficulty.EASY, gt, gt)
        assert "score" in result
        assert "breakdown" in result

    def test_medium_dispatch(self):
        gt = {"shared_theme": "X", "methodological_differences": ["a"],
              "result_differences": ["b"], "tradeoffs": ["c"], "paper_ids": ["P001"]}
        result = grade(TaskDifficulty.MEDIUM, gt, gt)
        assert result["score"] > 0

    def test_hard_dispatch(self):
        gt = {"overview": "X", "themes": ["a"], "evidence_points": ["b"],
              "contradictions": ["c"], "limitations": ["d"],
              "research_gaps": ["e"], "future_directions": ["f"],
              "paper_ids": ["P001"]}
        result = grade(TaskDifficulty.HARD, gt, gt)
        assert result["score"] > 0
