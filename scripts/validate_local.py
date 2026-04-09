"""Local validation script for LitReview-Env.

Validates that the environment works correctly without needing Docker
or an OpenAI API key. Runs through all tasks with mock responses.

Usage:
    python scripts/validate_local.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litreview_env.env import LitReviewEnvironment
from litreview_env.schemas import LitReviewAction, ActionType


def validate_task(env: LitReviewEnvironment, task_id: str) -> dict:
    """Validate a single task by running it through the environment."""
    print(f"\n--- Validating {task_id} ---")

    # Test reset
    obs = env.reset(task_id=task_id)
    assert obs.task_id == task_id, f"Expected task_id={task_id}, got {obs.task_id}"
    assert len(obs.papers) > 0, "No papers in observation"
    assert obs.instruction, "No instruction in observation"
    assert not obs.done, "Episode should not be done after reset"
    print(f"  ✓ Reset: {len(obs.papers)} papers, {obs.max_steps} max steps")

    # Test state
    state = env.state
    assert state.task_id == task_id
    assert state.step_count == 0
    assert not state.done
    print(f"  ✓ State: episode_id={state.episode_id[:8]}...")

    # Test NOP action
    nop_result = env.step(LitReviewAction(action_type=ActionType.NOP))
    assert 0.0 < nop_result.reward < 0.2, f"NOP should have low positive reward, got {nop_result.reward}"
    print(f"  ✓ NOP: reward={nop_result.reward}")

    # Test hint action
    hint_result = env.step(LitReviewAction(action_type=ActionType.REQUEST_HINT))
    assert 0.0 < hint_result.reward < 0.2, f"Hint should have low positive reward (cost), got {hint_result.reward}"
    print(f"  ✓ Hint: reward={hint_result.reward}")

    # Test invalid JSON submit
    invalid_result = env.step(LitReviewAction(
        action_type=ActionType.SUBMIT,
        content="this is not json",
    ))
    assert 0.0 < invalid_result.reward < 0.2, f"Invalid JSON should have low positive reward, got {invalid_result.reward}"
    print(f"  ✓ Invalid JSON: reward={invalid_result.reward}")

    # Test valid submission with ground truth (perfect score baseline)
    env.reset(task_id=task_id)
    from litreview_env.data_loader import load_task_by_id
    task = load_task_by_id(task_id)
    gt_json = json.dumps(task.ground_truth)
    gt_result = env.step(LitReviewAction(
        action_type=ActionType.SUBMIT,
        content=gt_json,
    ))
    assert gt_result.reward > 0, "Ground truth submission should have positive reward"
    score = gt_result.info.get("grading", {}).get("score", 0)
    print(f"  ✓ Ground truth submit: score={score:.4f}, reward={gt_result.reward:.4f}")

    # Test empty submission
    env.reset(task_id=task_id)
    empty_result = env.step(LitReviewAction(
        action_type=ActionType.SUBMIT,
        content="",
    ))
    assert 0.0 < empty_result.reward < 0.2, f"Empty submission should have low positive reward, got {empty_result.reward}"
    print(f"  ✓ Empty submit: reward={empty_result.reward}")

    return {
        "task_id": task_id,
        "papers": len(obs.papers),
        "ground_truth_score": score,
        "ground_truth_reward": gt_result.reward,
        "status": "PASS",
    }


def validate_step_limits(env: LitReviewEnvironment):
    """Validate that episodes terminate at max_steps."""
    print("\n--- Validating step limits ---")
    env.reset(task_id="easy_001")
    max_steps = env.state.max_steps

    for i in range(max_steps):
        result = env.step(LitReviewAction(action_type=ActionType.NOP))
        if result.done:
            assert i == max_steps - 1, f"Episode ended early at step {i+1}/{max_steps}"
            break

    assert env.state.done, "Episode should be done after max steps"
    print(f"  ✓ Episode correctly terminates after {max_steps} steps")

    # Verify stepping after done raises error
    try:
        env.step(LitReviewAction(action_type=ActionType.NOP))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("  ✓ Correctly raises error when stepping after done")


def validate_reset_without_init():
    """Validate that stepping without reset raises error."""
    print("\n--- Validating reset requirement ---")
    env = LitReviewEnvironment(seed=42)
    try:
        env.step(LitReviewAction(action_type=ActionType.NOP))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("  ✓ Correctly raises error when stepping without reset")


def main():
    print("=" * 60)
    print("LitReview-Env Local Validation")
    print("=" * 60)

    env = LitReviewEnvironment(seed=42)

    # Validate all tasks
    all_tasks = env.list_tasks()
    print(f"\nFound {len(all_tasks)} tasks: {all_tasks}")

    results = []
    for task_id in all_tasks:
        try:
            r = validate_task(env, task_id)
            results.append(r)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({"task_id": task_id, "status": "FAIL", "error": str(e)})

    # Validate edge cases
    validate_step_limits(env)
    validate_reset_without_init()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")

    for r in results:
        status = "✓" if r.get("status") == "PASS" else "✗"
        gt_score = r.get("ground_truth_score", "N/A")
        if isinstance(gt_score, float):
            gt_score = f"{gt_score:.4f}"
        print(f"  {status} {r['task_id']:15s} GT-score={gt_score}")

    print(f"\n{passed}/{len(results)} tasks passed, {failed} failed")

    if failed > 0:
        print("\n⚠ Some validations failed!")
        sys.exit(1)
    else:
        print("\n✅ All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
