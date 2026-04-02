"""Baseline inference script for LitReview-Env.

Runs an OpenAI-compatible model through all tasks and reports scores.
Uses multi-turn interaction: submits, reads feedback, and refines.

Usage:
    python scripts/baseline_inference.py
    python scripts/baseline_inference.py --difficulty easy
    python scripts/baseline_inference.py --task-id easy_001

Environment variables:
    OPENAI_API_KEY   - Required. Your API key (OpenAI, Gemini, Groq, etc.).
    OPENAI_MODEL     - Optional. Model to use (default: gpt-4o-mini).
    OPENAI_BASE_URL  - Optional. Custom API base URL.

Using with Groq (free):
    export OPENAI_API_KEY="gsk_..."
    export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
    export OPENAI_MODEL="llama-3.3-70b-versatile"
    python scripts/baseline_inference.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Auto-load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litreview_env.env import LitReviewEnvironment
from litreview_env.schemas import LitReviewAction, ActionType, TaskDifficulty


def get_openai_client():
    """Create an OpenAI client from environment variables."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Improved prompting with few-shot examples
# ---------------------------------------------------------------------------

def build_system_prompt(difficulty: str) -> str:
    """Build a detailed system prompt with format guidance."""
    base = (
        "You are an expert academic researcher performing a literature review. "
        "Your responses must be precise, evidence-grounded, and structured as valid JSON. "
        "Only include information directly supported by the provided abstracts. "
        "Do not hallucinate findings, methods, datasets, or paper references.\n\n"
        "CRITICAL RULES:\n"
        "- Return ONLY a valid JSON object, no markdown fences, no extra text\n"
        "- Be specific and detailed — vague or generic answers score poorly\n"
        "- Use exact terminology from the abstracts when possible\n"
        "- Include specific numbers, metrics, and results mentioned in the abstracts\n"
    )

    if difficulty == "easy":
        base += (
            "\nTASK: Extract a structured record from a single paper abstract.\n"
            "Return a JSON object with EXACTLY these keys:\n"
            "- problem: The specific research problem or challenge addressed (be detailed)\n"
            "- method: The methodology, approach, or technique proposed (include specifics)\n"
            "- dataset_or_domain: The datasets, benchmarks, or application domains used\n"
            "- key_finding: The main result or contribution (include specific numbers/metrics)\n"
            "- limitation: Any stated or implied limitations of the work\n"
            "- confidence: Your confidence level: high, medium, or low\n\n"
            "EXAMPLE OUTPUT:\n"
            '{"problem": "Existing models rely on complex recurrent architectures that are slow to train and hard to parallelize.",'
            '"method": "Transformer architecture based solely on multi-head self-attention mechanisms, eliminating recurrence and convolutions.",'
            '"dataset_or_domain": "Machine translation (WMT 2014 English-German and English-French benchmarks).",'
            '"key_finding": "Achieves 28.4 BLEU on EN-DE and 41.8 BLEU on EN-FR, surpassing all previous single models while training 10x faster.",'
            '"limitation": "Evaluation limited to translation tasks; generalization to other sequence tasks not fully explored.",'
            '"confidence": "high"}'
        )
    elif difficulty == "medium":
        base += (
            "\nTASK: Compare and contrast three related papers.\n"
            "Return a JSON object with EXACTLY these keys:\n"
            "- shared_theme: A sentence describing the common research thread\n"
            "- methodological_differences: A list of strings, one per paper, describing each paper's distinct approach (MUST reference paper_id like P004)\n"
            "- result_differences: A list of strings describing how results differ across papers (reference paper_ids)\n"
            "- tradeoffs: A list of practical tradeoffs between the approaches\n\n"
            "IMPORTANT: Each list item should reference paper IDs (P004, P005, etc.) and be specific."
        )
    elif difficulty == "hard":
        base += (
            "\nTASK: Synthesize a comprehensive structured literature review.\n"
            "Return a JSON object with EXACTLY these keys:\n"
            "- overview: A paragraph summarizing the research landscape across all papers\n"
            "- themes: A list of recurring research themes you identify\n"
            "- evidence_points: A list of specific empirical findings, each referencing paper_id (e.g., 'P010: Loss scales as power-law...')\n"
            "- contradictions: A list of tensions or disagreements between papers\n"
            "- limitations: A list of shared or notable limitations across the papers\n"
            "- research_gaps: A list of underexplored areas NOT covered by the papers\n"
            "- future_directions: A list of promising research directions grounded in the findings\n\n"
            "IMPORTANT: evidence_points MUST reference paper_ids. research_gaps should identify what's MISSING, not restate limitations."
        )

    return base


def build_user_prompt(observation) -> str:
    """Build the user prompt from an observation."""
    parts = [f"**Task Instruction:**\n{observation.instruction}\n"]

    for paper in observation.papers:
        parts.append(
            f"**[{paper.paper_id}] {paper.title}**\n"
            f"Authors: {', '.join(paper.authors)} ({paper.year}) — {paper.venue}\n"
            f"Domain: {paper.domain}\n"
            f"Abstract: {paper.abstract}\n"
        )

    parts.append(
        "Now provide your structured JSON response. "
        "Return ONLY the JSON object — no markdown code fences, no explanation text before or after."
    )

    return "\n".join(parts)


def build_refinement_prompt(feedback: str, previous_response: str) -> str:
    """Build a forceful self-critique prompt to maximize score refinement."""
    return (
        f"Your previous submission received this automated grader feedback:\n{feedback}\n\n"
        "You must improve your response to maximize your score. The grader requires EXACTNESS and HIGH DETAIL "
        "rather than generic summaries.\n\n"
        "BEFORE generating your new JSON, write a BRIEF SELF-CRITIQUE (1-2 paragraphs) analyzing why your previous response "
        "lost points. Ask yourself:\n"
        "1. Did I paraphrase too generically? Did I miss specific numbers, metrics, dataset names, or exact benchmark results?\n"
        "2. Are my lists (if any) comprehensive, or did I skip important details mentioned in the abstracts?\n"
        "3. Did I reference the EXACT paper IDs (e.g. P001) for my claims?\n\n"
        "After writing your critique, output the final, vastly improved JSON object. "
        "(The JSON parser will automatically extract the JSON from your response, so don't worry about extra text)."
    )


# ---------------------------------------------------------------------------
# Model calling with retry
# ---------------------------------------------------------------------------

def call_model(client, model: str, messages: list[dict], temperature: float = 0.0) -> str:
    """Call the model with retry logic for rate limits."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"  Model call failed: {e}")
                return ""
    return ""


# ---------------------------------------------------------------------------
# Multi-turn task runner
# ---------------------------------------------------------------------------

def run_task(
    env: LitReviewEnvironment,
    client: Any,
    model: str,
    task_id: str,
    temperature: float = 0.0,
    max_turns: int = 2,
    delay: float = 2.0,
) -> dict[str, Any]:
    """Run a single task through the model with multi-turn refinement.

    The agent:
    1. Submits an initial response
    2. Reads feedback and weak areas
    3. Re-submits an improved response (if score < 0.9 and steps remain)

    Returns:
        Task result dict with scores and metadata.
    """
    start_time = time.time()

    # Reset environment
    obs = env.reset(task_id=task_id)
    difficulty = obs.difficulty

    print(f"\n{'='*60}")
    print(f"Task: {task_id} (difficulty: {difficulty})")
    print(f"Papers: {len(obs.papers)}, Max steps: {obs.max_steps}")
    print(f"{'='*60}")

    system_prompt = build_system_prompt(difficulty)
    user_prompt = build_user_prompt(obs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    best_score = 0.0
    best_result = None
    last_content = ""

    for turn in range(min(max_turns, obs.max_steps)):
        print(f"  Turn {turn+1}/{max_turns}...")

        # Call model
        content = call_model(client, model, messages, temperature)
        if not content:
            break

        last_content = content

        # Submit to environment
        action = LitReviewAction(action_type=ActionType.SUBMIT, content=content)
        result = env.step(action)

        score = result.info.get("grading", {}).get("score", 0.0)
        print(f"    Score: {score:.4f} | Reward: {result.reward:.4f}")

        if score > best_score:
            best_score = score
            best_result = result

        # If score is good enough or episode is done, stop
        if score >= 0.90 or result.done:
            break

        # Build refinement prompt for next turn
        if turn < max_turns - 1 and not result.done:
            refinement = build_refinement_prompt(result.observation.feedback, content)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": refinement})
            time.sleep(delay)  # Rate limit protection

    # Use best result
    if best_result is None:
        # No successful submission, create a dummy result
        action = LitReviewAction(action_type=ActionType.SUBMIT, content=last_content)
        try:
            best_result = env.step(action)
        except RuntimeError:
            # Episode might be done
            best_result = result if 'result' in dir() else None

    elapsed = time.time() - start_time

    final_score = best_score
    print(f"  Final Score: {final_score:.4f} | Time: {elapsed:.1f}s")

    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "score": final_score,
        "reward": best_result.reward if best_result else 0.0,
        "elapsed_seconds": round(elapsed, 2),
        "model_response_length": len(last_content),
        "grading": best_result.info.get("grading", {}) if best_result else {},
        "model_response": last_content[:2000],
    }


def main():
    parser = argparse.ArgumentParser(description="LitReview-Env Baseline Inference")
    parser.add_argument("--difficulty", type=str, default=None, help="Filter by difficulty: easy, medium, hard")
    parser.add_argument("--task-id", type=str, default=None, help="Run a specific task by ID")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides OPENAI_MODEL env var)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON report path")
    parser.add_argument("--max-turns", type=int, default=2, help="Max turns per task (multi-turn refinement)")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between API calls in seconds")
    args = parser.parse_args()

    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    print(f"LitReview-Env Baseline Inference")
    print(f"Model: {model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max turns: {args.max_turns}")

    client = get_openai_client()
    env = LitReviewEnvironment(seed=args.seed)

    # Determine which tasks to run
    if args.task_id:
        task_ids = [args.task_id]
    elif args.difficulty:
        task_ids = env.list_tasks(args.difficulty)
    else:
        task_ids = env.list_tasks()

    if not task_ids:
        print("No tasks found!")
        sys.exit(1)

    print(f"Running {len(task_ids)} task(s): {task_ids}\n")

    # Run all tasks
    results: list[dict[str, Any]] = []
    for i, tid in enumerate(task_ids):
        try:
            r = run_task(env, client, model, tid, args.temperature, args.max_turns, args.delay)
            results.append(r)
        except Exception as e:
            print(f"  ERROR running {tid}: {e}")
            results.append({"task_id": tid, "error": str(e), "score": 0.0})

        # Delay between tasks
        if i < len(task_ids) - 1:
            time.sleep(args.delay)

    # Aggregate scores
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    by_difficulty: dict[str, list[float]] = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        by_difficulty.setdefault(diff, []).append(r.get("score", 0.0))
        print(f"  {r['task_id']:15s}  {diff:8s}  score={r.get('score', 0.0):.4f}")

    print(f"\nPer-difficulty averages:")
    for diff in ["easy", "medium", "hard"]:
        if diff in by_difficulty:
            avg = sum(by_difficulty[diff]) / len(by_difficulty[diff])
            print(f"  {diff:8s}: {avg:.4f} (n={len(by_difficulty[diff])})")

    overall = sum(s.get("score", 0.0) for s in results) / max(len(results), 1)
    print(f"\nOverall average: {overall:.4f}")

    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "overall_score": round(overall, 4),
        "per_difficulty": {
            d: round(sum(s) / len(s), 4)
            for d, s in by_difficulty.items()
        },
        "task_results": results,
    }

    output_path = args.output or f"baseline_report_{model.replace('/', '_')}_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
