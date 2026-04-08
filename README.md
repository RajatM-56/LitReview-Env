# LitReview-Env

> An OpenEnv environment for academic literature review benchmarking.

**LitReview-Env** is an agentic benchmark that evaluates AI models on their ability to read paper abstracts, extract structured findings, compare multiple studies, and synthesize coherent literature reviews with identified research gaps. It provides deterministic, programmatic grading across three difficulty levels.

---

## Motivation

Literature review is one of the most intellectually demanding tasks in research — it requires careful reading, structured extraction, cross-paper comparison, contradiction detection, and synthesis of future directions. Unlike toy benchmarks, LitReview-Env tests capabilities that are directly useful for real research assistance:

- **Information extraction** from dense academic text
- **Structured reasoning** across multiple sources
- **Evidence grounding** — not hallucinating claims
- **Gap analysis** — identifying what the literature *doesn't* cover

---

## Environment Overview

| Property | Value |
|---|---|
| **Interface** | OpenEnv (`reset()`, `step()`, `state()`) |
| **Action space** | `submit` (JSON response), `request_hint`, `nop` |
| **Observation space** | Task instruction + paper abstracts + feedback |
| **Reward range** | `[-1.0, 1.0]` continuous |
| **Tasks** | 6 total (3 easy, 2 medium, 1 hard) |
| **Grading** | Deterministic, no LLM judge needed |
| **Offline** | Fully self-contained, no internet required |

---

## Task Descriptions

### Task 1 — Easy: Single-Paper Extraction
- **Input:** 1 abstract
- **Goal:** Extract a structured record with: `problem`, `method`, `dataset_or_domain`, `key_finding`, `limitation`, `confidence`
- **Output:** JSON object with exactly these 6 fields
- **Grading:** Per-field text similarity + keyword overlap against ground truth
- **Expected difficulty:** Straightforward extraction for capable models

### Task 2 — Medium: Multi-Paper Comparison
- **Input:** 3 abstracts on a shared topic
- **Goal:** Compare and contrast the papers
- **Output:** JSON with: `shared_theme`, `methodological_differences`, `result_differences`, `tradeoffs`
- **Grading:** Theme similarity + list coverage + paper ID grounding
- **Expected difficulty:** Requires cross-paper reasoning and structured comparison

### Task 3 — Hard: Literature Review Synthesis
- **Input:** 8 abstracts on a broad research area
- **Goal:** Synthesize a structured literature review with gap analysis
- **Output:** JSON with: `overview`, `themes`, `evidence_points`, `contradictions`, `limitations`, `research_gaps`, `future_directions`
- **Grading:** Section coverage + evidence grounding + hallucination detection
- **Expected difficulty:** Challenging — requires understanding relationships across many papers

---

## Action Space

| Action Type | Description | Reward Effect |
|---|---|---|
| `submit` | Submit a JSON-formatted structured response | Score-based reward (0.0 – 1.0) |
| `request_hint` | Request a progressive hint | -0.05 per hint |
| `nop` | No operation | -0.10 (increasing penalty) |

---

## Observation Space

Each observation includes:

```python
{
    "task_id": "easy_001",
    "difficulty": "easy",
    "instruction": "Read the following paper abstract...",
    "papers": [
        {
            "paper_id": "P001",
            "title": "...",
            "authors": ["..."],
            "year": 2017,
            "venue": "NeurIPS",
            "abstract": "...",
            "domain": "NLP"
        }
    ],
    "feedback": "Score: 0.85/1.00 | Format: 100% | ...",
    "step_number": 1,
    "max_steps": 3,
    "done": false
}
```

---

## How the Reward Works

Rewards are continuous in `[-1.0, 1.0]`:

- **Submission reward** = graded score (0–1) based on content quality
- **Format bonus** (0.15 weight): All required fields present
- **Content score** (0.75–0.85 weight): Text similarity + keyword overlap + list coverage against ground truth
- **Grounding score** (0.10–0.15 weight): Paper ID references match the corpus
- **First valid submission bonus**: +0.05
- **Improvement bonus**: +0.02 when score exceeds previous best
- **Penalties**: Invalid JSON (-0.15), empty submit (-0.10), NOP (-0.10), repeated NOP (-0.20), hallucinated IDs (-0.05 each), hint requests (-0.05)

---

## How the Grader Works

All grading is **deterministic and programmatic** — no LLM judge is used.

1. **JSON Parsing**: Attempts to extract valid JSON from the response (handles markdown fences, trailing commas)
2. **Format Check**: Verifies all required fields are present
3. **Content Scoring**:
   - *String fields*: Blended SequenceMatcher similarity (60%) + keyword Jaccard overlap (40%)
   - *List fields*: Coverage scoring — each ground truth item is matched to the best predicted item using similarity threshold
   - *Confidence field* (easy): Exact match
4. **Grounding Check**: Verifies paper ID references (e.g., P001) exist in the corpus
5. **Weighted Combination**: Field scores are weighted by importance and combined with format and grounding scores

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd LitReview-Env

# Install dependencies
pip install -e ".[dev]"

# Validate the environment works
python scripts/validate_local.py

# Run tests
pytest tests/ -v
```

### Docker

```bash
# Build the image
docker build -t litreview-env .

# Run the container
docker run -p 8000:8000 litreview-env

# Access the web UI
open http://localhost:8000
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for baseline inference |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for baseline inference |
| `OPENAI_BASE_URL` | — | Custom API endpoint |
| `LITREVIEW_SEED` | `42` | Random seed for task selection |
| `LITREVIEW_DATA_DIR` | `./data` | Path to task data directory |

---

## Usage

### Python API

```python
from litreview_env import LitReviewEnvironment, LitReviewAction, ActionType

env = LitReviewEnvironment(seed=42)

# Start an episode
obs = env.reset(task_id="easy_001")
print(obs.instruction)
print(obs.papers[0].abstract)

# Submit a response
import json
action = LitReviewAction(
    action_type=ActionType.SUBMIT,
    content=json.dumps({
        "problem": "Sequence models are slow to train",
        "method": "Transformer with self-attention",
        "dataset_or_domain": "Machine translation",
        "key_finding": "State-of-the-art BLEU with faster training",
        "limitation": "Limited evaluation scope",
        "confidence": "high"
    })
)
result = env.step(action)
print(f"Score: {result.info['grading']['score']}")
print(f"Reward: {result.reward}")
print(f"Done: {result.done}")
```

### REST API

```bash
# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_001"}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit", "content": "{\"problem\": \"...\"}"}'

# State
curl http://localhost:8000/state
```

### Streamlit UI

```bash
python -m streamlit run ui\app.py
```

### Baseline Inference

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run all tasks
python scripts/baseline_inference.py

# Run specific difficulty
python scripts/baseline_inference.py --difficulty easy

# Run specific task
python scripts/baseline_inference.py --task-id easy_001

# Use a different model
python scripts/baseline_inference.py --model gpt-4o
```

---

## Baseline Scores

Expected baseline scores with `gpt-4o-mini` (temperature=0):

| Difficulty | Tasks | Expected Score Range |
|---|---|---|
| Easy | 3 | 0.55 – 0.75 |
| Medium | 2 | 0.40 – 0.60 |
| Hard | 1 | 0.30 – 0.50 |
| **Overall** | **6** | **0.40 – 0.60** |

> Scores depend on model capability. The grading is based on text similarity to expert-written ground truth, so perfect scores require closely matching the reference annotations' content and terminology.

---

## Repository Structure

```
LitReview-Env/
├── README.md                    # This file
├── Dockerfile                   # Container image definition
├── openenv.yaml                 # OpenEnv environment manifest
├── pyproject.toml               # Python project config
├── requirements.txt             # Pip dependencies
├── .dockerignore
├── .gitignore
├── server/
│   ├── __init__.py
│   └── app.py                   # FastAPI server + web UI
├── scripts/
│   ├── baseline_inference.py    # Baseline model evaluation
│   └── validate_local.py        # Local validation (no API key)
├── src/litreview_env/
│   ├── __init__.py              # Package exports
│   ├── env.py                   # Core environment (reset/step/state)
│   ├── schemas.py               # Pydantic v2 data models
│   ├── graders.py               # Deterministic graders per difficulty
│   ├── rewards.py               # Reward shaping logic
│   ├── tasks.py                 # Task management
│   ├── data_loader.py           # JSON data file loader
│   └── utils.py                 # Text similarity utilities
├── data/
│   ├── easy/tasks.json          # 3 easy tasks
│   ├── medium/tasks.json        # 2 medium tasks
│   └── hard/tasks.json          # 1 hard task
└── tests/
    ├── test_env.py              # Environment tests
    ├── test_grader.py           # Grader tests
    └── test_baseline.py         # Baseline script tests
```

---

## Assumptions and Limitations

- **Grading is text-similarity-based**: The grader uses SequenceMatcher and keyword overlap rather than semantic embeddings. This means paraphrased but correct answers may score lower than responses that closely match the reference wording.
- **No LLM judge**: This is intentional — grading is fully deterministic and inspectable. The tradeoff is that creative but correct reformulations receive lower scores.
- **Corpus is curated**: The paper abstracts are based on well-known ML/NLP papers. The ground truth annotations are expert-written reference answers.
- **Single-turn per submission**: Each `submit` action is graded independently. The environment allows multiple submissions per episode (up to `max_steps`) but doesn't build on partial answers.
- **Offline operation**: The environment is fully self-contained with bundled data. No internet access is needed at runtime.

---

## License

MIT
