"""FastAPI server for LitReview-Env.

Provides REST and WebSocket endpoints for the OpenEnv interface.
Also serves a minimal web UI for interactive exploration.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from litreview_env.env import LitReviewEnvironment
from litreview_env.schemas import LitReviewAction, ActionType


# ---------------------------------------------------------------------------
# Lifespan + global env
# ---------------------------------------------------------------------------

_env: Optional[LitReviewEnvironment] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    seed = int(os.environ.get("LITREVIEW_SEED", "42"))
    _env = LitReviewEnvironment(seed=seed)
    yield
    _env = None


app = FastAPI(
    title="LitReview-Env",
    description="An OpenEnv environment for academic literature review benchmarking",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_env() -> LitReviewEnvironment:
    if _env is None:
        raise RuntimeError("Environment not initialized")
    return _env


# ---------------------------------------------------------------------------
# Request / Response models (separate from env schemas for API clarity)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    difficulty: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str = "submit"
    content: str = ""


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": "litreview_env"}


@app.get("/info")
async def info():
    """Environment information."""
    env = _get_env()
    return {
        "name": "LitReview-Env",
        "description": "Academic literature review benchmarking environment",
        "task_count": env.task_count,
        "difficulties": ["easy", "medium", "hard"],
        "tasks": {
            "easy": env.list_tasks("easy"),
            "medium": env.list_tasks("medium"),
            "hard": env.list_tasks("hard"),
        },
    }


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment and start a new episode."""
    env = _get_env()
    try:
        task_id = request.task_id if request else None
        difficulty = request.difficulty if request else None
        obs = env.reset(task_id=task_id, difficulty=difficulty)
        return {
            "observation": obs.model_dump(),
            "state": env.state.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(request: Optional[StepRequest] = None):
    """Execute an action in the environment."""
    env = _get_env()
    req = request or StepRequest()
    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type: {req.action_type}. Must be one of: submit, request_hint, nop",
        )

    action = LitReviewAction(action_type=action_type, content=req.content)

    try:
        result = env.step(action)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
            "state": env.state.model_dump(),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current environment state metadata."""
    env = _get_env()
    return {"state": env.state.model_dump()}


@app.get("/tasks")
async def list_tasks(difficulty: Optional[str] = None):
    """List available task IDs."""
    env = _get_env()
    return {"tasks": env.list_tasks(difficulty)}


# ---------------------------------------------------------------------------
# WebSocket endpoint (OpenEnv standard)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time interaction."""
    await ws.accept()
    env = _get_env()

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            method = msg.get("method", "")

            if method == "reset":
                obs = env.reset(
                    task_id=msg.get("task_id"),
                    difficulty=msg.get("difficulty"),
                )
                await ws.send_json({
                    "method": "reset",
                    "observation": obs.model_dump(),
                    "state": env.state.model_dump(),
                })

            elif method == "step":
                action_type = ActionType(msg.get("action_type", "submit"))
                action = LitReviewAction(
                    action_type=action_type,
                    content=msg.get("content", ""),
                )
                result = env.step(action)
                await ws.send_json({
                    "method": "step",
                    "observation": result.observation.model_dump(),
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                    "state": env.state.model_dump(),
                })

            elif method == "state":
                await ws.send_json({
                    "method": "state",
                    "state": env.state.model_dump(),
                })

            else:
                await ws.send_json({"error": f"Unknown method: {method}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Minimal web UI for interacting with the environment."""
    return _WEB_UI_HTML


_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>LitReview-Env — Interactive Demo</title>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --accent-glow: rgba(88,166,255,0.15);
    --green: #3fb950;
    --red: #f85149;
    --orange: #d29922;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  }
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    line-height: 1.6;
    min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #a371f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header .badge {
    background: var(--accent-glow);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1.5rem;
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 1.5rem;
  }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .panel-header {
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .panel-body { padding: 1.25rem; }
  .controls { display: flex; flex-direction: column; gap: 1rem; }
  label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
    display: block;
  }
  select, textarea, input {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    font-family: var(--font);
    font-size: 0.875rem;
    transition: border-color 0.2s;
  }
  select:focus, textarea:focus, input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  textarea { font-family: var(--mono); font-size: 0.8rem; resize: vertical; min-height: 200px; }
  .btn {
    padding: 0.6rem 1.25rem;
    border: none;
    border-radius: 8px;
    font-family: var(--font);
    font-weight: 600;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), #388bfd);
    color: #fff;
  }
  .btn-primary:hover { filter: brightness(1.1); transform: translateY(-1px); }
  .btn-secondary {
    background: var(--border);
    color: var(--text);
  }
  .btn-secondary:hover { background: #484f58; }
  .btn-row { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .state-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }
  .state-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem;
    text-align: center;
  }
  .state-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    font-family: var(--mono);
  }
  .state-card .label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .reward-positive { color: var(--green); }
  .reward-negative { color: var(--red); }
  .output-area {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: var(--mono);
    font-size: 0.8rem;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 600px;
    overflow-y: auto;
    line-height: 1.5;
  }
  .papers-area {
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 0.75rem;
  }
  .paper-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
  }
  .paper-card h4 {
    font-size: 0.85rem;
    color: var(--accent);
    margin-bottom: 0.25rem;
  }
  .paper-card .meta { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.5rem; }
  .paper-card p { font-size: 0.8rem; line-height: 1.5; }
  .done-badge {
    background: var(--green);
    color: #000;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
  }
  .right-col { display: flex; flex-direction: column; gap: 1.5rem; }
  @media (max-width: 900px) {
    .container { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="header">
  <h1>📚 LitReview-Env</h1>
  <span class="badge">OpenEnv v1</span>
  <span class="badge">Interactive Demo</span>
</div>

<div class="container">
  <!-- Left: Controls -->
  <div style="display:flex;flex-direction:column;gap:1.5rem;">
    <div class="panel">
      <div class="panel-header">🎯 Task Controls</div>
      <div class="panel-body controls">
        <div>
          <label>Difficulty</label>
          <select id="difficulty">
            <option value="">Random</option>
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>
        <div>
          <label>Task ID (optional)</label>
          <input id="taskId" placeholder="e.g. easy_001"/>
        </div>
        <button class="btn btn-primary" onclick="doReset()" style="width:100%">🔄 Reset Environment</button>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">📝 Action</div>
      <div class="panel-body controls">
        <div>
          <label>Action Type</label>
          <select id="actionType">
            <option value="submit">Submit</option>
            <option value="request_hint">Request Hint</option>
            <option value="nop">No-Op</option>
          </select>
        </div>
        <div>
          <label>Content (JSON for submit)</label>
          <textarea id="content" placeholder='{"problem":"...","method":"..."}'></textarea>
        </div>
        <button class="btn btn-primary" onclick="doStep()" style="width:100%">▶️ Step</button>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">📊 Episode State</div>
      <div class="panel-body">
        <div class="state-grid" id="stateGrid">
          <div class="state-card"><div class="value" id="stepCount">0</div><div class="label">Step</div></div>
          <div class="state-card"><div class="value" id="maxSteps">-</div><div class="label">Max Steps</div></div>
          <div class="state-card"><div class="value" id="reward">0.00</div><div class="label">Cumulative</div></div>
          <div class="state-card"><div class="value" id="bestScore">0.00</div><div class="label">Best Score</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Right: Output -->
  <div class="right-col">
    <div class="panel">
      <div class="panel-header">📄 Papers <span id="doneStatus"></span></div>
      <div class="panel-body">
        <div id="instruction" style="font-size:0.85rem;margin-bottom:1rem;color:var(--text-muted);"></div>
        <div class="papers-area" id="papers"></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header">💬 Response & Feedback</div>
      <div class="panel-body">
        <div class="output-area" id="output">Click "Reset Environment" to start a new episode.</div>
      </div>
    </div>
  </div>
</div>

<script>
const API = '';

async function doReset() {
  const diff = document.getElementById('difficulty').value;
  const tid = document.getElementById('taskId').value.trim();
  const body = {};
  if (tid) body.task_id = tid;
  else if (diff) body.difficulty = diff;
  try {
    const r = await fetch(API+'/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) { showOutput('Error: ' + (d.detail || JSON.stringify(d))); return; }
    renderObs(d.observation);
    renderState(d.state);
    showOutput('Episode started. Task: ' + d.observation.task_id + ' (' + d.observation.difficulty + ')');
  } catch(e) { showOutput('Error: ' + e.message); }
}

async function doStep() {
  const at = document.getElementById('actionType').value;
  const content = document.getElementById('content').value;
  try {
    const r = await fetch(API+'/step', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action_type:at, content:content})});
    const d = await r.json();
    if (!r.ok) { showOutput('Error: '+(d.detail||JSON.stringify(d))); return; }
    renderObs(d.observation);
    renderState(d.state);
    let txt = 'Reward: ' + d.reward.toFixed(4) + '\\nDone: ' + d.done;
    txt += '\\n\\nFeedback:\\n' + d.observation.feedback;
    if (d.observation.hint) txt += '\\n\\nHint:\\n' + d.observation.hint;
    if (d.info && d.info.grading) txt += '\\n\\nGrading:\\n' + JSON.stringify(d.info.grading, null, 2);
    showOutput(txt);
  } catch(e) { showOutput('Error: ' + e.message); }
}

function renderObs(obs) {
  document.getElementById('instruction').textContent = obs.instruction || '';
  const pc = document.getElementById('papers');
  pc.innerHTML = '';
  (obs.papers||[]).forEach(p => {
    pc.innerHTML += '<div class="paper-card"><h4>['+p.paper_id+'] '+p.title+'</h4><div class="meta">'+p.authors.join(', ')+' ('+p.year+') — '+p.venue+'</div><p>'+p.abstract+'</p></div>';
  });
  document.getElementById('doneStatus').innerHTML = obs.done ? '<span class="done-badge">DONE</span>' : '';
}

function renderState(s) {
  document.getElementById('stepCount').textContent = s.step_count;
  document.getElementById('maxSteps').textContent = s.max_steps;
  const rw = document.getElementById('reward');
  rw.textContent = s.cumulative_reward.toFixed(2);
  rw.className = 'value ' + (s.cumulative_reward >= 0 ? 'reward-positive' : 'reward-negative');
  document.getElementById('bestScore').textContent = s.best_score.toFixed(2);
}

function showOutput(txt) {
  document.getElementById('output').textContent = txt;
}
</script>
</body>
</html>
"""
