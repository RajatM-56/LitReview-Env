"""
visualizations.py — All rendering functions for the single-page dashboard.
Uses custom HTML cards, Plotly dark-themed charts, and semantic layout.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# ── Plotly Template ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#c9d1d9", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

COLORS = {
    "accent": "#58a6ff",
    "green": "#3fb950",
    "yellow": "#d29922",
    "red": "#f85149",
    "purple": "#bc8cff",
    "cyan": "#39d2c0",
    "grid": "#21262d",
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION HEADERS
# ═════════════════════════════════════════════════════════════════════════════
def _section(icon: str, title: str, subtitle: str = ""):
    sub = f"  ·  {subtitle}" if subtitle else ""
    st.markdown(f"#### {icon} {title}{sub}")


# ═════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ═════════════════════════════════════════════════════════════════════════════
def render_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🔬</div>
        <div class="heading">Welcome to LitReview-Env</div>
        <div class="body">
            Use the sidebar to <b>Reset Environment</b> and load a task,
            or press <b>Mock Suite</b> to instantly run all benchmarks
            with perfect answers and explore the analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════
def render_dashboard(obs, traj, fb, batch):
    """Single-page layout rendered top-to-bottom."""

    # ── TIER 1: Suite Analytics (only if batch data exists) ──────────────
    if batch:
        _render_suite(batch)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── TIER 2: Current Task View ────────────────────────────────────────
    if obs:
        _render_task_header(obs)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        _render_output(traj)

    with col_right:
        if fb:
            _render_grader(fb)
        elif obs:
            _render_papers(obs)


# ═════════════════════════════════════════════════════════════════════════════
# TIER 1 — SUITE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
def _render_suite(batch):
    _section("📊", "Benchmark Suite Results", f"{len(batch)} tasks evaluated")

    df = pd.DataFrame(batch)

    # ── Metrics Row ──────────────────────────────────────────────────────
    avg = df["score"].mean()
    best = df["score"].max()
    worst = df["score"].min()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tasks", len(df))
    m2.metric("Average Score", f"{avg:.2f}")
    m3.metric("Best", f"{best:.2f}")
    m4.metric("Worst", f"{worst:.2f}")

    # ── Bar Chart (full width) ────────────────────────────────────────────
    df_sorted = df.sort_values("score", ascending=True)
    fig = go.Figure()
    colors = [COLORS["green"] if s >= 0.8 else COLORS["yellow"] if s >= 0.4 else COLORS["red"] for s in df_sorted["score"]]
    fig.add_trace(go.Bar(
        y=df_sorted["task_id"],
        x=df_sorted["score"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=[f"{s:.2f}" for s in df_sorted["score"]],
        textposition="auto",
        textfont=dict(size=11, color="#e6edf3"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(180, len(df) * 40),
        xaxis=dict(range=[0, 1.05], gridcolor="#21262d", title="Score"),
        yaxis=dict(gridcolor="#21262d", title=""),
        showlegend=False,
    )
    st.plotly_chart(fig, key="suite_bar", width="stretch")

    # ── Difficulty Breakdown ──────────────────────────────────────────────
    diff_agg = df.groupby("difficulty").agg(
        avg_score=("score", "mean"), count=("score", "count")
    ).reset_index()

    diff_cols = st.columns(len(diff_agg), gap="medium")
    for i, (_, row) in enumerate(diff_agg.iterrows()):
        with diff_cols[i]:
            d = row["difficulty"].upper()
            pct = row["avg_score"]
            st.metric(d, f"{int(row['count'])} tasks", f"avg {pct:.0%}")
            st.progress(min(max(pct, 0.0), 1.0))

    # ── Score Distribution Row ────────────────────────────────────────────
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    bin_colors = [COLORS["red"], COLORS["red"], COLORS["yellow"], COLORS["yellow"], COLORS["green"]]
    df["bucket"] = pd.cut(df["score"], bins=bins, labels=labels, right=False)
    dist = df["bucket"].value_counts().reindex(labels, fill_value=0)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=dist.index.tolist(),
        y=dist.values.tolist(),
        marker=dict(color=bin_colors, cornerradius=4, line=dict(width=0)),
        text=dist.values.tolist(),
        textposition="auto",
        textfont=dict(size=11, color="#e6edf3"),
    ))
    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        height=160,
        xaxis=dict(title="Score Range", gridcolor="#21262d"),
        yaxis=dict(title="Tasks", gridcolor="#21262d", dtick=1),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, key="score_dist", width="stretch")

    # ── Collapsible Detail Table ─────────────────────────────────────────
    with st.expander("View detailed results", icon="📋"):
        st.dataframe(
            df[["task_id", "difficulty", "score", "reward"]],
            hide_index=True,
            width=800,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TIER 2 — TASK HEADER
# ═════════════════════════════════════════════════════════════════════════════
def _render_task_header(obs):
    _section("📄", f"Task: {obs.task_id}", f"{obs.difficulty.upper()} · {len(obs.papers)} paper(s)")

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #161b2280, #1c222b80);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 1.2rem;
        font-size: 0.92rem;
        color: #c9d1d9;
        line-height: 1.65;
        border-left: 4px solid {COLORS['accent']};
    ">
        {obs.instruction}
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAPERS
# ═════════════════════════════════════════════════════════════════════════════
def _render_papers(obs):
    _section("📑", "Source Papers")
    for p in obs.papers:
        authors = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors += f" +{len(p.authors)-3}"
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{p.title}</div>
            <div class="paper-meta">
                {authors} · {p.year} · <span class="venue-badge">{p.venue}</span>
            </div>
            <div class="paper-abstract">{p.abstract[:400]}{"…" if len(p.abstract)>400 else ""}</div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# GRADER VIEW
# ═════════════════════════════════════════════════════════════════════════════
def _render_grader(fb):
    score = fb.get("score", 0.0)
    bd = fb.get("breakdown", {})

    # Color based on score
    if score >= 0.8:
        ring_color, badge_cls = COLORS["green"], "excellent"
    elif score >= 0.4:
        ring_color, badge_cls = COLORS["yellow"], "medium"
    else:
        ring_color, badge_cls = COLORS["red"], "poor"

    _section("⚖️", "Grader Analysis")

    # Left: Score + progress bars, Right: compact radar
    col_metrics, col_radar = st.columns([3, 2], gap="large")

    with col_metrics:
        # Score ring
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:24px; margin-bottom:1rem;">
            <div style="
                width:90px; height:90px; border-radius:50%;
                border:3px solid {ring_color};
                display:flex; flex-direction:column;
                align-items:center; justify-content:center;
                box-shadow: 0 0 20px {ring_color}33;
                flex-shrink:0;
            ">
                <span style="font-size:1.6rem; font-weight:800; color:#e6edf3;">{score:.0%}</span>
                <span style="font-size:0.6rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px;">Score</span>
            </div>
            <div style="flex:1;">
                <div style="font-size:0.8rem; color:#c9d1d9; line-height:1.8;">Final grading result for the current task. Breakdown shown in the progress bars below.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bars
        fmt = bd.get("format_score", 0.0)
        cnt = bd.get("content_score", 0.0)
        gnd = bd.get("grounding_score", 0.0)
        st.progress(min(max(fmt, 0.0), 1.0), text=f"Format — {fmt:.0%}")
        st.progress(min(max(cnt, 0.0), 1.0), text=f"Content — {cnt:.0%}")
        if "grounding_score" in bd:
            st.progress(min(max(gnd, 0.0), 1.0), text=f"Grounding — {gnd:.0%}")

    with col_radar:
        # Compact radar chart
        if "field_scores" in bd and bd["field_scores"]:
            fields = list(bd["field_scores"].keys())
            values = list(bd["field_scores"].values())
            fields_display = [f.replace("_", " ").title() for f in fields]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=fields_display + [fields_display[0]],
                fill="toself",
                fillcolor="rgba(88,166,255,0.12)",
                line=dict(color=COLORS["accent"], width=2),
                marker=dict(size=4, color=COLORS["accent"]),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=220,
                margin=dict(l=30, r=30, t=10, b=10),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262d", tickfont=dict(size=8, color="#8b949e")),
                    angularaxis=dict(gridcolor="#30363d", tickfont=dict(size=9, color="#c9d1d9")),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig, key="radar", width="stretch")

    # Warnings
    if bd.get("missing_fields"):
        st.error(f"**Missing fields:** {', '.join(bd['missing_fields'])}", icon="🚨")
    if bd.get("hallucinated_ids"):
        st.warning(f"**Hallucinated IDs:** {', '.join(bd['hallucinated_ids'])}", icon="⚠️")


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT VIEWER
# ═════════════════════════════════════════════════════════════════════════════
def _render_output(traj):
    _section("📤", "Agent Output")

    if len(traj) <= 1:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#8b949e;">
            <div style="font-size:2rem; margin-bottom:0.5rem; opacity:0.4;">✍️</div>
            <div style="font-size:0.85rem;">No submission yet.<br>Run a task to see the agent's output here.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Find last submit
    submits = [t for t in traj if t.get("type") == "step" and t.get("action", {}).get("action_type") == "submit"]
    if not submits:
        st.info("No submission found in trajectory.")
        return

    content = submits[-1]["action"]["content"]
    try:
        data = json.loads(content)
    except Exception:
        st.code(content, language="json")
        return

    for k, v in data.items():
        label = k.replace("_", " ").title()
        if isinstance(v, list):
            items_html = "".join(f"<li>{item}</li>" for item in v)
            val_html = f"<ul style='margin:0; padding-left:18px;'>{items_html}</ul>"
        else:
            val_html = str(v)

        st.markdown(f"""
        <div class="output-field">
            <div class="field-name">{label}</div>
            <div class="field-value">{val_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LEGACY COMPAT (kept for any stray import but not used in new layout)
# ═════════════════════════════════════════════════════════════════════════════
def render_inputs_tab(obs):
    _render_papers(obs)

def render_grader_tab(fb):
    _render_grader(fb)

def render_output_viewer(traj):
    _render_output(traj)

def render_suite_analytics(batch):
    _render_suite(batch)
