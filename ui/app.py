import streamlit as st
import json
import time
from datetime import datetime

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LitReview-Env · Benchmark Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from state_manager import init_session_state, reset_environment
from env_adapter import execute_step, execute_mock_run, execute_llm_step, execute_batch_suite
from visualizations import render_dashboard, render_empty_state
from theme import inject_theme
import os

def main():
    # Initialize session state first
    init_session_state()
    
    # Initialize theme preference if not exists
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "bright"
    
    # Inject the selected theme
    inject_theme(st.session_state.theme_mode)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1.2rem 0 0.6rem 0;">
            <span style="font-size:2.4rem;">🔬</span>
            <h2 style="margin:0.3rem 0 0 0; font-family:'Inter',sans-serif; font-weight:700; letter-spacing:-0.5px;">
                LitReview-Env
            </h2>
            <p style="margin:0; color:#888; font-size:0.82rem; letter-spacing:0.5px;">BENCHMARK STUDIO</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Theme Toggle ────────────────────────────────────────────────────────
        st.markdown("##### 🎨 Theme")
        theme_option = st.radio(
            "Select Theme",
            options=["bright", "dark"],
            format_func=lambda x: "☀️ Bright" if x == "bright" else "🌙 Dark",
            label_visibility="collapsed",
            horizontal=True,
            key="theme_radio"
        )
        if theme_option != st.session_state.theme_mode:
            st.session_state.theme_mode = theme_option
            st.rerun()

        st.markdown("---")

        # ── Environment Setup ────────────────────────────────────────────────
        st.markdown("##### 🧪 Environment")
        available_tasks = st.session_state.env.list_tasks()
        
        selected_task = st.selectbox(
            "Task",
            options=["random"] + available_tasks,
            label_visibility="collapsed",
            help="Select a task to load"
        )

        if st.button("Reset Environment", width="stretch", type="primary", icon="🔄"):
            target_id = None if selected_task == "random" else selected_task
            reset_environment(target_id)
            st.session_state.execution_time = None

        st.markdown("---")

        # ── Batch Execution with Timer ───────────────────────────────────────
        st.markdown("##### 🚀 Run Suite")
        c1, c2 = st.columns(2)
        if c1.button("Mock", width="stretch", icon="🎭"):
            st.session_state.start_time = time.time()
            execute_batch_suite(mock=True)
            st.session_state.execution_time = time.time() - st.session_state.start_time
        default_model = os.environ.get("OPENAI_MODEL", "gemini-2.0-flash")
        if c2.button("LLM", width="stretch", icon="🧠"):
            st.session_state.start_time = time.time()
            execute_batch_suite(model_name=default_model, mock=False)
            st.session_state.execution_time = time.time() - st.session_state.start_time
        
        # Show execution time if available
        if hasattr(st.session_state, 'execution_time') and st.session_state.execution_time:
            st.markdown(f"""
            <div style="text-align:center; padding:8px 0; background:#0d111760; border-radius:8px; margin-top:8px;">
                <span style="font-size:0.75rem; color:#58a6ff;">⏱️ Completed in {st.session_state.execution_time:.2f}s</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Single-Step Controls ─────────────────────────────────────────────
        with st.expander("Advanced Controls", icon="⚙️"):
            model_name = st.text_input("Model", value=default_model)
            if st.button("Single LLM Step", width="stretch"):
                execute_llm_step(model_name)
            if st.button("Mock Perfect Answer", width="stretch"):
                execute_mock_run()
            st.markdown("---")
            action_type = st.selectbox("Action", ["submit", "request_hint", "nop"])
            manual_content = st.text_area("Payload", height=80)
            if st.button("Send Manual Action", width="stretch"):
                execute_step(action_type, manual_content)
            
            st.markdown("---")
            st.markdown("##### 📥 Export & Clear")
            
            # Download Results
            if st.session_state.latest_grader_feedback or st.session_state.batch_results:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "task": selected_task,
                    "feedback": st.session_state.latest_grader_feedback,
                    "trajectory": [str(t) for t in st.session_state.trajectory],
                    "batch_results": st.session_state.batch_results
                }
                st.download_button(
                    label="📥 Download Results",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    width="stretch"
                )
            
            # Clear History
            if st.button("🗑️ Clear History", width="stretch"):
                st.session_state.trajectory = []
                st.session_state.latest_grader_feedback = None
                st.rerun()

        # ── Status Pill ──────────────────────────────────────────────────────
        st.markdown("---")
        status_color = "#66BB6A" if not st.session_state.done else "#78909C"
        status_text = "ACTIVE" if not st.session_state.done else "IDLE"
        st.markdown(f"""
        <div style="text-align:center; padding:0.6rem 0;">
            <span style="
                background:{status_color}20;
                color:{status_color};
                padding:4px 18px;
                border-radius:20px;
                font-size:0.75rem;
                font-weight:700;
                letter-spacing:1.5px;
                border: 1px solid {status_color}40;
            ">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Main Canvas Header with Help Button ──────────────────────────────────
    header_col1, header_col2 = st.columns([1, 0.05])
    
    with header_col2:
        if st.button("ℹ️", key="help_btn", help="Project Information"):
            st.session_state.show_help = not st.session_state.get("show_help", False)
    
    # ── Help Modal Popup ─────────────────────────────────────────────────────
    if st.session_state.get("show_help", False):
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            with st.container(border=True):
                st.markdown("## 🔬 LitReview-Env")
                
                st.markdown("**Agentic Benchmark for Literature Review**")
                st.markdown("An intelligent system for evaluating AI models on academic literature review tasks across three difficulty levels.")
                
                st.markdown("### 📋 Task Overview")
                st.markdown("""
- **Easy:** Single-paper extraction (6 fields: problem, method, dataset, key_finding, limitation, confidence)
- **Medium:** Multi-paper comparison (4 fields: shared_theme, methodological_differences, result_differences, tradeoffs)
- **Hard:** Literature review synthesis (7 fields: overview, themes, evidence_points, contradictions, limitations, research_gaps, future_directions)
                """)
                
                st.markdown("### 🎯 Grading Metrics")
                st.markdown("""
- **Format Score (33%):** Field presence & JSON validity
- **Content Score (33%):** Text similarity & keyword overlap with ground truth
- **Grounding Score (34%):** Evidence hallucination detection & citation accuracy
                """)
                
                st.markdown("### ⚙️ Actions")
                st.markdown("""
- **submit:** Submit your JSON response
- **request_hint:** Get guidance on next steps
- **nop:** No operation (pass)
                """)
                
                st.markdown("### 📊 Dashboard Features")
                st.markdown("""
- **Suite Results:** Batch-run aggregate statistics
- **Grader Analysis:** Per-field score breakdown & radar chart
- **Agent Output:** Submission history & trajectory
                """)
                
                st.markdown("### 🚀 Quick Start")
                st.markdown("""
1. Select a task from dropdown in sidebar
2. Click "Reset Environment" to load the task
3. Run "Mock Suite" for instant results with perfect answers
4. Or use "Advanced Controls" for step-by-step manual interaction
                """)
                
                if st.button("Close", key="close_modal_btn", width="stretch"):
                    st.session_state.show_help = False
                    st.rerun()

    # ── Main Canvas ──────────────────────────────────────────────────────────
    obs = st.session_state.current_obs
    traj = st.session_state.trajectory
    fb = st.session_state.latest_grader_feedback
    batch = st.session_state.batch_results

    if not batch and not fb and not obs:
        render_empty_state()
    else:
        render_dashboard(obs, traj, fb, batch)
    
    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; padding:1.5rem 0; color:#8b949e; font-size:0.95rem;'>"
        "Made with 💖 by Stepup"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
