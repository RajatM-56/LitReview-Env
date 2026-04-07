import streamlit as st

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
    inject_theme()
    init_session_state()

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

        # ── Environment Setup ────────────────────────────────────────────────
        st.markdown("##### 🧪 Environment")
        available_tasks = st.session_state.env.list_tasks()
        selected_task = st.selectbox("Task", ["random"] + available_tasks, label_visibility="collapsed")

        if st.button("Reset Environment", width="stretch", type="primary", icon="🔄"):
            target_id = None if selected_task == "random" else selected_task
            reset_environment(target_id)

        st.markdown("---")

        # ── Batch Execution ──────────────────────────────────────────────────
        st.markdown("##### 🚀 Run Suite")
        c1, c2 = st.columns(2)
        if c1.button("Mock", width="stretch", icon="🎭"):
            execute_batch_suite(mock=True)
        default_model = os.environ.get("OPENAI_MODEL", "gemini-2.0-flash")
        if c2.button("LLM", width="stretch", icon="🧠"):
            execute_batch_suite(model_name=default_model, mock=False)

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

    # ── Main Canvas ──────────────────────────────────────────────────────────
    obs = st.session_state.current_obs
    traj = st.session_state.trajectory
    fb = st.session_state.latest_grader_feedback
    batch = st.session_state.batch_results

    if not batch and not fb and not obs:
        render_empty_state()
    else:
        render_dashboard(obs, traj, fb, batch)


if __name__ == "__main__":
    main()
