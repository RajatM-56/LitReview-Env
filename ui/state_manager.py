import sys
from pathlib import Path

import streamlit as st

# Ensure src in path to import litreview_env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litreview_env.env import LitReviewEnvironment


def init_session_state():
    """Initialize all Streamlit session state variables."""
    if "env" not in st.session_state:
        # Default environment, but we can re-instantiate if needed
        st.session_state.env = LitReviewEnvironment(seed=42)
    
    if "trajectory" not in st.session_state:
        st.session_state.trajectory = []
        
    if "current_obs" not in st.session_state:
        st.session_state.current_obs = None
        
    if "cumulative_reward" not in st.session_state:
        st.session_state.cumulative_reward = 0.0
        
    if "done" not in st.session_state:
        st.session_state.done = True
        
    if "latest_grader_feedback" not in st.session_state:
        st.session_state.latest_grader_feedback = {}
        
    if "task_difficulty" not in st.session_state:
        st.session_state.task_difficulty = "None"
        
    if "task_id" not in st.session_state:
        st.session_state.task_id = "None"
        
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []

def reset_environment(task_id: str = None):
    """Reset the environment state and history."""
    st.session_state.env = LitReviewEnvironment() # fresh instance
    
    # Try fetching obs
    try:
        if task_id:
            obs = st.session_state.env.reset(task_id=task_id)
        else:
            obs = st.session_state.env.reset()
            
        st.session_state.current_obs = obs
        st.session_state.task_id = obs.task_id
        st.session_state.task_difficulty = obs.difficulty
        st.session_state.done = False
        st.session_state.trajectory = [{"step": 0, "type": "reset", "observation": obs}]
        st.session_state.cumulative_reward = 0.0
        st.session_state.latest_grader_feedback = {}
        
    except Exception as e:
        st.error(f"Error resetting environment: {e}")
