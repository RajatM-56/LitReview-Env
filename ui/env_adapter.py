import sys
import time
from pathlib import Path
import streamlit as st
import json

# Ensure src in path to import litreview_env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litreview_env.schemas import LitReviewAction, ActionType
from litreview_env.utils import safe_parse_json

# Import baseline helpers to run the model natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
try:
    from baseline_inference import get_openai_client, build_system_prompt, build_user_prompt, build_refinement_prompt, call_model
except ImportError:
    # Handle if they haven't run the install or if pathing issues occur
    pass


def execute_step(action_type: str, content: str):
    """Execute a single step in the environment."""
    if st.session_state.done:
        st.warning("Episode is finished. Please reset the environment.")
        return

    action = LitReviewAction(
        action_type=ActionType(action_type),
        content=content
    )
    
    with st.spinner("Environment is processing your action..."):
        time.sleep(0.5) # Slight UX delay
        result = st.session_state.env.step(action)
        
        # Update session state
        st.session_state.current_obs = result.observation
        st.session_state.done = result.done
        st.session_state.cumulative_reward += result.reward
        
        # Parse grader info
        grading_info = result.info.get("grading", {})
        if grading_info:
            st.session_state.latest_grader_feedback = grading_info
        
        # Record trajectory
        step_number = len(st.session_state.trajectory)
        st.session_state.trajectory.append({
            "step": step_number,
            "type": "step",
            "action": action.dict(),
            "reward": result.reward,
            "observation": result.observation,
            "info": result.info
        })
        
    st.success(f"Step executed! Reward: {result.reward:.4f}")


def execute_mock_run():
    """Execute a mock run with perfect ground-truth data to demo the UI."""
    if st.session_state.done:
        st.warning("Reset environment first.")
        return
        
    diff = st.session_state.env.state.difficulty
    task_id = st.session_state.env.state.task_id
    
    # We grab the true ground truth from the env internals to fake a perfect submission
    task_inst = st.session_state.env._task_manager.get_task(task_id)
    ground_truth_json = json.dumps(task_inst.ground_truth, indent=2)
    
    execute_step("submit", ground_truth_json)


def execute_llm_step(model_name: str):
    """Call the LLM for a single turn."""
    if st.session_state.done:
        return
        
    obs = st.session_state.current_obs
    difficulty = obs.difficulty
    
    client = get_openai_client()
    sys_prompt = build_system_prompt(difficulty)
    usr_prompt = build_user_prompt(obs)
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]
    
    # If this is a refinement turn (step > 0 and we have a previous submission that wasn't NOP/Hint)
    last_sub = ""
    for t in reversed(st.session_state.trajectory):
        if "action" in t and t["action"]["action_type"] == "submit":
            last_sub = t["action"]["content"]
            break
            
    if last_sub and not st.session_state.done:
        refinement_prompt = build_refinement_prompt(obs.feedback, last_sub)
        messages.append({"role": "assistant", "content": last_sub})
        messages.append({"role": "user", "content": refinement_prompt})
        
    with st.spinner(f"Calling {model_name}..."):
        content = call_model(client, model_name, messages, temperature=0.0)
        
    if not content:
        st.error("Model failed to return content or rate limited.")
        return
        
    execute_step("submit", content)


def execute_batch_suite(model_name: str = None, mock: bool = False):
    """Execute a full sequence of all available tasks in the environment automatically."""
    from state_manager import reset_environment
    
    tasks = st.session_state.env.list_tasks()
    st.session_state.batch_results = []
    
    progress_text = "Starting Benchmark Suite..."
    my_bar = st.progress(0, text=progress_text)
    
    for idx, tid in enumerate(tasks):
        my_bar.progress(idx / len(tasks), text=f"Evaluating task: {tid} ({idx+1}/{len(tasks)})")
        
        # Reset the environment for the specific task id
        reset_environment(tid)
        
        score = 0.0
        if mock:
            # Fake a perfect ground truth submission
            t_inst = st.session_state.env._task_manager.get_task(tid)
            gt_str = json.dumps(t_inst.ground_truth)
            execute_step("submit", gt_str)
            
            fb = st.session_state.latest_grader_feedback
            score = fb.get("score", 0.0) if fb else 0.0
            
            # Artificial sleep so the UI can be viewed briefly
            time.sleep(0.5)
        else:
            # LLM sequence evaluation - Minimized for Gemini rate limits
            for turn in range(1): # Reduced strict LLM calls to 1 turn per task
                time.sleep(4.1) # Hard pacing to ensure we stay under the 15 Requests-Per-Minute API limit natively
                execute_llm_step(model_name)
                
                # Check early exit conditions
                fb = st.session_state.latest_grader_feedback
                score = fb.get("score", 0.0) if fb else 0.0
                if st.session_state.done or score >= 0.90:
                    break
                    
        # Append result for the Analytics Tab
        st.session_state.batch_results.append({
            "task_id": tid,
            "difficulty": st.session_state.task_difficulty,
            "score": score,
            "reward": st.session_state.cumulative_reward
        })
        
    my_bar.progress(1.0, text="Suite Complete! View the Analytics tab.")
