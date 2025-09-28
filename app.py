# cartpole_app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import time

# Try gymnasium first, fall back to gym
try:
    import gymnasium as gym
    GYM_MODERN = True
except Exception:
    import gym
    GYM_MODERN = False

st.set_page_config(page_title="CartPole (Streamlit)", layout="wide")
st.title("CartPole — Run & Visualize in Streamlit 🎯")

st.sidebar.header("Controls")
max_steps = st.sidebar.slider("Max steps per episode", 50, 2000, 500, step=50)
frame_delay_ms = st.sidebar.slider("Frame duration (ms) in GIF", 10, 200, 40)
policy_mode = st.sidebar.selectbox("Policy", ["Heuristic (angle)", "Random", "Manual (slider)"])
num_episodes = st.sidebar.slider("Number of episodes to simulate (GIF)", 1, 10, 1)

st.markdown(
    """
This app runs OpenAI Gym's CartPole-v1 and visualizes the rollouts.
- **Heuristic (angle)**: simple deterministic policy — push right if pole angle > 0 else left.
- **Random**: random actions.
- **Manual (slider)**: control action using a slider (0 = left, 1 = right) while stepping.
"""
)

# Create environment
def make_env():
    if GYM_MODERN:
        # render_mode='rgb_array' available in gymnasium
        try:
            return gym.make("CartPole-v1", render_mode="rgb_array")
        except TypeError:
            # older gymnasium installs might ignore render_mode arg
            return gym.make("CartPole-v1")
    else:
        # older gym: make and rely on env.render(mode='rgb_array')
        return gym.make("CartPole-v1")

@st.cache_resource
def get_env():
    return make_env()

env = get_env()

# helper to get RGB frame from env (works with gymnasium and old gym)
def get_rgb_frame(environment):
    if GYM_MODERN:
        # gymnasium: call render() without mode arg if render_mode set on creation
        try:
            frame = environment.render()
        except TypeError:
            # fallback
            frame = environment.render(mode="rgb_array")
    else:
        frame = environment.render(mode="rgb_array")
    # frame is numpy array HxWx3
    return Image.fromarray(frame.astype(np.uint8))

def run_episode(policy, max_steps):
    frames = []
    # Reset: gymnasium returns obs, info; older gym returns obs
    if GYM_MODERN:
        obs, info = env.reset()
    else:
        obs = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    while (not done) and step < max_steps:
        if policy == "heuristic":
            # obs: [x, x_dot, theta, theta_dot] — use theta (index 2)
            theta = obs[2]
            action = 1 if theta > 0 else 0
        elif policy == "random":
            action = env.action_space.sample()
        else:
            # shouldn't happen here
            action = env.action_space.sample()

        if GYM_MODERN:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            obs, reward, done, info = env.step(action)

        total_reward += reward
        # collect frame
        try:
            img = get_rgb_frame(env)
            frames.append(img)
        except Exception:
            # if rendering fails, we skip frame collection
            pass

        step += 1

    return frames, total_reward, step

# --- GIF generation for requested episodes ---
if st.button("Run & Create GIF(s)"):
    all_gifs = []
    progress = st.progress(0)
    for ep in range(num_episodes):
        st.write(f"Running episode {ep+1}/{num_episodes} with policy = **{policy_mode}**")
        policy_key = "heuristic" if policy_mode.startswith("Heuristic") else ("random" if policy_mode.startswith("Random") else "heuristic")
        frames, tot_reward, steps = run_episode(policy_key, max_steps)
        if len(frames) == 0:
            st.warning("No frames collected (render may not be supported). Try installing gymnasium and pillow.")
            break

        # Save frames into animated GIF in-memory
        buf = io.BytesIO()
        try:
            frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=frame_delay_ms,
                loop=0,
            )
            buf.seek(0)
            st.image(buf.getvalue(), caption=f"Episode {ep+1} — reward {tot_reward:.1f} — steps {steps}")
            all_gifs.append(buf.getvalue())
        except Exception as e:
            st.error(f"Failed to create GIF: {e}")
        progress.progress((ep + 1) / num_episodes)
    progress.empty()
    st.success("Done generating GIF(s).")

st.markdown("---")
st.subheader("Manual step-through (interactive)")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Reset environment"):
        if GYM_MODERN:
            obs, info = env.reset()
        else:
            obs = env.reset()
        st.session_state["obs"] = obs
        st.session_state["done"] = False
        st.session_state["step_count"] = 0
        # show initial frame
        try:
            img0 = get_rgb_frame(env)
            st.image(img0, caption="Initial state")
        except Exception:
            st.write("Rendering not available in this environment.")

with col2:
    if "done" not in st.session_state:
        st.session_state["done"] = True
    if "step_count" not in st.session_state:
        st.session_state["step_count"] = 0

manual_action = st.slider("Manual action (only for Manual policy)", 0, 1, 0, 1)

if st.button("Step once"):
    if st.session_state.get("done", True):
        # reset before step if done
        if GYM_MODERN:
            obs, info = env.reset()
        else:
            obs = env.reset()
        st.session_state["done"] = False
        st.session_state["step_count"] = 0
        st.session_state["obs"] = obs

    obs = st.session_state.get("obs")
    if policy_mode.startswith("Heuristic"):
        theta = obs[2]
        action = 1 if theta > 0 else 0
    elif policy_mode.startswith("Random"):
        action = env.action_space.sample()
    else:
        action = manual_action

    if GYM_MODERN:
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    else:
        obs, reward, done, info = env.step(int(action))

    st.session_state["obs"] = obs
    st.session_state["done"] = done
    st.session_state["step_count"] += 1

    # render and display
    try:
        img = get_rgb_frame(env)
        st.image(img, caption=f"Step {st.session_state['step_count']}, action={int(action)}, reward={reward:.2f}")
    except Exception:
        st.write(f"Step {st.session_state['step_count']}: action={int(action)} — rendering not available.")

# cleanup: when app stops
def close_env():
    try:
        env.close()
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.write("Tip: If rendering fails, try installing `gymnasium` (newer) and `pillow`.")
st.sidebar.write("Run: `pip install gymnasium pillow streamlit numpy`")
