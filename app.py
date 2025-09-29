import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import io
import base64
from PIL import Image
import cv2

# Set page config
st.set_page_config(
    page_title="RL Agent Training Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DQN Network Architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.gamma = gamma
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
        
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = {
        'episodes': [],
        'rewards': [],
        'losses': [],
        'epsilon': [],
        'avg_rewards': []
    }
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'agent_frames' not in st.session_state:
    st.session_state.agent_frames = []

# Main title
st.title("🤖 Reinforcement Learning Agent Training Dashboard")
st.markdown("### Multi-Environment Deep Q-Network (DQN) & Continuous Control")

# Environment Selection
st.sidebar.header("🎮 Environment Selection")
env_names = list(ENVIRONMENTS.keys())
env_display_names = [f"{ENVIRONMENTS[env]['name']} ({ENVIRONMENTS[env]['type']})" for env in env_names]

selected_env_index = st.sidebar.selectbox(
    "Choose Environment:",
    range(len(env_names)),
    format_func=lambda x: env_display_names[x],
    index=env_names.index(st.session_state.selected_env) if st.session_state.selected_env in env_names else 0
)

st.session_state.selected_env = env_names[selected_env_index]
current_env_config = ENVIRONMENTS[st.session_state.selected_env]

# Display environment info
with st.sidebar.expander("ℹ️ Environment Info", expanded=False):
    st.write(f"**Name:** {current_env_config['name']}")
    st.write(f"**Type:** {current_env_config['type'].title()}")
    st.write(f"**Description:** {current_env_config['description']}")
    st.write(f"**Max Steps:** {current_env_config['max_steps']}")
    st.write(f"**Success Threshold:** {current_env_config['success_threshold']}")

st.sidebar.markdown("---")

# Sidebar for hyperparameters
st.sidebar.header("🔧 Hyperparameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
gamma = st.sidebar.slider("Discount Factor (γ)", 0.9, 0.999, 0.95)
epsilon = st.sidebar.slider("Initial Epsilon (ε)", 0.5, 1.0, 1.0)
epsilon_min = st.sidebar.slider("Minimum Epsilon", 0.001, 0.1, 0.01, format="%.3f")
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.99, 0.999, 0.995, format="%.3f")
hidden_size = st.sidebar.selectbox("Hidden Layer Size", [64, 128, 256], index=1)

# Training parameters
st.sidebar.header("🎯 Training Parameters")
num_episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 500)
target_update_freq = st.sidebar.slider("Target Network Update Frequency", 5, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# Initialize agent button
if st.sidebar.button("🚀 Initialize Agent", type="primary"):
    try:
        env = gym.make(st.session_state.selected_env)
        
        # Get state and action sizes based on environment type
        if current_env_config['type'] == 'discrete':
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
        else:
            # For continuous environments
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
        
        st.session_state.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay
        )
        
        st.session_state.training_data = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilon': [],
            'avg_rewards': []
        }
        
        st.session_state.training_completed = False
        
        st.sidebar.success(f"Agent initialized for {current_env_config['name']}!")
        env.close()
    except Exception as e:
        st.sidebar.error(f"Error initializing environment: {str(e)}")
        st.sidebar.info("Note: Some environments require additional packages. Check requirements.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Training Progress")
    
    # Create placeholder for real-time plots
    plot_placeholder = st.empty()
    
    # Training controls
    col_train1, col_train2, col_train3 = st.columns(3)
    
    with col_train1:
        start_training = st.button("▶️ Start Training", disabled=st.session_state.agent is None)
    
    with col_train2:
        stop_training = st.button("⏹️ Stop Training")
    
    with col_train3:
        reset_training = st.button("🔄 Reset Training Data")

with col2:
    st.header("📈 Statistics")
    stats_placeholder = st.empty()

# Reset training data
if reset_training:
    st.session_state.training_data = {
        'episodes': [],
        'rewards': [],
        'losses': [],
        'epsilon': [],
        'avg_rewards': []
    }
    st.success("Training data reset!")

# Training function
def train_agent():
    if st.session_state.agent is None:
        st.error("Please initialize the agent first!")
        return
    
    env_name = st.session_state.selected_env
    env_config = ENVIRONMENTS[env_name]
    
    try:
        env = gym.make(env_name)
    except Exception as e:
        st.error(f"Error creating environment: {str(e)}")
        st.info("Some environments may require additional packages. Check the requirements.")
        return
    
    agent = st.session_state.agent
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for episode in range(num_episodes):
        if stop_training:
            break
        
        try:
            state, _ = env.reset()
            total_reward = 0
            losses = []
            
            for step in range(env_config['max_steps']):
                action = agent.act(state)
                
                # Handle continuous vs discrete action spaces
                if env_config['type'] == 'continuous':
                    # Convert discrete action to continuous
                    action_continuous = np.array([action / (agent.action_size - 1) * 2 - 1])
                    if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
                        action_continuous = np.repeat(action_continuous, env.action_space.shape[0])
                    next_state, reward, done, truncated, _ = env.step(action_continuous)
                else:
                    next_state, reward, done, truncated, _ = env.step(action)
                
                agent.remember(state, action, reward, next_state, done or truncated)
                state = next_state
                total_reward += reward
                
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                    if loss > 0:
                        losses.append(loss)
                
                if done or truncated:
                    break
            
            # Update target network
            if episode % target_update_freq == 0:
                agent.update_target_network()
            
            # Store training data
            st.session_state.training_data['episodes'].append(episode + 1)
            st.session_state.training_data['rewards'].append(total_reward)
            st.session_state.training_data['losses'].append(np.mean(losses) if losses else 0)
            st.session_state.training_data['epsilon'].append(agent.epsilon)
            
            # Calculate moving average
            if len(st.session_state.training_data['rewards']) >= 10:
                avg_reward = np.mean(st.session_state.training_data['rewards'][-10:])
            else:
                avg_reward = np.mean(st.session_state.training_data['rewards'])
            st.session_state.training_data['avg_rewards'].append(avg_reward)
            
            # Update progress
            progress_bar.progress((episode + 1) / num_episodes)
            status_text.text(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
            
            # Update plots every 10 episodes
            if episode % 10 == 0:
                update_plots(plot_placeholder, stats_placeholder)
        
        except Exception as e:
            st.error(f"Error during episode {episode + 1}: {str(e)}")
            break
    
    env.close()
    st.session_state.training_completed = True
    st.success(f"Training completed! Final average reward: {avg_reward:.2f}")
    
    # Auto-trigger visualization after training
    st.balloons()
    time.sleep(1)
    visualize_trained_agent()

def visualize_trained_agent():
    """Visualize the trained agent playing the selected environment"""
    if st.session_state.agent is None:
        st.error("No trained agent available!")
        return
    
    env_name = st.session_state.selected_env
    env_config = ENVIRONMENTS[env_name]
    
    try:
        env = gym.make(env_name, render_mode='rgb_array')
    except Exception as e:
        st.error(f"Error creating environment for visualization: {str(e)}")
        return
    
    agent = st.session_state.agent
    
    # Save current epsilon and set to 0 for greedy policy
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    # Container for visualization
    viz_container = st.container()
    with viz_container:
        st.subheader(f"🎬 {env_config['name']} Agent Visualization")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            num_episodes_viz = st.selectbox("Episodes to visualize:", [1, 3, 5], index=0)
        with col2:
            frame_delay = st.slider("Frame delay (ms):", 50, 500, 100)
        with col3:
            show_info = st.checkbox("Show game info", value=True)
    
    # Create placeholders
    image_placeholder = st.empty()
    info_placeholder = st.empty() if show_info else None
    episode_progress = st.progress(0)
    
    st.session_state.agent_frames = []
    
    try:
        for episode in range(num_episodes_viz):
            state, _ = env.reset()
            episode_frames = []
            total_reward = 0
            step_count = 0
            
            episode_progress.progress((episode) / num_episodes_viz)
            
            for step in range(env_config['max_steps']):
                # Get action from trained agent (greedy policy)
                action = agent.act(state)
                
                # Handle continuous vs discrete action spaces
                if env_config['type'] == 'continuous':
                    action_continuous = np.array([action / (agent.action_size - 1) * 2 - 1])
                    if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
                        action_continuous = np.repeat(action_continuous, env.action_space.shape[0])
                    next_state, reward, done, truncated, _ = env.step(action_continuous)
                    action_display = f"Continuous: {action_continuous[0]:.3f}"
                else:
                    next_state, reward, done, truncated, _ = env.step(action)
                    action_display = f"Action: {action}"
                
                total_reward += reward
                step_count += 1
                
                # Render frame
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)
                    
                    # Display current frame
                    img = Image.fromarray(frame)
                    image_placeholder.image(img, caption=f"{env_config['name']} - Episode {episode + 1} - Step {step_count}", use_column_width=True)
                    
                    # Show game info
                    if show_info and info_placeholder:
                        with info_placeholder.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Episode", episode + 1)
                            with col2:
                                st.metric("Step", step_count)
                            with col3:
                                st.metric("Total Reward", f"{total_reward:.2f}")
                            with col4:
                                st.metric("Action", action_display)
                            
                            # Show state information
                            st.write(f"**State ({env_config['type']}):**")
                            state_cols = st.columns(min(len(next_state), 6))
                            for i, val in enumerate(next_state[:6]):
                                with state_cols[i]:
                                    st.write(f"s[{i}]: {val:.3f}")
                
                state = next_state
                
                # Add delay for smooth visualization
                time.sleep(frame_delay / 1000.0)
                
                if done or truncated:
                    break
            
            st.session_state.agent_frames.extend(episode_frames)
            
            # Episode summary
            if show_info and info_placeholder:
                with info_placeholder.container():
                    if total_reward >= env_config['success_threshold']:
                        st.success(f"✅ Episode {episode + 1} - SUCCESS! Reward: {total_reward:.2f} (Steps: {step_count})")
                    else:
                        st.info(f"Episode {episode + 1} - Reward: {total_reward:.2f} (Steps: {step_count})")
        
        episode_progress.progress(1.0)
        
    except Exception as e:
        st.error(f"Error during visualization: {str(e)}")
    
    finally:
        # Restore original epsilon
        agent.epsilon = original_epsilon
        env.close()

def create_gif_from_frames():
    """Create a GIF from collected frames"""
    if not st.session_state.agent_frames:
        st.warning("No frames available to create GIF")
        return None
    
    # Convert frames to PIL Images
    pil_images = []
    for frame in st.session_state.agent_frames[::2]:  # Take every other frame to reduce size
        img = Image.fromarray(frame)
        pil_images.append(img)
    
    # Save as GIF in memory
    gif_buffer = io.BytesIO()
    pil_images[0].save(
        gif_buffer,
        format='GIF',
        save_all=True,
        append_images=pil_images[1:],
        duration=100,  # milliseconds per frame
        loop=0
    )
    gif_buffer.seek(0)
    
    return gif_buffer.getvalue()

def update_plots(plot_placeholder, stats_placeholder):
    if not st.session_state.training_data['episodes']:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Training Loss', 'Epsilon Decay', 'Moving Average Reward'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    episodes = st.session_state.training_data['episodes']
    rewards = st.session_state.training_data['rewards']
    losses = st.session_state.training_data['losses']
    epsilons = st.session_state.training_data['epsilon']
    avg_rewards = st.session_state.training_data['avg_rewards']
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=rewards, mode='lines+markers', name='Reward', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Training loss
    fig.add_trace(
        go.Scatter(x=episodes, y=losses, mode='lines', name='Loss', line=dict(color='red')),
        row=1, col=2
    )
    
    # Epsilon decay
    fig.add_trace(
        go.Scatter(x=episodes, y=epsilons, mode='lines', name='Epsilon', line=dict(color='green')),
        row=2, col=1
    )
    
    # Moving average reward
    fig.add_trace(
        go.Scatter(x=episodes, y=avg_rewards, mode='lines', name='Avg Reward', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Training Metrics")
    plot_placeholder.plotly_chart(fig, use_container_width=True, key=f"training_plots_{len(episodes)}_{int(time.time()*1000)}")
    
    # Update statistics
    if rewards:
        with stats_placeholder.container():
            st.metric("Current Episode", len(episodes))
            st.metric("Last Reward", f"{rewards[-1]:.2f}")
            st.metric("Best Reward", f"{max(rewards):.2f}")
            st.metric("Average Reward", f"{np.mean(rewards):.2f}")
            if losses:
                st.metric("Current Loss", f"{losses[-1]:.4f}")
            st.metric("Current Epsilon", f"{epsilons[-1]:.3f}")

    if not st.session_state.training_data['episodes']:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Training Loss', 'Epsilon Decay', 'Moving Average Reward'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    episodes = st.session_state.training_data['episodes']
    rewards = st.session_state.training_data['rewards']
    losses = st.session_state.training_data['losses']
    epsilons = st.session_state.training_data['epsilon']
    avg_rewards = st.session_state.training_data['avg_rewards']
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=rewards, mode='lines+markers', name='Reward', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Training loss
    fig.add_trace(
        go.Scatter(x=episodes, y=losses, mode='lines', name='Loss', line=dict(color='red')),
        row=1, col=2
    )
    
    # Epsilon decay
    fig.add_trace(
        go.Scatter(x=episodes, y=epsilons, mode='lines', name='Epsilon', line=dict(color='green')),
        row=2, col=1
    )
    
    # Moving average reward
    fig.add_trace(
        go.Scatter(x=episodes, y=avg_rewards, mode='lines', name='Avg Reward', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Training Metrics")
    plot_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Update statistics
    if rewards:
        with stats_placeholder.container():
            st.metric("Current Episode", len(episodes))
            st.metric("Last Reward", f"{rewards[-1]:.2f}")
            st.metric("Best Reward", f"{max(rewards):.2f}")
            st.metric("Average Reward", f"{np.mean(rewards):.2f}")
            if losses:
                st.metric("Current Loss", f"{losses[-1]:.4f}")
            st.metric("Current Epsilon", f"{epsilons[-1]:.3f}")

# Start training
if start_training:
    st.session_state.is_training = True
    train_agent()

# Display current plots if data exists
if st.session_state.training_data['episodes']:
    update_plots(plot_placeholder, stats_placeholder)

# Agent Visualization Section (only show after training is complete)
if st.session_state.training_completed and st.session_state.agent is not None:
    st.header("🎬 Trained Agent Visualization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎮 Visualize Agent", type="primary"):
            visualize_trained_agent()
    
    with col2:
        if st.button("🎬 Create GIF") and st.session_state.agent_frames:
            gif_data = create_gif_from_frames()
            if gif_data:
                st.download_button(
                    label="📥 Download Agent GIF",
                    data=gif_data,
                    file_name="trained_agent_cartpole.gif",
                    mime="image/gif"
                )
    
    with col3:
        if st.button("🔄 Clear Frames"):
            st.session_state.agent_frames = []
            st.success("Frames cleared!")

# Test trained agent section
st.header("🎮 Test Trained Agent")
if st.session_state.agent is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test Agent (5 Episodes)"):
            env = gym.make('CartPole-v1', render_mode='rgb_array')
            agent = st.session_state.agent
            test_rewards = []
            
            for episode in range(5):
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                
                for step in range(500):
                    # Use greedy policy (no exploration)
                    temp_epsilon = agent.epsilon
                    agent.epsilon = 0
                    action = agent.act(state)
                    agent.epsilon = temp_epsilon
                    
                    state, reward, done, truncated, _ = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if done or truncated:
                        break
                
                test_rewards.append(total_reward)
                st.write(f"Test Episode {episode + 1}: {total_reward} steps")
            
            env.close()
            st.write(f"Average test performance: {np.mean(test_rewards):.2f} steps")
    
    with col2:
        st.write("**Agent Status:**")
        if st.session_state.agent:
            st.write(f"✅ Agent initialized")
            st.write(f"🧠 Memory size: {len(st.session_state.agent.memory)}")
            st.write(f"🎯 Current epsilon: {st.session_state.agent.epsilon:.3f}")
        else:
            st.write("❌ No agent initialized")

# Download training data
if st.session_state.training_data['episodes']:
    st.header("💾 Export Training Data")
    
    # Create DataFrame
    df = pd.DataFrame(st.session_state.training_data)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="rl_training_data.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("📊 Show Data Table"):
            st.dataframe(df.tail(20))

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, PyTorch, and Gymnasium")
