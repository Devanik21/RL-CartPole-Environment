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
if 'is_training' not in st.session_state:
    st.session_state.is_training = False

# Main title
st.title("🤖 Reinforcement Learning Agent Training Dashboard")
st.markdown("### CartPole Environment - Deep Q-Network (DQN)")

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
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
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
    
    st.sidebar.success("Agent initialized successfully!")
    env.close()

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
    
    env = gym.make('CartPole-v1')
    agent = st.session_state.agent
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for episode in range(num_episodes):
        if stop_training:
            break
            
        state, _ = env.reset()
        total_reward = 0
        losses = []
        
        for step in range(500):  # Max steps per episode
            action = agent.act(state)
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
    
    env.close()
    st.success(f"Training completed! Final average reward: {avg_reward:.2f}")

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
