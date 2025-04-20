import streamlit as st
import numpy as np
import pickle
from environment.ms_env import MSTreatmentEnv
from agent.q_learning import QLearningAgent

# Load trained agent
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Define same state bins used in training
state_bins = [
    np.linspace(0, 1, 10),   # fatigue
    np.linspace(0, 10, 10),  # edss
    np.linspace(0, 1, 10)    # relapse
]

# Recreate agent with loaded Q-table
agent = QLearningAgent(state_bins, action_size=4)
agent.q_table = q_table

# Create environment
env = MSTreatmentEnv()

# Initialize session state
if "state" not in st.session_state:
    st.session_state.state = env.reset()
    st.session_state.history = []

# App title
st.title("🧠 MS Treatment Agent (RL)")
st.write("An interactive simulation of a reinforcement learning agent for Multiple Sclerosis treatment planning.")

# Display current state
fatigue, edss, relapse = st.session_state.state
st.metric("Fatigue", f"{fatigue:.2f}")
st.metric("EDSS Score", f"{edss:.2f}")
st.metric("Relapse Risk", f"{relapse:.2f}")

# Agent action
state = st.session_state.state
action = agent.choose_action(state)
actions_dict = {
    0: "No Treatment",
    1: "Drug A (mild)",
    2: "Drug B (strong)",
    3: "Therapy"
}

# Show selected action
st.subheader("🩺 Agent’s Suggested Action")
st.success(f"{actions_dict[action]}")

# Allow user to override
user_override = st.radio("👤 Override Agent's Action?", list(actions_dict.items()), index=action)

# Button to simulate one step
if st.button("🚀 Apply Treatment"):
    next_state, reward, done, _ = env.step(user_override[0])
    st.session_state.history.append((*st.session_state.state, user_override[0], reward))
    st.session_state.state = next_state

    if done:
        st.warning("❗Patient has reached a critical condition (EDSS ≥ 8.0). Resetting...")
        st.session_state.state = env.reset()
        st.session_state.history = []

# Plot history
if st.session_state.history:
    import pandas as pd
    import matplotlib.pyplot as plt

    hist = pd.DataFrame(st.session_state.history,
                        columns=["Fatigue", "EDSS", "Relapse", "Action", "Reward"])
    
    st.subheader("📈 Patient History")
    st.line_chart(hist[["Fatigue", "EDSS", "Relapse"]])
