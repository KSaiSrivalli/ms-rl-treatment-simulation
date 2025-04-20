import streamlit as st
import numpy as np
import os
import pickle

from agent.q_learning import QLearningAgent
from environment.ms_env import MSTreatmentEnv

# Define bins (make sure this matches what you used during training)
state_bins = [
    np.linspace(0, 10, 5),  # Symptom severity
    np.linspace(0, 10, 5),  # Inflammation level
    np.linspace(0, 10, 5)   # Fatigue level
]

# Path to Q-table
q_table_path = "models/q_table.pkl"  # Adjust if needed

# Initialize agent
agent = QLearningAgent(state_bins=state_bins, action_size=4)

# Attempt to load Q-table
if os.path.exists(q_table_path) and os.path.getsize(q_table_path) > 0:
    try:
        agent.load(q_table_path)
        st.success("✅ Q-table loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Failed to load Q-table. Agent will act untrained.\n\nError: {e}")
else:
    st.warning("⚠️ No trained Q-table found. Agent will act untrained.")

# Set up Streamlit app
st.title("🧠 MS Treatment Recommendation (RL Agent)")
st.write("Simulating treatment recommendation for Multiple Sclerosis using Reinforcement Learning")

# User inputs for state
symptom = st.slider("Symptom severity", 0.0, 10.0, 5.0)
inflammation = st.slider("Inflammation level", 0.0, 10.0, 5.0)
fatigue = st.slider("Fatigue level", 0.0, 10.0, 5.0)

# Create environment and set state
env = MSTreatmentEnv()
state = np.array([symptom, inflammation, fatigue])
env.state = state

# Display state
st.subheader("📊 Current Patient State")
st.write(f"Symptom Severity: {symptom}")
st.write(f"Inflammation Level: {inflammation}")
st.write(f"Fatigue Level: {fatigue}")

# Agent chooses action
if st.button("🩺 Recommend Treatment"):
    action = agent.choose_action(state)
    
    treatment_map = {
        0: "Drug A",
        1: "Drug B",
        2: "Lifestyle Intervention",
        3: "Combined Therapy"
    }

    next_state, reward, done, _ = env.step(action)
    
    st.subheader("🧾 Agent Recommendation")
    st.write(f"**Recommended Treatment**: {treatment_map[action]}")
    st.write(f"**Expected Reward**: {reward:.2f}")
    st.write(f"**Projected Next State**: {next_state}")

    if done:
        st.info("🚨 End of episode (recovery or worsening detected)")
