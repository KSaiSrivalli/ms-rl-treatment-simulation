import streamlit as st
import numpy as np
import pickle
from environment.ms_env import MSTreatmentEnv
from agent.q_learning import QLearningAgent

# Load trained Q-table
try:
    with open("q_table.pkl", 'rb') as f:
        q_table = pickle.load(f)
    trained = True
except:
    trained = False

# Streamlit app UI
st.title('Multiple Sclerosis Treatment Simulator')

if trained:
    st.success("⚡️ Trained Q-table found!")
else:
    st.warning("⚠️ No trained Q-table found. Agent will act untrained.")

# Initialize environment and agent
env = MSTreatmentEnv()
state_bins = [
    np.linspace(0, 10, 6),
    np.linspace(0, 10, 6),
    np.linspace(0, 10, 6),
]
agent = QLearningAgent(state_bins=state_bins, action_size=4)
if trained:
    agent.q_table = q_table

# User input: simulate action
symptom = st.slider('Symptom Severity', 0, 10, 5)
inflammation = st.slider('Inflammation Level', 0, 10, 5)
fatigue = st.slider('Fatigue Level', 0, 10, 5)

state = np.array([symptom, inflammation, fatigue])

# Choose action using the agent
if trained:
    action = agent.choose_action(state)
    action_labels = ['Drug A', 'Drug B', 'Lifestyle Change', 'No Action']
    st.write(f"Action chosen by agent: {action_labels[action]}")
else:
    st.write("Agent is untrained, choosing random action.")

# Simulate environment with the action
next_state, reward, _, _ = env.step(action)

# Display results
st.write(f"New State: Symptom: {next_state[0]:.2f}, Inflammation: {next_state[1]:.2f}, Fatigue: {next_state[2]:.2f}")
st.write(f"Reward: {reward:.2f}")
