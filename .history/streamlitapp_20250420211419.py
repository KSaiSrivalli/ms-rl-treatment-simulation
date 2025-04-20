import streamlit as st
import numpy as np
import pickle
import os

from environment.ms_env import MSTreatmentEnv
from agent.q_learning import QLearningAgent

# Load trained Q-table
q_table_path = "q_table.pkl"

if os.path.exists(q_table_path):
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    st.success("✅ Q-table loaded successfully!")
else:
    st.warning("⚠️ No trained Q-table found. Agent will act untrained.")
    q_table = None

# Initialize environment
env = MSTreatmentEnv()

# Initialize agent with matching state bins
state_bins = [
    np.linspace(0, 10, 5),  # symptom severity
    np.linspace(0, 10, 5),  # inflammation
    np.linspace(0, 10, 5)   # fatigue
]

agent = QLearningAgent(state_bins=state_bins, action_size=4)
agent.q_table = q_table if q_table is not None else agent.q_table

# Streamlit UI
st.title("🧠 MS Treatment Recommender using Reinforcement Learning")

st.markdown("Adjust patient state below:")

symptom = st.slider("Symptom Severity", 0.0, 10.0, 5.0)
inflammation = st.slider("Inflammation Level", 0.0, 10.0, 5.0)
fatigue = st.slider("Fatigue Level", 0.0, 10.0, 5.0)

state = np.array([symptom, inflammation, fatigue])

if st.button("🩺 Recommend Treatment"):
    action = agent.choose_action(state)
    action_names = ["Drug A", "Drug B", "Lifestyle Change", "No Action"]

    st.subheader(f"🧾 Recommended Action: **{action_names[action]}**")

    # Simulate step to show projected outcome
    next_state, reward, done, _ = env.step(action)
    st.metric("📊 Expected Reward", f"{reward:.2f}")
    st.markdown("**Next State (Predicted):**")
    st.write({
        "Symptom Severity": round(next_state[0], 2),
        "Inflammation Level": round(next_state[1], 2),
        "Fatigue Level": round(next_state[2], 2),
    })
