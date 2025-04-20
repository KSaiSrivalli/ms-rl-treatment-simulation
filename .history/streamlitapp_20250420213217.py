import streamlit as st
import numpy as np
import pickle
from environment.ms_env import MSTreatmentEnv
from agent.q_learning import QLearningAgent

# Streamlit app layout
st.title("Multiple Sclerosis Treatment Simulation")
st.markdown("""
    A Reinforcement Learning-based Agent for MS Treatment Planning.
    This application simulates the treatment planning for Multiple Sclerosis (MS)
    using a reinforcement learning (RL) agent. The agent learns how to treat MS symptoms 
    through actions like medications, lifestyle changes, and more.
""")

# Initialize environment and agent
state_bins = [5, 5, 5]  # 5 bins for each of Symptom, Inflammation, and Fatigue
action_size = 4  # 4 possible actions
env = MSTreatmentEnv()

# Initialize QLearningAgent
agent = QLearningAgent(state_bins=state_bins, action_size=action_size)

# Load previously trained Q-table if available
q_table_path = 'q_table.pkl'
try:
    with open(q_table_path, 'rb') as f:
        agent.q_table = pickle.load(f)
    st.write("Trained Q-table loaded.")
except FileNotFoundError:
    st.warning("No trained Q-table found. Agent will act untrained.")

# Run the simulation
if st.button("Start Simulation"):
    state = env.reset()  # Reset environment to initial state
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(tuple(state))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Update Q-table
        agent.update_q_table(tuple(state), action, reward, tuple(next_state))
        state = next_state

        # Display the simulation progress
        st.write(f"Action: {action}, State: {state}, Reward: {reward}, Total Reward: {total_reward}")

    st.write(f"Simulation Complete. Total Reward: {total_reward}")

    # Save the Q-table after training
    agent.save(q_table_path)
    st.write("Q-table saved.")

