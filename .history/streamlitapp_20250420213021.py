import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from environment.ms_env import MSTreatmentEnv
from agent.q_learning import QLearningAgent

# Title and header
st.title("Multiple Sclerosis Treatment Simulation")
st.subheader("A Reinforcement Learning-based Agent for MS Treatment Planning")
st.markdown("""
    This application simulates the treatment planning for Multiple Sclerosis (MS) using a reinforcement learning (RL) agent.
    The agent learns how to treat MS symptoms through actions like medications, lifestyle changes, and more.
""")

# Load or train the Q-learning agent
save_path = "q_table.pkl"
agent = QLearningAgent(state_bins=[5, 5, 5], action_size=4)

# Try loading the trained Q-table
try:
    with open(save_path, "rb") as f:
        agent.q_table = pickle.load(f)
    st.success("Trained Q-table successfully loaded!")
except (FileNotFoundError, EOFError):
    st.warning("⚠️ No trained Q-table found. Agent will act untrained.")
    agent.q_table = np.zeros([5, 5, 5, agent.action_size])  # Initialize Q-table if not found

# Initialize environment
env = MSTreatmentEnv()

# Interactive simulation start
start_simulation = st.button("Start Simulation")

if start_simulation:
    st.write("Simulation starting...")

    # Track performance over episodes
    rewards = []

    # Simulate the agent's actions and state transitions
    for episode in range(50):
        state = env.reset()
        total_reward = 0

        st.write(f"Episode {episode + 1}/50 — Initial State: Symptom: {state[0]}, Inflammation: {state[1]}, Fatigue: {state[2]}")

        for step in range(100):  # Max steps per episode
            action = agent.choose_action(state)  # Choose action based on the Q-table
            next_state, reward, done, _ = env.step(action)

            # Update total reward
            total_reward += reward

            # Update Q-table using the learning algorithm
            agent.update_q_table(state, action, reward, next_state)

            if done:
                break

            state = next_state  # Move to the next state

        rewards.append(total_reward)
        st.write(f"Episode {episode + 1} — Total Reward: {total_reward:.2f}")

    # Plot performance graph
    fig, ax = plt.subplots()
    ax.plot(range(1, 51), rewards, label="Total Reward per Episode")
    ax.set(xlabel="Episode", ylabel="Total Reward", title="Total Reward over 50 Episodes")
    st.pyplot(fig)

    # Save the trained Q-table
    agent.save(save_path)
    st.success("Training complete. Model saved to q_table.pkl.")

# Display simulation results
st.subheader("Simulation Results")
st.write("After running the simulation, the agent was trained over 50 episodes. The total reward over time improves as the agent learns optimal treatment strategies.")

# Display the current state and action choices
action_choice = st.selectbox("Choose an action for the agent:", ["Drug A", "Drug B", "Lifestyle Change", "No Action"])

st.write(f"You chose: {action_choice}")
