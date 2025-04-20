from environment.ms_env import MSTreatmentEnv
from agent.q_learning import QLearningAgent
import numpy as np

# Environment
env = MSTreatmentEnv()

# Define bins for discretization
state_bins = [
    np.linspace(0, 10, 6),  # symptom
    np.linspace(0, 10, 6),  # inflammation
    np.linspace(0, 10, 6),  # fatigue
]

# Agent
agent = QLearningAgent(state_bins=state_bins, action_size=4)

# Training
episodes = 50
print("Training agent...")
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    for step in range(50):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f"Episode {episode+1}/{episodes} — Total Reward: {total_reward:.2f}")

# Save model
save_path = "q_table.pkl"
agent.save(save_path)
print(f"Training complete. Model saved to {save_path}")
