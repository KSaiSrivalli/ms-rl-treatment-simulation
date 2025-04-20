from environment.ms_env import MSTreatmentEnv
from agent.q_learning import QLearningAgent
import numpy as np
import pickle  # Make sure pickle is imported


# Create environment
env = MSTreatmentEnv()

# Define state bins for discretization (with fewer bins for faster processing)
state_bins = [
    np.linspace(0, 1, 5),   # fatigue
    np.linspace(0, 10, 5),  # edss
    np.linspace(0, 1, 5)    # relapse
]

# Initialize agent
agent = QLearningAgent(state_bins=state_bins, action_size=4)

# Training config
episodes = 50  # Reduce to 50 episodes for faster results
save_path = "q_table.pkl"

print("Training agent...")

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Episode {ep+1}/{episodes} — Total Reward: {total_reward:.2f}")

# Save trained model
agent.save(save_path)
print(f"Training complete. Model saved to {save_path}")
