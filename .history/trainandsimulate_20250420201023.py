from environment.ms-env import MSTreatmentEnv
from agent.q_learning import QLearningAgent
import numpy as np

# Create environment
env = MSTreatmentEnv()

# Define state bins for discretization (same length as observation dimensions)
state_bins = [
    np.linspace(0, 1, 10),   # fatigue
    np.linspace(0, 10, 10),  # edss
    np.linspace(0, 1, 10)    # relapse risk
]

# Initialize agent
agent = QLearningAgent(state_bins=state_bins, action_size=env.action_space.n)

# Training config
episodes = 500
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

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{episodes} — Total Reward: {total_reward:.2f} — Epsilon: {agent.epsilon:.2f}")

# Save trained model
agent.save(save_path)
print(f"Training complete. Model saved to {save_path}")
