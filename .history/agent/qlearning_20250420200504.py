import numpy as np
import os
import pickle

class QLearningAgent:
    def __init__(self, state_bins, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_bins = state_bins  # list of np arrays to discretize each feature
        self.action_size = action_size
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros(tuple(len(b) + 1 for b in state_bins) + (action_size,))

    def discretize_state(self, state):
        return tuple(np.digitize(s, bins) for s, bins in zip(state, self.state_bins))

    def choose_action(self, state):
        state_idx = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state_idx])

    def update(self, state, action, reward, next_state, done):
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)

        best_next_action = np.max(self.q_table[next_state_idx])
        td_target = reward + self.gamma * best_next_action * (not done)
        td_error = td_target - self.q_table[state_idx + (action,)]

        self.q_table[state_idx + (action,)] += self.alpha * td_error

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
