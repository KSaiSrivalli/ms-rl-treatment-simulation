import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, state_bins, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_bins = state_bins  # state discretization
        self.action_size = action_size  # number of possible actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # decay factor for epsilon
        self.epsilon_min = epsilon_min  # minimum value for epsilon

        # Initialize Q-table with zeros
        self.q_table = np.zeros([len(state_bins[0]), len(state_bins[1]), len(state_bins[2]), action_size])  # Q-table

    def discretize(self, state):
        """
        Convert continuous state into discrete state (index based on bins).
        """
        state_discretized = []
        for i in range(len(state)):
            state_discretized.append(np.digitize(state[i], self.state_bins[i]) - 1)
        return tuple(state_discretized)

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy strategy.
        """
        state_discretized = self.discretize(state)

        # Explore: random action
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        # Exploit: choose best action
        return np.argmax(self.q_table[state_discretized])

    def learn(self, state, action, reward, next_state):
        """
        Update Q-table using the Q-learning update rule.
        """
        state_discretized = self.discretize(state)
        next_state_discretized = self.discretize(next_state)

        # Update Q-value using the Q-learning equation
        old_q_value = self.q_table[state_discretized + (action,)]
        future_q_value = np.max(self.q_table[next_state_discretized])

        # Q-learning update rule
        self.q_table[state_discretized + (action,)] = old_q_value + self.alpha * (reward + self.gamma * future_q_value - old_q_value)

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, save_path):
        """
        Save the Q-table to a file.
        """
        with open(save_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, load_path):
        """
        Load the Q-table from a file.
        """
        with open(load_path, 'rb') as f:
            self.q_table = pickle.load(f)
