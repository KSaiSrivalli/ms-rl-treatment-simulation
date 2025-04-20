import numpy as np  # Ensure numpy is imported
import random  # Ensure random is imported
import pickle

class QLearningAgent:
    def __init__(self, state_bins, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_bins = state_bins
        self.action_size = action_size
        self.alpha = alpha   # Learning rate
        self.gamma = gamma   # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.q_table = np.zeros([len(state_bins[0]), len(state_bins[1]), len(state_bins[2]), action_size])  # Q-table
    
    def discretize(self, state):
        """ Convert state to discrete bins """
        return tuple([np.digitize(state[i], self.state_bins[i]) - 1 for i in range(len(state))])
    
    def choose_action(self, state):
        """ Epsilon-greedy action selection """
        state_discretized = self.discretize(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Exploration
        else:
            return np.argmax(self.q_table[state_discretized])  # Exploitation
    
    def update(self, state, action, reward, next_state, done):
        """ Q-learning update rule """
        state_discretized = self.discretize(state)
        next_state_discretized = self.discretize(next_state)
        
        best_next_action = np.argmax(self.q_table[next_state_discretized])
        td_target = reward + (self.gamma * self.q_table[next_state_discretized + (best_next_action,)] if not done else 0)
        td_error = td_target - self.q_table[state_discretized + (action,)]
        self.q_table[state_discretized + (action,)] += self.alpha * td_error

    def save(self, file_path):
        """ Save the Q-table to a file """
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, file_path):
        """ Load the Q-table from a file """
        with open(file_path, 'rb') as f:
            self.q_table = pickle.load(f)
