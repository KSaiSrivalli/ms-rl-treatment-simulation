import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, state_bins, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initializes the QLearningAgent.
        :param state_bins: List of bins for each state variable (e.g., [5, 5, 5] for Symptom, Inflammation, Fatigue)
        :param action_size: Number of actions the agent can take
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration rate
        """
        self.state_bins = state_bins  # List of bins for each state variable
        self.action_size = action_size  # Number of actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table with zeros (dimensions: state_bins x action_size)
        self.q_table = np.zeros(state_bins + [action_size])

    def choose_action(self, state):
        """
        Chooses an action based on epsilon-greedy policy.
        :param state: The current state of the agent
        :return: Action to take
        """
        if np.random.rand() < self.epsilon:  # Exploration
            return np.random.randint(self.action_size)
        else:  # Exploitation
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the agent's experience.
        :param state: The current state
        :param action: The action taken
        :param reward: The reward received
        :param next_state: The resulting state after taking the action
        """
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                                      self.alpha * (reward + self.gamma * self.q_table[next_state][best_next_action])

    def save(self, filename):
        """
        Saves the Q-table to a file.
        :param filename: File to save the Q-table to
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        """
        Loads the Q-table from a file.
        :param filename: File to load the Q-table from
        """
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
