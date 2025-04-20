import numpy as np

class MSTreatmentEnv:
    def __init__(self):
        self.state = None  # Initialize state as None
        self.action_space = 4  # Number of possible actions (Drug A, Drug B, Lifestyle Change, No Action)
        self.state_bounds = np.array([10, 10, 10])  # Bounds for symptom, inflammation, and fatigue

    def reset(self):
        """
        Reset the environment state to a random initial state
        """
        self.state = np.random.uniform(0, 10, size=3)  # Initial state: symptom, inflammation, fatigue
        return self.state

    def step(self, action):
        """
        Take an action and return the new state and reward
        """
        # Ensure that the state is not None before proceeding
        if self.state is None:
            self.state = self.reset()

        # Simulate treatment response based on the action taken
        if action == 0:  # Drug A
            self.state[0] = max(0, self.state[0] - 1.0)  # Symptom improvement
        elif action == 1:  # Drug B
            self.state[1] = max(0, self.state[1] - 1.5)  # Inflammation reduction
        elif action == 2:  # Lifestyle change
            self.state[2] = max(0, self.state[2] - 0.5)  # Fatigue reduction
        elif action == 3:  # No action
            self.state[0] = max(0, self.state[0] - 0.2)  # Symptom slightly decreases

        # Calculate reward
        reward = -np.sum(self.state)  # Example reward: lower state is better

        # Done condition: if all states are zero or below
        done = np.all(self.state <= 0)

        return self.state, reward, done, {}

