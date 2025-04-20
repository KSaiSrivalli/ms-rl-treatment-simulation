import gym
import numpy as np
from gym import spaces

class MSTreatmentEnv(gym.Env):
    def __init__(self):
        super(MSTreatmentEnv, self).__init__()

        # State: [symptom_severity, inflammation_level, fatigue_level] (values 0-10)
        self.observation_space = spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)

        # Actions: [0: Drug A, 1: Drug B, 2: Lifestyle Change, 3: No Action]
        self.action_space = spaces.Discrete(4)

        self.state = None
        self.prev_state = None

    def reset(self):
        self.state = np.random.uniform(4, 8, size=3)  # Moderate disease
        self.prev_state = self.state.copy()
        return self.state

    def step(self, action):
        if action == 0:  # Drug A
            self.state[0] = max(0, self.state[0] - 1.5)
            self.state[1] = max(0, self.state[1] - 2.0)
        elif action == 1:  # Drug B
            self.state[0] = max(0, self.state[0] - 1.0)
            self.state[1] = max(0, self.state[1] - 1.5)
            self.state[2] = max(0, self.state[2] - 0.5)
        elif action == 2:  # Lifestyle change
            self.state[2] = max(0, self.state[2] - 1.2)
        elif action == 3:  # No action
            self.state += np.random.normal(0.5, 0.2, size=3)  # slight worsening

        self.state = np.clip(self.state, 0, 10)

        # Reward: total improvement from last state
        delta = self.prev_state - self.state
        reward = np.sum(delta)

        # Bonus reward if all 3 improved
        if np.all(delta > 0):
            reward += 1.0

        # Penalty if all worsened
        if np.all(delta < 0):
            reward -= 1.0

        self.prev_state = self.state.copy()
        done = False
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Current State: Symptom: {self.state[0]:.2f}, Inflammation: {self.state[1]:.2f}, Fatigue: {self.state[2]:.2f}")
