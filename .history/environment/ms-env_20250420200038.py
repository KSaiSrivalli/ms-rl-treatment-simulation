import gym
import numpy as np
from gym import spaces

class MSTreatmentEnv(gym.Env):
    def __init__(self):
        super(MSTreatmentEnv, self).__init__()
        # States: fatigue, edss, relapse_risk
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]),
                                            high=np.array([1.0, 10.0, 1.0]),
                                            dtype=np.float32)
        # Actions: 0 = No treatment, 1 = Drug A, 2 = Drug B, 3 = Therapy
        self.action_space = spaces.Discrete(4)
        self.state = None

    def reset(self):
        self.state = np.array([0.5, 3.0, 0.3])  # initial condition
        return self.state

    def step(self, action):
        fatigue, edss, relapse = self.state

        # Apply action effects (simple rules + noise)
        if action == 0:  # No treatment
            fatigue += 0.05
            edss += 0.1
        elif action == 1:  # Drug A
            fatigue -= 0.1
            relapse += 0.05
        elif action == 2:  # Drug B
            fatigue -= 0.15
            edss -= 0.2
            relapse += 0.1
        elif action == 3:  # Therapy
            fatigue -= 0.05
            edss -= 0.05

        # Clip state values to valid range
        self.state = np.clip([fatigue, edss, relapse],
                             [0.0, 0.0, 0.0],
                             [1.0, 10.0, 1.0])

        # Define reward (lower fatigue and EDSS = better)
        reward = - (0.6 * self.state[0] + 0.4 * self.state[1])

        done = bool(self.state[1] >= 8.0)  # terminal if condition gets too bad

        return self.state, reward, done, {}
