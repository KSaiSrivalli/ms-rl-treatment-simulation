import numpy as np

class MSTreatmentEnv:
    def __init__(self):
        # Simplified treatment parameters
        self.fatigue = 0.5
        self.edss = 5
        self.relapse = 0.2
        self.done = False

    def reset(self):
        """ Reset environment to start a new episode """
        self.fatigue = 0.5
        self.edss = 5
        self.relapse = 0.2
        self.done = False
        return [self.fatigue, self.edss, self.relapse]
    
    def step(self, action):
        """ Simulate taking an action in the environment """
        # Simplified treatment effects based on action
        if action == 0:  # No Treatment
            self.fatigue += 0.05
            self.edss += 0.2
            self.relapse += 0.05
        elif action == 1:  # Mild Treatment (Drug A)
            self.fatigue -= 0.03
            self.edss -= 0.1
            self.relapse -= 0.02
        elif action == 2:  # Strong Treatment (Drug B)
            self.fatigue -= 0.05
            self.edss -= 0.2
            self.relapse -= 0.03
        else:  # Therapy
            self.fatigue -= 0.02
            self.edss -= 0.05
            self.relapse -= 0.01
        
        # Check for terminal condition
        if self.edss >= 8:
            self.done = True
            reward = -10  # Severe deterioration
        elif self.edss <= 0:
            self.done = True
            reward = 10  # Full recovery
        else:
            self.done = False
            reward = -0.1  # Mild deterioration
        
        # Return next state, reward, and whether the episode is done
        return [self.fatigue, self.edss, self.relapse], reward, self.done, {}
