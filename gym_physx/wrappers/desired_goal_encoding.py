import numpy as np
import gym


class DesiredGoalEncoder(gym.Wrapper):
    """
    Encode desired_goal
    """
    def __init__(self, env, encoder):
        super(DesiredGoalEncoder, self).__init__(env)
        self.encoder = encoder
        # modify space for desired_goal
        self.observation_space.spaces["desired_goal"] = gym.spaces.Box(
            low=np.array(self.encoder.hidden_dim * [self.encoder.encoding_low]),
            high=np.array(self.encoder.hidden_dim * [self.encoder.encoding_high])
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info["original_plan"] = observation["desired_goal"].copy()
        observation["desired_goal"] = self.encoder.encode(
            observation["desired_goal"]
        )
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        observation["desired_goal"] = self.encoder.encode(
            observation["desired_goal"]
        )
        return observation
