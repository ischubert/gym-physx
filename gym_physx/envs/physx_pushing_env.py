"""
PhysX-based Robotic Pushing Environment
"""
import gym


class PhysxPushingEnv(gym.Env):
    """
    PhysX-based Robotic Pushing Environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
