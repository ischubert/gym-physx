"""
OpenAI-Gym Wrapper for PhysX-based Robotic Pushing Environment
"""
from gym.envs.registration import register
from .envs import shaping

register(
    id='physx-pushing-v0',
    entry_point='gym_physx.envs:PhysxPushingEnv',
)