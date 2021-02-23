"""
Tests for the gym-physx wrappers
"""
# %%
import gym
from gym_physx.envs.shaping import PlanBasedShaping
from gym_physx.wrappers import DesiredGoalEncoder
from gym_physx.encoders import ToyEncoder


def test_toy_wrapper():
    """
    Test that the toy wrapper correctly manipulates observation space
    as well as .step() and reset()
    """
    env = gym.make(
        'gym_physx:physx-pushing-v0',
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed')
    )
    assert len(env.observation_space.sample()["desired_goal"]) == 300
    observation = env.reset()
    assert len(observation["desired_goal"]) == 300
    observation, _, _, _ = env.step(env.action_space.sample())
    assert len(observation["desired_goal"]) == 300

    encoder = ToyEncoder()
    env = DesiredGoalEncoder(env, encoder)

    assert len(env.observation_space.sample()["desired_goal"]) == 5
    observation = env.reset()
    assert len(observation["desired_goal"]) == 5
    observation, _, _, _ = env.step(env.action_space.sample())
    assert len(observation["desired_goal"]) == 5

# %%