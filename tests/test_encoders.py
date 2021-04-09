# %%
"""
Tests of the encoder classes
"""

import numpy as np
import pytest
import gym
from gym_physx.envs.shaping import PlanBasedShaping
from gym_physx.encoders.config_encoder import ConfigEncoder
from gym_physx.wrappers import DesiredGoalEncoder

@pytest.mark.parametrize("n_trials", [20])
@pytest.mark.parametrize("fixed_finger_initial_position", [True, False])
@pytest.mark.parametrize("komo_plans", [True, False])
def test_config_encoder(n_trials, fixed_finger_initial_position, komo_plans):
    """
    Test the ConfigEncoder class
    """
    env = gym.make(
        'gym_physx:physx-pushing-v0',
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed', width=0.5),
        fixed_initial_config=None,
        fixed_finger_initial_position=fixed_finger_initial_position,
        plan_generator=None,
        komo_plans=komo_plans
    )

    encoder = ConfigEncoder(
        env.box_xy_min, env.box_xy_max,
        env.plan_length, env.dim_plan,
        fixed_finger_initial_position
    )

    env = DesiredGoalEncoder(env, encoder)

    for _ in range(n_trials):
        observation = env.reset()
        observation, _, _, info = env.step(env.action_space.sample())

        assert env.observation_space.contains(observation)

        if fixed_finger_initial_position:
            assert observation['desired_goal'].shape == (4,)
            assert np.all(observation['desired_goal'][:2] == info[
                "original_plan"
            ].reshape(env.plan_length, env.dim_plan)[0, 3:5])
            assert np.all(observation['desired_goal'][2:] == info[
                "original_plan"
            ].reshape(env.plan_length, env.dim_plan)[-1, 3:5])
        else:
            assert observation['desired_goal'].shape == (6,)
            assert np.all(observation['desired_goal'][:2] == info[
                "original_plan"
            ].reshape(env.plan_length, env.dim_plan)[0, :2])
            assert np.all(observation['desired_goal'][2:4] == info[
                "original_plan"
            ].reshape(env.plan_length, env.dim_plan)[0, 3:5])
            assert np.all(observation['desired_goal'][4:] == info[
                "original_plan"
            ].reshape(env.plan_length, env.dim_plan)[-1, 3:5])


# %%
