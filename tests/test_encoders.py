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
        fixed_finger_initial_position,
        0 # always use n_keyframes=0 here
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

@pytest.mark.parametrize("n_trials", [100])
@pytest.mark.parametrize("fixed_finger_initial_position", [True, False])
@pytest.mark.parametrize("n_keyframes", [0, 1, 2, 3, 4, 5])
def test_reconstruction_from_config_encoding(
        n_trials,
        fixed_finger_initial_position,
        n_keyframes
):
    """
    Proof that there is a function (reconstruct_plan(encoding))
    using which it is always possible to reconstruct the plan
    from the config encoding
    """
    env = gym.make(
        'gym_physx:physx-pushing-v0',
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed', width=0.5),
        fixed_initial_config=None,
        fixed_finger_initial_position=fixed_finger_initial_position,
        plan_generator=None,
        komo_plans=False,
        n_keyframes=n_keyframes,
        plan_length=50*(1+n_keyframes)
    )
    encoder = ConfigEncoder(
        env.box_xy_min, env.box_xy_max,
        env.plan_length, env.dim_plan,
        fixed_finger_initial_position,
        n_keyframes
    )

    for _ in range(n_trials):
        obs = env.reset()
        encoding = encoder.encode(obs['desired_goal'])

        # reconstruct plan only from encoding (and experiment parameters)
        reconstructed_plan = reconstruct_plan(
            encoding,
            fixed_finger_initial_position,
            n_keyframes
        )

        # assert correct reconstruction
        assert np.max(np.abs(reconstructed_plan - obs['desired_goal'])) < 1e-14

def reconstruct_plan(
        encoding,
        fixed_finger_initial_position,
        n_keyframes
):
    """
    reconstruct plan from encoding
    """
    # Quick check
    assert len(encoding) == 2*n_keyframes + (
        4 if fixed_finger_initial_position else 6
    )

    # this env is freshly created only to access its methods.
    new_env = gym.make(
        'gym_physx:physx-pushing-v0',
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed', width=0.5),
        fixed_initial_config=None,
        fixed_finger_initial_position=fixed_finger_initial_position,
        plan_generator=None,
        komo_plans=False,
        n_keyframes=n_keyframes,
        plan_length=50*(1+n_keyframes)
    )

    # extract config from encoding
    finger_position = np.array(
        [0, 0]) if fixed_finger_initial_position else encoding[:2]
    finger_position = np.array(
        list(finger_position) + [0.64]
    )
    offset = 0 if fixed_finger_initial_position else 2
    box_initial_position = encoding[offset:2+offset]
    box_goal_position = encoding[2+offset:4+offset]
    relevant_intermediate_frames = [
        encoding[4+offset+2*index:4+offset+2*index+2]
        for index in range(n_keyframes)
    ]

    # compile keyframes
    keyframes = [np.array(list(box_initial_position) + [new_env.floor_level])]
    for int_frame in relevant_intermediate_frames:
        keyframes.append(np.array(list(int_frame) + [new_env.floor_level]))

    # append goal twice
    keyframes.append(np.array(list(box_goal_position) + [new_env.floor_level]))
    keyframes.append(np.array(list(box_goal_position) + [new_env.floor_level]))

    # and treat second-to-last frame
    if n_keyframes == 0:
        # in this case, the first push is along the longest
        # direction
        first_dir = np.argmax(
            np.abs(keyframes[-1] - keyframes[0])
        )
        assert first_dir in [0, 1]
        second_dir = 0 if first_dir == 1 else 1
        keyframes[-2][first_dir] = keyframes[-1][first_dir]
        keyframes[-2][second_dir] = keyframes[0][second_dir]
    else:
        # in this case, the second-to-last push is
        # perpendicular to the third-to-last
        direction = 1 if keyframes[-4][0] == keyframes[-3][0] else 0
        keyframes[-2][direction] = keyframes[-3][direction]

    # create waypoints
    waypoints = np.array(
        new_env._get_waypoints(finger_position, keyframes) # pylint: disable=protected-access
    )

    # return plan
    return new_env._densify_waypoints(waypoints) # pylint: disable=protected-access
