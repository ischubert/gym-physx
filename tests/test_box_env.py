# %%
"""
Tests for the PhysxPushingEnv
"""

import os
import glob
import json
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import pytest
import gym
from stable_baselines3 import HER, DDPG, SAC, TD3
from gym_physx.envs.shaping import PlanBasedShaping
from gym_physx.generators.plan_generator import PlanFromDiskGenerator

@pytest.mark.parametrize("n_trials", [20])
@pytest.mark.parametrize("from_disk", [True, False])
def test_plan_generator_from_file(n_trials, from_disk):
    """
    Test the plan generator class that provides plans
    loaded from the disk
    """
    # load test files
    data_path = os.path.join(os.path.dirname(__file__), 'test_plans')

    # generate generator object
    plan_dim = 6
    plan_len = 50

    # either let generator load plans from disk
    if from_disk:
        file_list = glob.glob(
            os.path.join(
                data_path,
                "plans_*.pkl"
            )
        )
        num_plans_per_file = 1000
        plan_array = None
        flattened = False

    # or load plans beforehand and provide it to the generator as object
    else:
        file_list = None
        num_plans_per_file = None
        with open(os.path.join(data_path, "buffered_plans.pkl"), 'rb') as data_stream:
            plan_array = pickle.load(data_stream)
        flattened = True

    generator = PlanFromDiskGenerator(
        plan_dim,
        plan_len,
        file_list=file_list,
        num_plans_per_file=num_plans_per_file,
        plan_array=plan_array,
        flattened=flattened
    )

    # Assert (again; done in __init__() as well) that files are in the expected format
    generator.test_consistency()

    env_gen = gym.make(
        'gym_physx:physx-pushing-v0',
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed'),
        fixed_initial_config=None,
        plan_generator=generator
    )

    env_plan = gym.make(
        'gym_physx:physx-pushing-v0',
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed'),
        fixed_initial_config=None,
        plan_generator=None
    )

    trials = []
    for _ in range(n_trials):
        obs_gen = env_gen.reset()

        plan_gen = obs_gen['desired_goal'].reshape(generator.plan_len, generator.plan_dim)
        finger_position = plan_gen[0, :2]
        box_position = plan_gen[0, 3:5]
        goal_position = plan_gen[-1, 3:5]

        obs_plan = env_plan._controlled_reset(  # pylint: disable=protected-access
            finger_position,
            box_position,
            goal_position
        )

        # assert that the saved plan and the recomputed plan are approximately consistent
        trials.append(
            np.mean(
                np.abs(obs_gen['desired_goal'] - obs_plan['desired_goal'])
            ) < 0.05
        )

    assert np.mean(trials) >= 0.9

def test_observations(view=False, n_trials=5):
    """
    Test the consistency of all observations
    """

    shaping_objects = [
        PlanBasedShaping(shaping_mode=strategy, gamma=gamma)
        for strategy, gamma in zip(
            [None, 'relaxed', 'potential_based'],
            [None, None, 0.9]
        )
    ]

    with open(os.path.join(
            os.path.dirname(__file__),
            'expected_rewards.json'
    ), 'r') as data:
        expected_rewards = json.load(data)["expected_rewards"]

    for shaping_object, expected_reward in zip(
            shaping_objects,
            expected_rewards
    ):
        for _ in range(n_trials):
            env = gym.make(
                'gym_physx:physx-pushing-v0',
                plan_based_shaping=shaping_object
            )
            if view:
                view = env.render()

            states, achieved_goals, desired_goals, rewards, dones, infos = [], [], [], [], [], []

            obs = env._controlled_reset(  # pylint: disable=protected-access
                [-0.3, 0],
                [-0.6, -0.6],
                [0.6, 0.6]
            )
            states.append(obs["observation"])
            achieved_goals.append(obs["achieved_goal"])
            desired_goals.append(obs["desired_goal"])

            actions = [
                [-0.05, 0, 0],
                [0, -0.05, 0],
                [0.05, 0, 0],
                [0, -0.05, 0],
                [0.05, 0, 0],
                [0, 0.05, 0],
            ]
            durations = [15, 13, 25, 5, 5, 26]

            assert len(actions) == len(durations)
            for action, duration in zip(actions, durations):
                for timestep in range(duration):
                    obs, reward, done, info = env.step(action)

                    states.append(obs["observation"])
                    achieved_goals.append(obs["achieved_goal"])
                    desired_goals.append(obs["desired_goal"])
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)

                    # This also checks for all subspaces
                    assert env.observation_space.contains(obs)
                    assert env.action_space.contains(action)

                    if view and (timestep % 10 == 0 or duration-timestep < 3):
                        fig = plt.figure()
                        axis = fig.gca(projection='3d')
                        axis.set_title("Dims 0 to 2")
                        axis.plot(
                            np.array(states)[:, 0],
                            np.array(states)[:, 1],
                            np.array(states)[:, 2],
                            marker='v',
                            label='states 0-2'
                        )
                        if shaping_object.shaping_mode is not None:
                            axis.plot(
                                np.array(achieved_goals)[:, 0],
                                np.array(achieved_goals)[:, 1],
                                np.array(achieved_goals)[:, 2],
                                label='achieved goals 0-2'
                            )
                            axis.plot(
                                np.array(desired_goals).reshape(
                                    (-1, env.plan_length, env.dim_plan))[-1, :, 0],
                                np.array(desired_goals).reshape(
                                    (-1, env.plan_length, env.dim_plan))[-1, :, 1],
                                np.array(desired_goals).reshape(
                                    (-1, env.plan_length, env.dim_plan))[-1, :, 2],
                                label='latest plan 0-2'
                            )
                        axis.legend()
                        plt.show()
                        plt.show()

                        fig = plt.figure()
                        axis = fig.gca(projection='3d')
                        axis.set_title("Dims 3 to 5")
                        axis.plot(
                            np.array(states)[:, 3],
                            np.array(states)[:, 4],
                            np.array(states)[:, 5],
                            marker='v',
                            label='states 3-5'
                        )
                        if shaping_object.shaping_mode is not None:
                            axis.plot(
                                np.array(achieved_goals)[:, 3],
                                np.array(achieved_goals)[:, 4],
                                np.array(achieved_goals)[:, 5],
                                label='achieved goals 3-5',
                            )
                            axis.plot(
                                np.array(desired_goals).reshape(
                                    (-1, env.plan_length, env.dim_plan))[-1, :, 3],
                                np.array(desired_goals).reshape(
                                    (-1, env.plan_length, env.dim_plan))[-1, :, 4],
                                np.array(desired_goals).reshape(
                                    (-1, env.plan_length, env.dim_plan))[-1, :, 5],
                                label='latest plan 3-5'
                            )
                        else:
                            axis.plot(
                                np.array(achieved_goals)[:, 0],
                                np.array(achieved_goals)[:, 1],
                                len(achieved_goals)*[0],
                                label='achieved goals 0-1',
                                marker='v'
                            )
                            axis.plot(
                                np.array(desired_goals)[:, 0],
                                np.array(desired_goals)[:, 1],
                                len(desired_goals)*[0],
                                marker='*',
                                label='desired goals 0-1'
                            )
                        axis.legend()
                        plt.show()

                    assert len(np.array(states).shape) == 2
                    assert np.array(states).shape[-1] == 10

                    for desired_goal in desired_goals:
                        assert np.all(desired_goals[0] == desired_goal)

                    if shaping_object.shaping_mode is not None:
                        assert len(np.array(achieved_goals).shape) == 2
                        assert np.array(achieved_goals).shape[-1] == 6
                        assert len(np.array(desired_goals).shape) == 2
                        assert np.array(desired_goals).shape[-1] == 50*6

                        assert np.all(np.array(states)[
                            :, :6] == np.array(achieved_goals))
                    else:
                        assert len(np.array(achieved_goals).shape) == 2
                        assert np.array(achieved_goals).shape[-1] == 2
                        assert len(np.array(desired_goals).shape) == 2
                        assert np.array(desired_goals).shape[-1] == 2

                        assert np.all(np.array(states)[
                            :, 3:5] == np.array(achieved_goals))

                    if shaping_object.shaping_mode == "potential_based":
                        previous_achieved_goals = np.array(achieved_goals)[:-1]
                    else:
                        previous_achieved_goals = None

                    computed_rewards = env.compute_reward(
                        np.array(achieved_goals)[1:],
                        np.array(desired_goals)[1:],
                        None,
                        previous_achieved_goal=previous_achieved_goals
                    )

                    if view and (timestep % 10 == 0 or duration-timestep < 3):
                        plt.plot(rewards, marker='1', markersize=20)
                        plt.plot(computed_rewards, marker='2', markersize=20)
                        plt.plot(expected_reward)
                        plt.legend([
                            'Collected rewards',
                            'Computed Rewards',
                            "Appr. Expected Rewards"
                        ])
                        plt.show()

                    assert len(np.array(rewards).shape) == 1
                    assert len(computed_rewards.shape) == 1
                    assert computed_rewards.shape[0] == np.array(
                        rewards).shape[0]
                    assert np.all(computed_rewards == np.array(rewards))

            assert len(np.array(expected_reward).shape) == 1
            assert computed_rewards.shape[0] == np.array(
                expected_reward).shape[0]
            assert np.all(
                np.abs(
                    (np.array(expected_reward) - np.array(rewards))
                ) < 5e-2
            )


def test_stable_baselines_her():
    """
    Test the gym API by running the stable_baselines3 HER implementation
    https://github.com/DLR-RM/stable-baselines3 as reference.
    This test does not check for performance.
    """
    for model_class in [DDPG, SAC, TD3]:
        # Create env without shaping
        env = gym.make('gym_physx:physx-pushing-v0')

        # The environment does not have a time limit itself, but
        # this can be provided using the TimeLimit wrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=500)

        model = HER(
            'MlpPolicy',
            env,
            model_class,
            verbose=1,
            device='cpu'
        )

        model.learn(2100)


def test_simulation(n_trials=5, view=False):
    """
    Test if the sequence of actions defined below
    indeed reaches the goal, and whether the rewards are
    as expected for all 3 shaping options.
    Parts of this is redundant with test_observations(), but
    redundancy does not hurt when testing.
    """
    shaping_objects = [
        PlanBasedShaping(shaping_mode=strategy, gamma=gamma)
        for strategy, gamma in zip(
            [None, 'relaxed', 'potential_based'],
            [None, None, 0.9]
        )
    ]

    with open(os.path.join(
            os.path.dirname(__file__),
            'expected_rewards.json'
    ), 'r') as data:
        expected_rewards = json.load(data)["expected_rewards"]

    expected_success = expected_rewards[0]
    for shaping_object, expected_reward in zip(
            shaping_objects, expected_rewards
    ):
        env = gym.make(
            'gym_physx:physx-pushing-v0',
            plan_based_shaping=shaping_object
        )
        if view:
            view = env.render()

        for _ in range(n_trials):
            rewards = []
            successes = []
            env._controlled_reset(  # pylint: disable=protected-access
                [-0.3, 0],
                [-0.6, -0.6],
                [0.6, 0.6]
            )
            env.config.frame('target').setContact(0)
            actions = [
                [-0.05, 0, 0],
                [0, -0.05, 0],
                [0.05, 0, 0],
                [0, -0.05, 0],
                [0.05, 0, 0],
                [0, 0.05, 0],
            ]
            durations = [15, 13, 25, 5, 5, 26]

            assert len(actions) == len(durations)
            for action, duration in zip(actions, durations):
                for _ in range(duration):
                    _, reward, _, info = env.step(action)
                    if view:
                        time.sleep(0.02)
                        print(f'reward={reward}')
                    rewards.append(reward)
                    successes.append(info["is_success"])
            assert np.all(
                np.abs(
                    (np.array(expected_reward) - np.array(rewards))
                ) < 5e-2
            )

            assert np.all(
                np.array(successes).astype(float) == np.array(expected_success)
            )


def test_friction(view=False):
    """
    Test the effects of friction if angle of attack is not
    aligned with the center of mass
    """
    env = gym.make('gym_physx:physx-pushing-v0')
    if view:
        view = env.render()

    successes = []
    for _ in range(20):
        for reset_pos, expected in zip(
                [
                    [0.5, 0.],
                    [0.5, 0.1],
                    [0.5, -0.1]
                ],
                [
                    [
                        -0.5, 0., 0.14, -0.75723034, -0.00260414,
                        0.64451677, 0.9986539, 0.00936635, -0.00884138, 0.05024423
                    ],
                    [
                        -0.5, 0.1, 0.14, -0.68895733, -0.07530067,
                        0.64443678, 0.94599276, 0.01166595, -0.0064683, 0.32391319
                    ],
                    [
                        -0.5, -0.1, 0.14, -0.66184765, 0.10549879,
                        0.64465803, 0.94096258, 0.00520798, -0.0121258, -0.33825325
                    ]
                ]
        ):
            if view:
                time.sleep(2)
            env._controlled_reset(  # pylint: disable=protected-access
                reset_pos,
                [0., 0.],
                [-0.6, -0.6]
            )
            action = [-0.05, 0., 0.]
            for _ in range(20):
                observation, _, _, _ = env.step(action)
                if view:
                    time.sleep(0.02)
            print(observation['observation'])
            successes.append(np.linalg.norm(
                observation['observation']-expected) < 1e-8)
    assert np.all(successes)


def test_reset():
    """
    Make sure that after a random reset, box and finger are
    never in collision
    """
    env = gym.make('gym_physx:physx-pushing-v0')
    for _ in range(5000):
        # reset to random finger, box, and target pos
        env.reset()
        # allowed states are in at least one of the planar
        # coordinates further away from each other than $MIN_DIST
        assert any(
            np.abs(
                env.config.frame(
                    "finger"
                ).getPosition()[:2] - env.config.frame(
                    "box"
                ).getPosition()[:2]
            ) > 0.21
        )


@pytest.mark.parametrize("n_trials", [50])
@pytest.mark.parametrize("komo_plans", [False, True])
def test_planning_module(n_trials, komo_plans):
    # MAKE SURE BOX IS NEVER PENETRATED
    """
    Test whether the planning module returns feasible and dense
    plans with acceptable costs
    """
    env = gym.make(
        'gym_physx:physx-pushing-v0',
        # using relaxed reward shaping only to enforce that the
        # environment plans automatically
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed'),
        komo_plans=komo_plans
    )

    height_offset = env.config.frame(
        "finger"
    ).getPosition()[2] - env.config.getJointState()[2]

    acceptable_costs_count = 0
    for _ in range(n_trials):
        observation = env.reset()
        plan = observation["desired_goal"]

        # Assert that the observation is included in observation space
        assert env.observation_space.contains(observation)
        # Should be already included in the assertion above
        assert env.observation_space["desired_goal"].contains(plan)

        # reshape plan into [time, dims]
        plan = plan.reshape(env.plan_length, env.dim_plan)

        # Make sure every line of the plan is included in achieved_goal space
        for achieved_goal in plan:
            assert env.observation_space["achieved_goal"].contains(
                achieved_goal)

        # ensure acceptable costs
        costs = env.komo.getConstraintViolations() if komo_plans else 0
        acceptable_costs_count += int(costs < 50)

        # ensure that initial state of the plan is consistent with env state
        assert np.all(np.abs(
            observation["observation"][:6] - plan[0]
        ) < env.plan_max_stepwidth * 2)
        # ensure that the initial state of the plan is consistent with achieved_goal
        assert np.all(np.abs(
            observation["achieved_goal"][:6] - plan[0]
        ) < env.plan_max_stepwidth * 2)
        # ensure initial plan state consistent with internal joint state...
        assert np.all(np.abs(
            env.config.getJointState()[:3] - plan[0, :3]
        ) < env.plan_max_stepwidth * 2)
        # ...and consistent with internal box state
        assert np.all(np.abs(
            env.config.frame('box').getPosition() - plan[0, 3:]
        ) < env.plan_max_stepwidth * 2)
        # ensure that final state of plan reaches goal
        assert np.all(np.abs(
            env.config.frame('target').getPosition() - plan[-1, 3:]
        ) < env.plan_max_stepwidth * 2)

        # enusure (again) that planned finger positions are within the env's limits
        assert all(np.abs(
            plan[:, :2]
        ).flatten() <= env.maximum_xy_for_finger)
        assert np.all(
            plan[:, 2] >= env.minimum_rel_z_for_finger -
            env.plan_max_stepwidth/2
        )
        assert np.all(
            plan[:, 2] <= env.maximum_rel_z_for_finger +
            env.plan_max_stepwidth/2
        )

        # ensure that finger is never "inside" box
        for state in plan:
            assert (
                (
                    # either the finger has to be outside the box...
                    # (only take inner disk of radius 0.2 here for simplicity)
                    np.linalg.norm(
                        state[:2] - state[3:5]
                    ) > 0.18  # account for the box's border radius of 0.05
                ) or (
                    # or the finger's z coordinate is above the box
                    (state[2] + height_offset) - state[5] > 0.1 + 0.06
                    # account for height offset between joint and config coords
                )
            )

        # ensure sufficient plan density and a smooth trajectory
        assert all(
            np.linalg.norm(
                plan[1:] - plan[:-1],
                axis=-1
            ) <= 1.2 * np.sqrt(2)*env.plan_max_stepwidth*150/env.plan_length
        )

    # A certain amount of plans have to have acceptable cost
    assert acceptable_costs_count/n_trials >= 48/50

with open(
        os.path.join(os.path.dirname(__file__), 'fixed_reset.json'),
        'r'
) as infile:
    fixed_reset_data = json.load(infile)

@pytest.mark.parametrize("n_episodes", [5])
@pytest.mark.parametrize(
    "shaping_object",
    [
        PlanBasedShaping(shaping_mode=strategy, gamma=gamma)
        for strategy, gamma in zip(
            [None, 'relaxed', 'potential_based'],
            [None, None, 0.9]
        )
    ]
)
@pytest.mark.parametrize(
    "fixed_initial_config",
    [
        None,
        {
            'finger_position': [-0.8, -0.1],
            'box_position': [-0.5, 0.],
            'goal_position': [0.5, 0.]
        },
        {
            'finger_position': [-0.8, -0.1],
            'box_position': [-0.5, 0.],
            'goal_position': [0.5, 0.],
            'static_plan': np.array(fixed_reset_data['reference_plan'])
        },
    ]
)
def test_fixed_initial_config(n_episodes, shaping_object, fixed_initial_config):
    """
    Test setting in which the environment is reset to the same config
    (i.e. same finger+box position and same goal) after each reset
    """
    assert shaping_object.shaping_mode in [None, 'relaxed', 'potential_based']
    env = gym.make(
        'gym_physx:physx-pushing-v0',
        plan_based_shaping=shaping_object,
        fixed_initial_config=fixed_initial_config
    )

    if fixed_initial_config is None:
        assert not isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.observation_space, gym.spaces.Dict)
    else:
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert not isinstance(env.observation_space, gym.spaces.Dict)

    for _ in range(n_episodes):
        obs = env.reset()
        collected_rewards = []
        desired_goals = []
        achieved_goals = []

        for __ in range(21):
            assert env.observation_space.contains(obs)

            if fixed_initial_config is None:
                assert obs['observation'].shape == (10,)
                if shaping_object.shaping_mode is not None:
                    assert obs['achieved_goal'].shape == (6,)
                    assert obs['desired_goal'].shape == (300,)
                else:
                    assert obs['achieved_goal'].shape == (2,)
                    assert obs['desired_goal'].shape == (2,)
            else:
                assert obs.shape == (10,)
                if shaping_object.shaping_mode is None:
                    assert env.current_desired_goal.shape == (2,)
                    assert np.all(env.current_desired_goal ==
                                  fixed_initial_config["goal_position"])
                else:
                    assert env.current_desired_goal.shape == (300,)
                    assert np.all(env.current_desired_goal == env.static_plan)
                    if 'static_plan' in fixed_initial_config:
                        # In this case env.static_plan has to be strictly equal to the reference
                        assert np.all(env.static_plan ==
                                      fixed_initial_config['static_plan'])
                        assert np.all(env.static_plan == np.array(
                            fixed_reset_data['reference_plan']))
                        assert np.all(env.current_desired_goal == np.array(
                            fixed_reset_data['reference_plan']))
                    else:
                        # In this case the equality only is approximate
                        # (limited by the accuracy of the planner)
                        assert np.mean(
                            np.abs(env.static_plan -
                                   np.array(fixed_reset_data['reference_plan']))
                        ) < 5e-3

            obs, reward, _, _ = env.step([0.05, 0, 0])

            collected_rewards.append(reward)
            desired_goals.append(env.current_desired_goal)
            achieved_goals.append(env.current_achieved_goal)

        reference_rewards = np.array(
            fixed_reset_data[str(shaping_object.shaping_mode)])
        if shaping_object.shaping_mode == 'potential_based':
            reference_rewards = reference_rewards[1:]
            collected_rewards = collected_rewards[1:]
            computed_rewards = env.compute_reward(
                np.array(achieved_goals)[1:],
                np.array(desired_goals)[1:],
                None,
                previous_achieved_goal=np.array(achieved_goals)[:-1]
            )
        else:
            computed_rewards = env.compute_reward(
                np.array(achieved_goals),
                np.array(desired_goals),
                None
            )

        # Computed rewards have to be strictly consistent
        assert np.all(np.array(collected_rewards) == computed_rewards)

        if fixed_initial_config is not None:
            # Reference rewards have to be
            if 'static_plan' in fixed_initial_config:
                # ..striclty consistent if the reference plan was used
                if shaping_object.shaping_mode is not None:
                    assert np.all(
                        env.current_desired_goal == np.array(
                            fixed_initial_config['static_plan'])
                    )
                assert np.all(np.array(collected_rewards) == reference_rewards)

            # ...appr. consistent if the plan was re-computed
            assert np.mean(
                np.abs(np.array(collected_rewards) - reference_rewards)
            ) < 5e-3

@pytest.mark.parametrize("n_episodes", [10])
@pytest.mark.parametrize(
    "shaping_object",
    [PlanBasedShaping(shaping_mode=None, gamma=None)]
)
@pytest.mark.parametrize("fixed_finger_initial_position", [True, False])
def test_fixed_initial_finger_position(n_episodes, shaping_object, fixed_finger_initial_position):
    assert shaping_object.shaping_mode in [None, 'relaxed', 'potential_based']
    env = gym.make(
        'gym_physx:physx-pushing-v0',
        plan_based_shaping=shaping_object,
        fixed_initial_config=None,
        fixed_finger_initial_position=fixed_finger_initial_position
    )

    finger_positions = []
    for _ in range(n_episodes):
        obs = env.reset()
        finger_positions.append(obs['observation'][:2])

    finger_positions = np.array(finger_positions)
    assert finger_positions.shape == (n_episodes, 2)
    if fixed_finger_initial_position:
        assert np.max(finger_positions) == 0
    else:
        assert not np.max(finger_positions) == 0


# %%
