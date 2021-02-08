# %%
"""
Tests for the BoxEnv
"""

import os
import json
import time
import numpy as np
import gym
from gym_physx.envs.shaping import PlanBasedShaping
from plan_rl.DDPG.ddpg import ddpg


# TODO implement test
def test_gym_api():
    """
    Test the gym API by running the HER implementation at
    https://github.com/TianhongDai/hindsight-experience-replay
    as reference. This test does not check for performance.
    """
    raise NotImplementedError


def test_simulation(n_trials=20, view=False):
    """
    Test if the sequence of actions defined below
    indeed reaches the goal, and whether the rewards are
    as expected in all 3 scenarios
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
                ) < 1e-1
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


def test_planning_module():
    """
    Test whether the planning module returns feasible and dense
    plans with acceptable costs
    """
    env = gym.make(
        'gym_physx:physx-pushing-v0',
        # using relaxed reward shaping only to enforce that the
        # environment plans automatically
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed')
    )

    height_offset = env.config.frame(
        "finger"
    ).getPosition()[2] - env.config.getJointState()[2]

    for _ in range(50):
        observation = env.reset()
        plan = observation["current_plan"]

        # ensure acceptable costs
        assert env.komo.getConstraintViolations() < 50

        # ensure that initial state of the plan is consistent with env state
        assert np.all(np.abs(
            observation["observation"][:6] - plan[0]
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

        # enusure that planned finger positions are within the env's limits
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
            ) <= np.sqrt(2)*env.plan_max_stepwidth
        )

# %%
