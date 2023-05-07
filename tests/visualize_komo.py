# %%
"""
Test of the BoxEnv() environment with
KOMO planning
"""
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gym

from gym_physx.envs.shaping import PlanBasedShaping
sys.path.append(os.getenv("HOME") + '/git/rai-python/rai/rai/ry')
if os.getenv("HOME") + '/git/rai-python/rai/rai/ry' in sys.path:
    import libry as ry   # pylint: disable=import-error

ENV = gym.make(
    'gym_physx:physx-pushing-v0',
    # using relaxed reward shaping only to enforce that the
    # environment plans automatically
    plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed'),
    komo_plans=False
)
VIEW = ENV.render()

# %%
def show_plan(plan_in):
    """
    show KOMO plan
    """
    show_env = gym.make(
        'gym_physx:physx-pushing-v0',
        # using relaxed reward shaping only to enforce that the
        # environment plans automatically
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed'),
        komo_plans=False
    )
    show_env.config.setJointState(
        ENV.config.getJointState()
    )
    show_env.config.frame('target').setPosition(
        ENV.config.frame('target').getPosition()
    )
    show_env.config.frame('box').setPosition(
        ENV.config.frame('box').getPosition()
    )
    show_env.config.frame('box').setQuaternion(
        ENV.config.frame('box').getQuaternion()
    )

    for ind, pos in enumerate(plan_in):
        show_env.config.addFrame(str(ind))
        show_env.config.frame(str(ind)).setShape(
            ry.ST.sphere,
            size=ENV.config.frame('finger').info()['size']
        )
        show_env.config.frame(str(ind)).setColor(
            np.append(
                (len(plan_in)-ind)/len(plan_in)*np.array([1., 1., 1.]),
                1.
            )
        )
        show_env.config.frame(
            str(ind)).setPosition(pos[:3] + show_env.config.frame('floor').getPosition())

        # show_env.config.addFrame(str(ind) + 'box')
        # show_env.config.frame(str(ind) + 'box').setShape(
        #     ry.ST.box,
        #     size=ENV.config.frame('box').info()['size']
        # )
        # show_env.config.frame(str(ind) + 'box').setColor(
        #     np.append(
        #         (len(plan_in)-ind)/len(plan_in)*np.array([1., 1., 1.]),
        #         1.
        #     )
        # )
        # show_env.config.frame(
        #     str(ind) + 'box').setPosition(pos[3:])

    show_env.config.addFrame('finger_new_pos')
    show_env.config.frame('finger_new_pos').setShape(ry.ST.sphere, size=[0.12])
    show_env.config.frame('finger_new_pos').setColor(
        [1., 0., 0., 0.]
    )
    show_view = show_env.render()
    return show_view


# %%
violations = []
for _ in range(10):
    obs = ENV.reset()

    plan = obs['desired_goal'].reshape(
        ENV.plan_length, ENV.dim_plan)

    if ENV.komo_plans:
        ENV.komo.displayTrajectory()
        print(f'KOMO violations: {ENV.komo.getConstraintViolations()}')
        violations.append(ENV.komo.getConstraintViolations())

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(plan[:, 0], plan[:, 1], plan[:, 2], label='finger')
    ax.plot(plan[:, 3], plan[:, 4], plan[:, 5], label='box')
    plt.legend()
    plt.show()

    plt.plot(plan[:, 0], plan[:, 1])
    plt.show()
    plan_view = show_plan(plan)

    if ENV.komo_plans:
        print(ENV.komo.getReport()[-2:])

# %%
plan_lengths = [50, 150, 200]
envs = [
    gym.make(
        'gym_physx:physx-pushing-v0',
        # using relaxed reward shaping only to enforce that the
        # environment plans automatically
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed'),
        plan_length=plan_length
    )
    for plan_length in plan_lengths
]

for _ in range(20):
    finger_position = envs[0]._sample_finger_pos()  # pylint: disable=protected-access
    for __ in range(1000):
        box_position = envs[0]._sample_box_position()  # pylint: disable=protected-access
        if envs[0]._box_finger_not_colliding(  # pylint: disable=protected-access
                finger_position,
                box_position
        ):
            break
    goal_position = envs[0]._sample_box_position()  # pylint: disable=protected-access

    for env, plan_length in zip(envs, plan_lengths):
        plan = env.controlled_reset(  # pylint: disable=protected-access
            finger_position,
            box_position,
            goal_position
        )['desired_goal'].reshape(env.plan_length, env.dim_plan)

        plt.plot(
            plan[:, 0], plan[:, 1]
        )
    plt.legend(plan_lengths)
    plt.show()

# %%
env = gym.make(
    'gym_physx:physx-pushing-v0',
    # using relaxed reward shaping only to enforce that the
    # environment plans automatically
    plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed'),
    komo_plans=False,
    plan_length=550,
    n_keyframes=10
)

for _ in range(20):
    finger_position = env._sample_finger_pos()  # pylint: disable=protected-access
    for __ in range(1000):
        box_position = env._sample_box_position()  # pylint: disable=protected-access
        if env._box_finger_not_colliding(  # pylint: disable=protected-access
                finger_position,
                box_position
        ):
            break
    goal_position = env._sample_box_position()  # pylint: disable=protected-access

    for _ in range(3):
        plan = env.controlled_reset(  # pylint: disable=protected-access
            finger_position,
            box_position,
            goal_position
        )['desired_goal'].reshape(env.plan_length, env.dim_plan)

        view = show_plan(plan)
        time.sleep(60)

# %%
