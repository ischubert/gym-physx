# %%
"""
Test of the BoxEnv() environment with
KOMO planning
"""
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
    plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed')
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
        plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed')
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

    show_env.config.addFrame('finger_new_pos')
    show_env.config.frame('finger_new_pos').setShape(ry.ST.sphere, size=[0.12])
    show_env.config.frame('finger_new_pos').setColor(
        [1., 0., 0., 0.]
    )
    show_view = show_env.render()
    return show_view


# %%
violations = []
for _ in range(50):
    obs = ENV.reset()

    plan = obs['current_plan']
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

    print(
        ENV.komo.getReport()[-2:]
    )

# %%
