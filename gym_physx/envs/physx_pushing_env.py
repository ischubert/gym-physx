"""
PhysX-based Robotic Pushing Environment
"""
import sys
import os
import json
import time
import numpy as np
import gym

from .shaping import PlanBasedShaping

# TODO Compile rai as static lib
sys.path.append(os.getenv("HOME") + '/git/rai-python/rai/rai/ry')
if os.getenv("HOME") + '/git/rai-python/rai/rai/ry' in sys.path:
    import libry as ry  # pylint: disable=import-error


class PhysxPushingEnv(gym.Env):
    """
    PhysX-based Robotic Pushing Environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            plan_based_shaping=PlanBasedShaping(),
            max_action=0.1,
            action_duration=0.5,
            tau=.01,
            target_tolerance=0.1,
            plan_max_stepwidth=0.05,
            densify_plans=True,
            config_file=None,
            fps=None
    ):
        self.plan_based_shaping = plan_based_shaping
        self.max_action = max_action
        self.action_duration = action_duration
        self.tau = tau
        self.target_tolerance = target_tolerance
        self.plan_max_stepwidth = plan_max_stepwidth
        self.densify_plans = densify_plans
        self.config_file = config_file
        self.fps = fps

        self.n_steps = int(self.action_duration/self.tau)
        self.proportion_per_step = 1/self.n_steps

        self.skeleton = None
        self.komo = None
        self.current_goal = None
        self.current_plan = None
        self.previous_state = None

        self.config_file_default = os.path.join(
            os.path.dirname(__file__), 'config_data/pushing.g'
        )

        with open(os.path.join(
                os.path.dirname(__file__),
                'config_data/pushing.json'
        ), 'r') as config_data:
            json_config = json.load(config_data)

        self.floor_level = json_config["floor_level"]
        self.finger_xy_min = json_config["finger_xy_min"]
        self.finger_xy_max = json_config["finger_xy_max"]
        self.box_xy_min = json_config["box_xy_min"]
        self.box_xy_max = json_config["box_xy_max"]
        self.collision_distance = json_config["collision_distance"]
        self.finger_relative_level = json_config["finger_relative_level"]
        self.maximum_xy_for_finger = json_config["maximum_xy_for_finger"]
        self.minimum_rel_z_for_finger = json_config["minimum_rel_z_for_finger"]
        self.maximum_rel_z_for_finger = json_config["maximum_rel_z_for_finger"]
        self.subspace_for_shaping = json_config["subspace_for_shaping"]
        self.floor_color = np.array(json_config["floor_color"])
        self.finger_color = np.array(json_config["finger_color"])
        self.box_color = np.array(json_config["box_color"])
        self.target_color = np.array(json_config["target_color"])

        self.config = self._create_config()
        self.simulation = self.config.simulation(
            ry.SimulatorEngine.physx, False)
        self.config.setJointState(json_config["initial_joint_state"])

        self.finger_radius = self.config.frame('finger').info()['size'][0]
        self.minimum_rel_z_for_finger_in_config_coords = self.minimum_rel_z_for_finger + \
            self.config.frame('floor').getPosition()[2]
        self.maximum_rel_z_for_finger_in_config_coords = self.maximum_rel_z_for_finger + \
            self.config.frame('floor').getPosition()[2]

        self.reset()


    def step(self, action):
        """
        Simulate the system's transition under an action
        """
        self.previous_state = self._get_state()
        # clip action
        action = np.clip(
            action,
            -self.max_action,
            self.max_action
        )

        # gradual pushing movement
        joint_q = self.config.getJointState()
        for _ in range(self.n_steps):
            new_x = joint_q[0] + self.proportion_per_step * action[0]
            if abs(new_x) < self.maximum_xy_for_finger:
                joint_q[0] = new_x

            new_y = joint_q[1] + self.proportion_per_step * action[1]
            if abs(new_y) < self.maximum_xy_for_finger:
                joint_q[1] = new_y

            new_z = joint_q[2] + self.proportion_per_step * action[2]
            if new_z < self.maximum_rel_z_for_finger and new_z > self.minimum_rel_z_for_finger:
                joint_q[2] = new_z

            self.config.setJointState(joint_q)
            self.simulation.step(u_control=[0, 0, 0, 0, 0, 0, 0], tau=self.tau)
            if self.fps is not None:
                time.sleep(1/self.fps)

        observation = self._get_observation()
        reward = self._calculate_reward()
        done = False
        info = {}

        return observation, reward, done, info

    def reset(
            self,
    ):
        """
        Reset the environment randomly
        """
        finger_position = self._sample_finger_pos()
        for _ in range(1000):
            box_position = self._sample_box_position()
            if self._box_finger_not_colliding(
                    finger_position,
                    box_position
            ):
                break
        goal_position = self._sample_box_position()

        return self._controlled_reset(
            finger_position,
            box_position,
            goal_position
        )

    def render(self, mode='human'):
        """
        Create interactive view of the environment
        """
        return self.config.view()

    def close(self):
        raise NotImplementedError

    def _get_approximate_plan(self):
        """
        Uses rai/KOMO to calculate plan using the current
        state and target. This plan is based on the differentiable
        physics model rai uses, and can not be directly executed
        in the physx simulation.
        """

        plan = []

        # create copy of of self.config
        planner_initial_config = self._create_config()
        self._refresh_target(planner_initial_config)
        for frame_name in self.config.getFrameNames():
            planner_initial_config.frame(frame_name).setPosition(
                self.config.frame(frame_name).getPosition()
            )
            planner_initial_config.frame(frame_name).setQuaternion(
                self.config.frame(frame_name).getQuaternion()
            )

        # decide whether hard-coded waypoint is needed
        target_pos = planner_initial_config.frame(
            'target'
        ).getPosition()
        box_pos = planner_initial_config.frame(
            'box'
        ).getPosition()
        finger_pos = planner_initial_config.frame(
            'finger'
        ).getPosition()
        target_box_diff = target_pos-box_pos
        if np.dot(
                target_box_diff,
                box_pos-finger_pos
        ) <= 0:
            # hardcode first part of movement: define finger waypoints
            wp_1 = finger_pos
            wp_2 = finger_pos + np.array([0, 0, 0.4])
            for exp in range(10):
                wp_3 = box_pos - 0.8**exp * 0.7 * \
                    (target_box_diff/np.linalg.norm(target_box_diff))
                if all(np.abs(wp_3[:2]) < self.maximum_xy_for_finger):
                    break
            wp_3[2] = wp_2[2]

            waypoints = [wp_1, wp_2, wp_3]

            for current_wp, next_wp in zip(waypoints, waypoints[1:]):
                n_steps = int(np.linalg.norm(
                    next_wp-current_wp)/self.plan_max_stepwidth)
                unit_vector = (next_wp-current_wp) / \
                    np.linalg.norm(next_wp-current_wp)
                for ind in range(n_steps):
                    plan.append([
                        *(current_wp + ind*self.plan_max_stepwidth*unit_vector),
                        *box_pos
                    ])

            # second part (pushing) is done by KOMO:
            # set last waypoint as starting point for KOMO
            planner_initial_config.frame('finger').setPosition(waypoints[-1])

        # approximately calculate how much time should be spent moving
        # without contact to the box and with contact to the box
        box_target_dist = np.linalg.norm(target_box_diff)
        finger_box_dist = np.linalg.norm(
            box_pos - planner_initial_config.frame('finger').getPosition()
        )
        no_contact_to_contact_ratio = (
            finger_box_dist-0.2  # 0.2 is half the width of the box
        )/(
            box_target_dist+finger_box_dist-0.2
        )

        # approximately calculate the total number of time steps needed
        num_steps = int(
            (box_target_dist+finger_box_dist-0.2)/self.plan_max_stepwidth
        )
        # print(f'no_contact_to_contact_ratio {no_contact_to_contact_ratio}')
        # print(f'num_steps {num_steps}')

        # plan from the current position or (if applicable) from the last waypoint
        self.skeleton = [
            # makes the finger free
            [0., 1.], ry.SY.magic, ['finger'],
            [0., 1.], ry.SY.dampMotion, ['finger'],
            # the following skeleton symbols introduce POAs and force vectors as
            # decision variables. For more information, see
            # https://ipvs.informatik.uni-stuttgart.de/mlr/papers/20-toussaint-RAL.pdf
            [no_contact_to_contact_ratio, 1.1], ry.SY.quasiStaticOn, ["box"],
            [no_contact_to_contact_ratio, 1.], ry.SY.contact, ["finger", "box"]
        ]
        self.komo = planner_initial_config.komo_path(
            phases=1.,
            stepsPerPhase=num_steps,
            timePerPhase=1.,
            # k_order=2,
            useSwift=False  # useSwift=True ()=calling collision detection)
        )
        self.komo.addSquaredQuaternionNorms()
        self.komo.addSkeleton(self.skeleton)
        # 1. objective: box should be at target at the end
        self.komo.addObjective(
            time=[1.], feature=ry.FS.positionDiff, frames=["box", "target"],
            type=ry.OT.eq, scale=[1e2], order=0
        )
        # 2, objective: velocity of everything should be 0 at the end
        self.komo.addObjective(
            time=[1.], feature=ry.FS.qItself, frames=[],  # [] means all frames
            type=ry.OT.sos, scale=[1e0], order=1
        )
        # 3. objective: minimum z coord  of finger
        self.komo.addObjective(
            # [] means all frames
            time=[0., 1.], feature=ry.FS.position, frames=["finger"],
            type=ry.OT.ineq, scaleTrans=[[0., 0., -1.]], target=[
                0., 0.,
                self.minimum_rel_z_for_finger_in_config_coords
            ], order=0
        )
        # 4. objective: maximum z coord  of finger
        self.komo.addObjective(
            # [] means all frames
            time=[0., 1.], feature=ry.FS.position, frames=["finger"],
            type=ry.OT.ineq, scaleTrans=[[0., 0., 1.]], target=[
                0., 0.,
                self.maximum_rel_z_for_finger_in_config_coords
            ], order=0
        )
        self.komo.setupConfigurations()
        self.komo.optimize()

        plan = np.concatenate(
            (
                np.array(plan).reshape(-1, self.subspace_for_shaping),
                self.komo.getPathFrames(['finger', 'box'])[
                    :, :, :3].reshape(-1, self.subspace_for_shaping)
            ),
            axis=0
        )

        if self.densify_plans:
            for _ in range(10):
                step_width_too_large = np.linalg.norm(
                    plan[1:] - plan[:-1], axis=-1
                ) > np.sqrt(2)*self.plan_max_stepwidth
                if any(step_width_too_large):
                    plan = np.insert(
                        plan,
                        np.where(step_width_too_large)[0] + 1,
                        0.5*(plan[1:] + plan[:-1])[step_width_too_large],
                        axis=0
                    )
                else:
                    break
        plan[:, 2] = plan[:, 2]-self.config.frame('floor').getPosition()[2]
        return plan

    def _get_observation(self):
        """
        Returns current observation. Desired goal is the current plan,
        since it is the goal of the policy to follow the current plan
        """
        return {
            'observation': self._get_state(),
            'achieved_goal': self.config.frame('box').getPosition().copy(),
            'desired_goal': self.current_goal.copy(),
            'current_plan': None if self.current_plan is None else self.current_plan.copy()
        }

    def _calculate_reward(self):
        """
        Calculate reward (shaped or unshaped) for the last action
        """
        binary_reward = float(np.linalg.norm(
            self.config.frame(
                'box'
            ).getPosition() - self.config.frame(
                'target'
            ).getPosition()
        ) < self.target_tolerance)

        # Previous state is only considered in potential_based mode
        previous_state = None
        if self.plan_based_shaping.shaping_mode=='potential_based':
            previous_state = self.previous_state[:self.subspace_for_shaping]

        return self.plan_based_shaping.shaped_reward_function(
            self._get_state()[:self.subspace_for_shaping],
            binary_reward,
            self.current_plan,
            previous_state=previous_state
        )

    def _get_state(self):
        """
        Get the current state, i.e. position of the finger as well
        as the position and Quaternion of the box
        """
        return np.concatenate([
            self.config.getJointState()[:3],
            self.config.frame('box').getPosition(),
            self.config.frame('box').getQuaternion()
        ])

    def _controlled_reset(
            self,
            finger_position,
            box_position,
            goal_position
    ):
        """
        Reset the environment to specific state
        """
        self.previous_state = None
        self.current_goal = goal_position.copy()

        assert self._box_finger_not_colliding(
            finger_position,
            box_position
        )

        joint_q = np.array([
            *finger_position,
            self.finger_relative_level,
            1., 0., 0., 0.
        ])

        self.config.setJointState(joint_q)
        self.simulation.step(u_control=[0, 0, 0, 0, 0, 0, 0], tau=self.tau)
        self._reset_box(box_position)
        self._refresh_target(self.config)
        self._set_frame_state(
            goal_position,
            "target"
        )

        if self.plan_based_shaping.shaping_mode is not None:
            self.current_plan = self._get_approximate_plan()

        return self._get_observation()

    def _reset_box(self, coords):
        """
        Reset the box to an arbitrary position
        """
        # always reset box to the center
        self._set_frame_state(
            coords,
            'box'
        )
        state_now = self.config.getFrameState()
        self.simulation.setState(state_now, np.zeros((state_now.shape[0], 6)))

    def _set_frame_state(
            self,
            state,
            frame_name
    ):
        """
        Select frame of the configuration by name and set to any state
        """
        self.config.frame(frame_name).setPosition([
            *state[:2],
            self.floor_level
        ])
        self.config.frame(frame_name).setQuaternion(
            [1., 0., 0., 0.]
        )

    def _refresh_target(self, config):
        """
        Reset the target position
        """
        config.delFrame("target")
        config.addFrame(name="target")
        config.frame('target').setShape(
            ry.ST.sphere, size=[self.target_tolerance])
        config.frame('target').setColor(
            self.target_color
        )

    def _create_config(self):
        """
        return new config
        """
        config = ry.Config()
        if self.config_file is not None:
            config.addFile(self.config_file)
        else:
            config.addFile(self.config_file_default)

        config.makeObjectsFree(['finger'])

        config.frame('floor').setColor(self.floor_color)
        config.frame('finger').setColor(self.finger_color)
        config.frame('box').setColor(self.box_color)

        return config

    def _sample_finger_pos(self):
        """
        Sample random position for the finger on the table
        """
        return (
            self.finger_xy_max - self.finger_xy_min
        ) * np.random.rand(2) + self.finger_xy_min

    def _sample_box_position(self):
        """
        Sample random position for the box on the table
        """
        return (
            self.box_xy_max - self.box_xy_min
        ) * np.random.rand(2) + self.box_xy_min

    def _box_finger_not_colliding(
            self,
            finger_position,
            box_position
    ):
        """
        return whether box and finger are in collision
        """
        return any(np.abs(
            np.array(finger_position) - np.array(box_position)
        ) > self.collision_distance)
