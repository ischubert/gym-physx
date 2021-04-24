"""
PhysX-based Robotic Pushing Environment
"""
import sys
import os
import json
import time
import numpy as np
from scipy.interpolate import interp1d
import gym

from .shaping import PlanBasedShaping

# TODO Compile rai as static lib or add to wheel (?)
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
            action_uncertainty=0.0,
            tau=.01,
            target_tolerance=0.1,
            plan_max_stepwidth=0.05,
            densify_plans=True,
            plan_length=50,
            config_files='pushing',
            n_keyframes=0,
            fixed_initial_config=None,
            fixed_finger_initial_position=False,
            plan_generator=None,
            komo_plans=True,
            fps=None,
            config_file=None
    ):
        self.plan_based_shaping = plan_based_shaping
        self.max_action = max_action
        self.action_duration = action_duration
        self.action_uncertainty = action_uncertainty
        self.tau = tau
        self.target_tolerance = target_tolerance
        self.plan_max_stepwidth = plan_max_stepwidth
        self.densify_plans = densify_plans
        self.plan_length = plan_length
        self.n_keyframes = n_keyframes
        self.fixed_initial_config = fixed_initial_config
        self.fixed_finger_initial_position = fixed_finger_initial_position
        self.plan_generator = plan_generator
        self.komo_plans = komo_plans
        self.fps = fps
        self.config_file = config_file

        self.n_steps = int(self.action_duration/self.tau)
        self.proportion_per_step = 1/self.n_steps

        self.skeleton = None
        self.komo = None
        self.current_desired_goal = None
        self.current_achieved_goal = None
        self.previous_achieved_goal = None
        self.static_plan = None

        self.config_file_default = os.path.join(
            os.path.dirname(__file__), 'config_data/' + config_files + '.g'
        )

        if self.fixed_initial_config is not None:
            assert not self.fixed_finger_initial_position, "Both fixed_initial_config and fixed_finger_initial_position were given"
            for key in ['finger_position', 'box_position', 'goal_position']:
                assert key in self.fixed_initial_config, f"fixed_initial_config was set but {key} is missing"
            if "static_plan" in self.fixed_initial_config:
                print("Fixed initial config: Using given static plan")
            else:
                print("Fixed initial config: Automatically create static plan")

        # Read in config file
        with open(os.path.join(
                os.path.dirname(__file__),
                'config_data/' + config_files + '.json'
        ), 'r') as config_data:
            json_config = json.load(config_data)
        # general dimensions
        self.floor_level = json_config["floor_level"]
        self.finger_relative_level = json_config["finger_relative_level"]
        self.collision_distance = json_config["collision_distance"]
        # reset configuration
        self.reset_finger_xy_min = json_config["reset_finger_xy_min"]
        self.reset_finger_xy_max = json_config["reset_finger_xy_max"]
        self.reset_box_xy_min = json_config["reset_box_xy_min"]
        self.reset_box_xy_max = json_config["reset_box_xy_max"]
        # box boundaries
        self.box_xy_min = json_config["box_xy_min"]
        self.box_xy_max = json_config["box_xy_max"]
        self.maximum_xy_for_finger = json_config["maximum_xy_for_finger"]
        self.minimum_rel_z_for_finger = json_config["minimum_rel_z_for_finger"]
        self.maximum_rel_z_for_finger = json_config["maximum_rel_z_for_finger"]
        # plan dimensionality
        self.dim_plan = json_config["dim_plan"]
        self.plan_based_shaping.set_plan_len_and_dim(
            plan_len=self.plan_length, plan_dim=self.dim_plan
        )

        # assert sufficient plan density
        assert self.plan_length >= 50* (self.n_keyframes + 1), "Please use higher plan_length"
        # assert consistent plan size if plan_generator is given
        if self.plan_generator is not None:
            assert self.plan_generator.plan_dim == self.dim_plan, "plan_generator: wrong plan_dim"
            assert self.plan_generator.plan_len == self.plan_length, "plan_generator: wrong plan_length"

        # rendering colors
        self.floor_color = np.array(json_config["floor_color"])
        self.finger_color = np.array(json_config["finger_color"])
        self.box_color = np.array(json_config["box_color"])
        self.target_color = np.array(json_config["target_color"])

        # Create rai config
        self.config = self._create_config()
        self.simulation = self.config.simulation(
            ry.SimulatorEngine.physx, False)
        self.config.setJointState(json_config["initial_joint_state"])

        self.finger_radius = self.config.frame('finger').info()['size'][0]
        self.box_xy_size = self.config.frame('box').info()['size'][0]
        self.minimum_rel_z_for_finger_in_config_coords = self.minimum_rel_z_for_finger + \
            self.config.frame('floor').getPosition()[2]
        self.maximum_rel_z_for_finger_in_config_coords = self.maximum_rel_z_for_finger + \
            self.config.frame('floor').getPosition()[2]

        # Define state space
        state_space = gym.spaces.Box(
            low=np.array([
                -self.maximum_xy_for_finger,
                -self.maximum_xy_for_finger,
                self.minimum_rel_z_for_finger,
                self.box_xy_min,
                self.box_xy_min,
                0,
                -1, -1, -1, -1
            ]),
            high=np.array([
                self.maximum_xy_for_finger,
                self.maximum_xy_for_finger,
                self.maximum_rel_z_for_finger,
                self.box_xy_max,
                self.box_xy_max,
                json_config["box_z_max"],
                1, 1, 1, 1
            ]),
        )

        # Define observation space
        if self.plan_based_shaping.shaping_mode is None:
            # Without plan-based shaping, the desired goal
            # is represented by the desired box position.
            # The achieved goal is the observed box position.
            desired_goal_space = gym.spaces.Box(
                low=self.reset_box_xy_min*np.ones(2),
                high=self.reset_box_xy_max*np.ones(2)
            )
            achieved_goal_space = gym.spaces.Box(
                low=self.box_xy_min*np.ones(2),
                high=self.box_xy_max*np.ones(2),
            )

        else:
            # With plan-based shaping, the desired goal
            # is represented by a plan. The plan is the intended
            # 6D trajectory of both finger and box.
            #
            # The achieved goal is the 6D position of box and finger
            #
            # the plans are flattened and the entries are as follows:
            # [
            #   t=0: finger_x, t=0: finger_y, t=0: finger_z,
            #   t=0: box_x, t=0: box_y, t=0: box_z,
            #   t=1: finger_x, t=1: finger_y, t=1: finger_z,
            #   t=1: box_x, t=1: box_y, t=1: box_z,
            #   ...
            #   t=plan_length-1: finger_x, t=plan_length-1: finger_y, t=plan_length-1: finger_z,
            #   t=plan_length-1: box_x, t=plan_length-1: box_y, t=plan_length-1: box_z,
            # ]
            achieved_goal_space_low = [
                -self.maximum_xy_for_finger,
                -self.maximum_xy_for_finger,
                self.minimum_rel_z_for_finger-self.plan_max_stepwidth/2,
                self.box_xy_min,
                self.box_xy_min,
                0
            ]
            achieved_goal_space_high = [
                self.maximum_xy_for_finger,
                self.maximum_xy_for_finger,
                self.maximum_rel_z_for_finger+self.plan_max_stepwidth/2,
                self.box_xy_max,
                self.box_xy_max,
                json_config["box_z_max"]
            ]
            desired_goal_space = gym.spaces.Box(
                low=np.array(self.plan_length * achieved_goal_space_low),
                high=np.array(self.plan_length * achieved_goal_space_high)
            )
            achieved_goal_space = gym.spaces.Box(
                low=np.array(achieved_goal_space_low),
                high=np.array(achieved_goal_space_high)
            )

        if self.fixed_initial_config is None:
            self.observation_space = gym.spaces.Dict(
                spaces={
                    "observation": state_space,
                    "desired_goal": desired_goal_space,
                    "achieved_goal": achieved_goal_space
                },
            )
        else:
            # In this case, the env is not goal-conditioned
            self.observation_space = state_space

        # Define action space
        self.action_space = gym.spaces.Box(
            low=-self.max_action*np.ones(3),
            high=+self.max_action*np.ones(3)
        )

        self.reset()

    def step(self, action):
        """
        Simulate the system's transition under an action
        """
        # Update self.previous_achieved_goal before step
        self.previous_achieved_goal = self.current_achieved_goal.copy()

        # perturb action
        action += self.action_uncertainty * np.linalg.norm(action) *2*(np.random.rand(3)-1)
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

        # Update achieved_goal after simulation end
        self._update_achieved_goal()

        # Collect output
        observation = self._get_observation()
        reward = self._calculate_current_reward()
        done = False
        info = {
            "is_success": (np.linalg.norm(
                self.config.frame(
                    'box'
                ).getPosition()[:2] - self.config.frame(
                    'target'
                ).getPosition()[:2],
                axis=-1
            ) < self.target_tolerance)
        }

        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment randomly
        """
        if self.fixed_initial_config is None:
            # Sample a finger position and an allowed box position
            if self.plan_generator is None:
                if self.fixed_finger_initial_position:
                    finger_position = np.array([0, 0])
                else:
                    finger_position = self._sample_finger_pos()
                for _ in range(1000):
                    box_position = self._sample_box_position()
                    if self._box_finger_not_colliding(
                            finger_position,
                            box_position
                    ):
                        break
                goal_position = self._sample_box_position()
                precomputed_plan = None
            else:
                reset_data = self.plan_generator.sample()

                finger_position = reset_data['finger_position']
                box_position = reset_data['box_position']
                goal_position = reset_data['goal_position']
                precomputed_plan = reset_data['precomputed_plan']

        else:
            finger_position = self.fixed_initial_config["finger_position"]
            box_position = self.fixed_initial_config["box_position"]
            goal_position = self.fixed_initial_config["goal_position"]
            precomputed_plan = None

        return self._controlled_reset(
            finger_position,
            box_position,
            goal_position,
            precomputed_plan=precomputed_plan
        )

    def render(self, mode='human'):
        """
        Create interactive view of the environment
        """
        return self.config.view()

    def close(self):
        raise NotImplementedError

    def compute_reward(
            self,
            achieved_goal,
            desired_goal,
            info,
            previous_achieved_goal=None
    ):
        """
        This method exposes the reward function in a way that is compatible with
        the gym API for HER without plan-based reward shaping
        (see https://openai.com/blog/ingredients-for-robotics-research/).
        In addition, state current_plan can also be provided for plan-based reward
        shaping (and has to be if shaping mode is not None).
        Shaping mode potential_based also requires a previous_state
        """
        # Previous_reward has to be given if potential-based RS is used
        if self.plan_based_shaping.shaping_mode == 'potential_based':
            assert previous_achieved_goal is not None

        # If reward shaping is not used, desired_goal and achieved_goal
        # are 2D box positions and the binary reward can be calculated immediately
        if self.plan_based_shaping.shaping_mode is None:
            binary_reward = (np.linalg.norm(
                achieved_goal[:, :] - desired_goal[:, :],
                axis=-1
            ) < self.target_tolerance).astype(float)
        # If reward shaping is used, the desired box position is encoded in the last
        # step of the plan (i.e. desired_goal)
        else:
            # If the plan has been modified, recover the original plan
            # and use it for shaping
            if info is not None:
                if "original_plan" in info[0]:
                    desired_goal = np.array(
                        [ele["original_plan"] for ele in info]).copy()

            binary_reward = (np.linalg.norm(
                achieved_goal[:, -3:] - desired_goal.reshape(
                    -1,
                    self.plan_length,
                    self.dim_plan
                )[:, -1, -3:],
                axis=-1
            ) < self.target_tolerance).astype(float)

        return self.plan_based_shaping.shaped_reward_function(
            achieved_goal,
            desired_goal,
            binary_reward,
            previous_achieved_goal=previous_achieved_goal
        )

    def _get_approximate_plan(self):
        """
        Calculate approximate plan
        """

        if self.komo_plans:
            return self._get_komo_plan()
        else:
            return self._get_manhattan_plan()


    def _get_manhattan_plan(self):
        """
        Calculate Manhattan-like plan using the current
        state and target. This plan can not be directly executed
        in the physx simulation.
        """
        target_pos = self.config.frame(
            'target'
        ).getPosition()
        box_pos = self.config.frame(
            'box'
        ).getPosition()
        finger_pos = self.config.frame(
            'finger'
        ).getPosition()

        # underlying dim: finger init pos (2D), start+goal (4D),
        # plus 2D for all intermediate keyframes
        # the 2 pushes from the last intermediate (or the initial pos) to the goal
        # are a deterministic function of the position of the last intermediate
        # (or the initial pos) and the goal

        # define key frames
        keyframes = [box_pos.copy()] + [
            np.array(list(self._sample_box_position()) + [
                self.floor_level
            ]) for _ in range(self.n_keyframes)
        ] + [target_pos.copy(), target_pos.copy()]

        # modify intermediate keyframes
        for previous, current in zip(keyframes[:-3], keyframes[1:-2:]):
            if np.random.rand() >= 0.5:
                current[0] = previous[0]
            else:
                current[1] = previous[1]

        if self.n_keyframes == 0:
            # in this case, the first push is along the longest
            # direction
            first_dir = np.argmax(
                np.abs(target_pos - box_pos)
            )
            assert first_dir in [0, 1]
            second_dir = 0 if first_dir == 1 else 1
            keyframes[-2][first_dir] = target_pos[first_dir]
            keyframes[-2][second_dir] = box_pos[second_dir]
        else:
            # in this case, the second-to-last push is
            # perpendicular to the third-to-last
            direction = 1 if keyframes[-4][0] == keyframes[-3][0] else 0
            keyframes[-2][direction] = keyframes[-3][direction]

        waypoints = np.array(
            self._get_waypoints(finger_pos, keyframes)
        )

        return self._densify_waypoints(waypoints)


    def _densify_waypoints(self, waypoints):
        """
        Return a full plan from sequence of waypoints
        """
        distances = np.linalg.norm(
            waypoints[1:] - waypoints[:-1],
            axis=-1
        )
        distances = np.array([0] + list(distances))
        cumulated_distance = np.cumsum(distances)

        plan = interp1d(
            cumulated_distance,
            waypoints,
            kind='linear',
            axis=0
        )(np.linspace(
            cumulated_distance[0],
            cumulated_distance[-1],
            self.plan_length
        ))

        plan[:, 2] = plan[:, 2]-self.config.frame('floor').getPosition()[2]

        return plan.reshape(-1)


    def _get_waypoints(self, finger_initial, box_keyframes):
        """
        create waypoints from initial finger position and box keyframes
        """
        waypoints = []

        # 1st waypoint: initial pos
        waypoints.append(np.array([
            *finger_initial,
            *box_keyframes[0],
        ]))

        # 2nd waypoint: initial pos with elevated finger pos
        waypoints.append(np.array([
            finger_initial[0], finger_initial[1], finger_initial[2] + 0.4,
            *box_keyframes[0],
        ]))

        for ind, (from_frame, to_frame) in enumerate(
            zip(box_keyframes[:-1], box_keyframes[1:])
        ):
            # the following sequence basically performs a push
            # along a single direction
            # assert that steps only differ in 1 dimension
            assert sum(from_frame == to_frame) == 2

            first_direction = np.argmax(
                np.abs(to_frame - from_frame)
            )
            assert first_direction in [0, 1]

            # Offset vec for first contact
            offset_vec = [0, 0]
            offset_vec[first_direction] += (
                self.box_xy_size/2 + self.finger_radius
            ) * np.sign(from_frame[first_direction] - to_frame[first_direction])

            # 3rd waypoint: finger first touch, elevated
            waypoints.append(np.array([
                from_frame[0] + offset_vec[0],
                from_frame[1] + offset_vec[1],
                finger_initial[2] + 0.4,
                *from_frame,
            ]))

            # 4th waypoint: finger first touch, ground level
            waypoints.append(np.array([
                from_frame[0] + offset_vec[0],
                from_frame[1] + offset_vec[1],
                finger_initial[2],
                *from_frame,
            ]))

            # 5th waypoint: finger first touch at intermediate step, ground level
            intermediate_box_pos = from_frame.copy()
            intermediate_box_pos[first_direction] = to_frame[first_direction]
            waypoints.append(np.array([
                intermediate_box_pos[0] + offset_vec[0],
                intermediate_box_pos[1] + offset_vec[1],
                finger_initial[2],
                *intermediate_box_pos,
            ]))

            # do not perform the "step-back-and-go-up"
            # squence if it is the last
            if not ind == len(box_keyframes[:-1]) - 1:
                # Offset vec after first contact
                offset_vec = [0, 0]
                offset_vec[first_direction] += (
                    self.box_xy_size/2 + self.finger_radius + 0.2
                ) * np.sign(from_frame[first_direction] - to_frame[first_direction])

                # 6th waypoint: finger first touch at intermediate step, ground level, step back
                intermediate_box_pos = from_frame.copy()
                intermediate_box_pos[first_direction] = to_frame[first_direction]
                waypoints.append(np.array([
                    intermediate_box_pos[0] + offset_vec[0],
                    intermediate_box_pos[1] + offset_vec[1],
                    finger_initial[2],
                    *intermediate_box_pos,
                ]))

                # 7th waypoint: finger first touch at intermediate step, elevated
                waypoints.append(np.array([
                    intermediate_box_pos[0] + offset_vec[0],
                    intermediate_box_pos[1] + offset_vec[1],
                    finger_initial[2] + 0.4,
                    *intermediate_box_pos,
                ]))

        return waypoints


    def _get_komo_plan(self):
        """
        Uses rai/KOMO to calculate plan using the current
        state and target. This plan is based on the differentiable
        physics model rai uses, and can not be directly executed
        in the physx simulation.
        """

        assert self.n_keyframes == 0, "n_keyframes =/= 0 is not implemented for KOMO plans"
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

        assert no_contact_to_contact_ratio > 0

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
                np.array(plan).reshape(-1, self.dim_plan),
                self.komo.getPathFrames(['finger', 'box'])[
                    :, :, :3].reshape(-1, self.dim_plan)
            ),
            axis=0
        )

        # Densify plans
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

        # And resample according to the desired plan_length
        if self.plan_length is not None:
            plan = interp1d(
                np.linspace(0, 1, len(plan)),
                plan,
                axis=0,
                bounds_error=True
            )(np.linspace(0, 1, self.plan_length))

        return plan.reshape(-1)

    def _get_observation(self):
        """
        Returns current observation.
        """
        if self.fixed_initial_config is None:
            return {
                'observation': self._get_state(),
                'achieved_goal': self.current_achieved_goal.copy(),
                'desired_goal': self.current_desired_goal.copy()
            }
        else:
            return self._get_state()

    def _update_achieved_goal(self):
        """
        Update self.current_achieved_goal using _get_state()
        """
        if self.plan_based_shaping.shaping_mode is None:
            # without reward shaping, achieved_goal is 2D box position
            self.current_achieved_goal = self.config.frame(
                'box'
            ).getPosition()[:2]
        else:
            # with reward shaping, achieved_goal ist 3D finger + 3D box pos
            self.current_achieved_goal = self._get_state()[
                :self.dim_plan
            ]

    def _calculate_current_reward(self):
        """
        Calculate reward (shaped or unshaped) for the last action
        """
        # Previous achieved goal is only considered in potential_based mode
        previous_achieved_goal = None
        if self.plan_based_shaping.shaping_mode == 'potential_based':
            previous_achieved_goal = self.previous_achieved_goal.copy()[
                None, :]

        return float(self.compute_reward(
            self.current_achieved_goal.copy()[None, :],
            self.current_desired_goal.copy()[None, :],
            None,
            previous_achieved_goal=previous_achieved_goal
        ))

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
            goal_position,
            precomputed_plan=None
    ):
        """
        Reset the environment to specific state
        """
        # TODO runtime error when goal and initial box position too close
        # Reset previous_achieved_goal
        self.previous_achieved_goal = None

        # Check that box and finger are not in collision
        assert self._box_finger_not_colliding(
            finger_position,
            box_position
        ), "Invalid reset position: Finger and Box are colliding"

        # Set rai config
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

        # update achieved_goal according to new config
        self._update_achieved_goal()

        # update desired_goal according to new config
        if self.plan_based_shaping.shaping_mode is None:
            self.current_desired_goal = np.array(goal_position.copy())
        else:
            if self.fixed_initial_config is None:
                if precomputed_plan is None:
                    self.current_desired_goal = self._get_approximate_plan()
                else:
                    self.current_desired_goal = precomputed_plan
            else:
                # create self.static plan if it has not been initialized
                if self.static_plan is None:
                    if 'static_plan' in self.fixed_initial_config:
                        # the plan can be given by the user...
                        self.static_plan = self.fixed_initial_config["static_plan"]
                    else:
                        # ...or it can be calculated automatically
                        self.static_plan = self._get_approximate_plan()

                self.current_desired_goal = self.static_plan.copy()

        return self._get_observation()

    def _reset_box(self, coords):
        """
        Reset the box to an arbitrary position
        """
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
            self.reset_finger_xy_max - self.reset_finger_xy_min
        ) * np.random.rand(2) + self.reset_finger_xy_min

    def _sample_box_position(self):
        """
        Sample random position for the box on the table
        """
        return (
            self.reset_box_xy_max - self.reset_box_xy_min
        ) * np.random.rand(2) + self.reset_box_xy_min

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
