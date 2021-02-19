"""
Classes for reward shaping in gym-phsyx
"""
import numpy as np


class PlanBasedShaping():
    """
    Plan-based shaping for gym-physx environments
    """

    def __init__(
            self,
            shaping_mode=None,
            gamma=None,
            potential_function='gaussian',
            width=0.2,
            potential_based_scaling=1
    ):
        self.shaping_mode = shaping_mode
        self.gamma = gamma
        self.potential_function = potential_function
        self.width = width
        self.potential_based_scaling = potential_based_scaling

        self.plan_len = None
        self.plan_dim = None

        assert self.shaping_mode in [
            None,
            'potential_based',
            'relaxed'
        ]

        assert self.potential_function in [
            'gaussian',
            'box_distance'
        ]

        if self.shaping_mode == 'potential_based':
            assert self.gamma is not None

    def set_plan_len_and_dim(self, plan_len=None, plan_dim=None):
        """
        Setter function for plan_len and plan_dim
        """
        self.plan_len = plan_len
        self.plan_dim = plan_dim


    def shaped_reward_function(
            self,
            achieved_goal,
            desired_goal,
            reward,
            previous_achieved_goal=None
    ):
        """
        Get shaped reward function from the reward given by the
        FetchPush-v1 environment
        """

        if self.shaping_mode is None:
            return reward

        assert (
            self.plan_len is not None
        ) and (
            self.plan_dim is not None
        ), "Please use set_plan_len_and_dim() to initialize PlanBasedShaping"

        if self.shaping_mode == 'potential_based':
            assert previous_achieved_goal is not None
            assert self.gamma is not None
            return reward + self.gamma * self.potential_based_scaling*(
                self.plan_based_reward(
                    achieved_goal,
                    desired_goal
                ) - self.plan_based_reward(
                    previous_achieved_goal,
                    desired_goal
                )
            )

        if self.shaping_mode == 'relaxed':
            mask = np.logical_not(reward == 1)
            reward[mask] = reward[mask] + self.plan_based_reward(
                achieved_goal[mask, :],
                desired_goal[mask, :]
            )
            return reward

    def plan_based_reward(self, achieved_goal, desired_goal):
        """
        give value of state based on distance to plan and
        how far advanced in the plan the corresponding state is
        """

        # TODO double check that this is always correct
        desired_goal = desired_goal.reshape(-1, self.plan_len, self.plan_dim)

        if self.potential_function == 'gaussian':
            exponential_dists = np.exp(
                -np.linalg.norm(
                    achieved_goal[:, None, :] - desired_goal[:, :, :],
                    axis=-1
                )**2/2/self.width**2
            )

            smallest_dist = np.argmax(exponential_dists, axis=-1)
            return 0.5 * exponential_dists[
                np.arange(len(exponential_dists)), smallest_dist
            ] * smallest_dist/self.plan_len

        if self.potential_function == 'box_distance':
            # simply use the negative box distance in this case,
            # ignoring the plan!
            return -np.linalg.norm(
                desired_goal[:, -1, 3:] - achieved_goal[:, 3:]
            )

        raise Exception(
            f"potential_function={self.potential_function} is not known")
