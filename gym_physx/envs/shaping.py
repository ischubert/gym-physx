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
            potential_based_scaling=1,
            add_1_to_reward=False
    ):
        self.shaping_mode = shaping_mode
        self.gamma = gamma
        self.potential_function = potential_function
        self.width = width
        self.potential_based_scaling = potential_based_scaling
        self.add_1_to_reward = add_1_to_reward

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

    def shaped_reward_function(
            self,
            state,
            reward,
            plan=None,
            previous_state=None
    ):
        """
        Get shaped reward function from the reward given by the
        FetchPush-v1 environment
        """

        if self.add_1_to_reward:
            reward += 1

        if self.shaping_mode is None:
            return reward

        assert plan is not None

        if self.shaping_mode == 'potential_based':
            assert previous_state is not None
            assert self.gamma is not None
            return reward + self.gamma * self.potential_based_scaling*(
                self.plan_based_reward(
                    state,
                    plan
                ) - self.plan_based_reward(
                    previous_state,
                    plan
                )
            )

        if self.shaping_mode == 'relaxed':
            if reward == 1:
                return reward
            return reward + self.plan_based_reward(
                state,
                plan
            )

    def plan_based_reward(self, state, plan):
        """
        give value of state based on distance to plan and
        how far advanced in the plan the corresponding state is
        """

        if self.potential_function == 'gaussian':
            exponential_dists = np.exp(
                -np.linalg.norm(
                    state[None, :] - plan[:, :],
                    axis=-1
                )**2/2/self.width**2
            )

            smallest_dist = np.argmax(exponential_dists)

            return 0.5 * exponential_dists[
                smallest_dist
            ] * smallest_dist/len(plan)
        if self.potential_function == 'box_distance':
            # simply use the negative box distance in this case,
            # ignoring the plan!
            return -np.linalg.norm(
                plan[-1, 3:] - state[3:]
            )

        raise Exception(
            f"potential_function={self.potential_function} is not known")
