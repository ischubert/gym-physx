# %%
"""
Encoder class that encodes plan based on initial and final config
"""
import numpy as np
from gym_physx.encoders.base_encoder import BaseEncoder


class ConfigEncoder(BaseEncoder):
    """
    Encoder class that encodes plan based on initial and final config
    """
    encoding_low = None
    encoding_high = None
    hidden_dim = None

    def __init__(
            self,
            low, high,
            plan_len, plan_dim,
            fixed_finger_initial_position,
            n_keyframes
    ):
        self.encoding_low = low
        self.encoding_high = high
        self.plan_len = plan_len
        self.plan_dim = plan_dim
        self.fixed_finger_initial_position = fixed_finger_initial_position
        self.n_keyframes = n_keyframes
        self.hidden_dim = 2*self.n_keyframes + (
            4 if self.fixed_finger_initial_position else 6
        )

    def encode(self, plan):
        # add initial and final box positions
        plan_reshaped = plan.reshape(self.plan_len, self.plan_dim)
        relevant_box_positions = np.concatenate([
            plan_reshaped[0, 3:5],
            plan_reshaped[-1, 3:5]
        ])

        if self.n_keyframes > 0:
            # in this case, all intermediate positions
            # but the second-to-last have to be added as well

            # filter out all box positions that appear longer than
            # for a single time step
            _, unique_indices, unique_counts = np.unique(
                plan_reshaped[:, 3:5],
                axis=0,
                return_counts=True,
                return_index=True
            )
            # inds_resting_box are the indices of the resting box at the beginning,
            # at the intermediate steps, but not the last position
            # (since plan ends immediately at the end)
            inds_resting_box = np.sort(unique_indices[unique_counts > 1])
            assert len(inds_resting_box) == 2+self.n_keyframes

            # we are not interested in the beginning, not interested in the last position,
            # and not interested in the second-to-last position, since the second-to-last position
            # is a function of the last position and the third-to-last position
            # therefore:
            ind_intermediate_positions = inds_resting_box[
                1: -1 # leave out first and second-to-last
            ]
            relevant_box_positions = np.concatenate([
                relevant_box_positions,
                plan_reshaped[ind_intermediate_positions, 3:5].reshape(-1)
            ])

        if self.fixed_finger_initial_position:
            assert len(relevant_box_positions) == self.hidden_dim
            return relevant_box_positions

        # add initial finger position
        relevant_finger_and_box_positions = np.concatenate([
            plan_reshaped[0, :2],
            relevant_box_positions
        ])
        assert len(relevant_finger_and_box_positions) == self.hidden_dim
        return relevant_finger_and_box_positions

# %%
