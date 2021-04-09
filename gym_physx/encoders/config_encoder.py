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
        fixed_finger_initial_position
    ):
        self.encoding_low = low
        self.encoding_high = high
        self.plan_len = plan_len
        self.plan_dim = plan_dim
        self.fixed_finger_initial_position = fixed_finger_initial_position
        self.hidden_dim = 4 if self.fixed_finger_initial_position else 6

    def encode(self, plan):
        plan_reshaped = plan.reshape(self.plan_len, self.plan_dim)
        box_initial_and_final = np.concatenate([
            plan_reshaped[0, 3:5],
            plan_reshaped[-1, 3:5]
        ])
        if self.fixed_finger_initial_position:
            return box_initial_and_final

        return np.concatenate([
            plan_reshaped[0, :2],
            box_initial_and_final
        ])

# %%
