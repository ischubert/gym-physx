# %%
"""
Simple encoder class for testing
"""
from gym_physx.encoders.base_encoder import BaseEncoder
import numpy as np


class ToyEncoder(BaseEncoder):
    """
    Simple encoder class for testing
    """
    encoding_low = -1
    encoding_high = 1
    hidden_dim = 5

    def encode(self, plan):
        return np.linspace(self.encoding_low, self.encoding_high, self.hidden_dim)

# %%
