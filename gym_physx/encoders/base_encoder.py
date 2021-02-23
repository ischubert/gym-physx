"""
Base encoder class
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseEncoder(ABC):
    """
    Abstract encoder class for desired_goal
    """

    @property
    @abstractmethod
    def encoding_low(self) -> float:
        """
        Lower boundary of encoding space
        """

    @property
    @abstractmethod
    def encoding_high(self) -> float:
        """
        Upper boundary of encoding space
        """

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """
        Encoding dimension
        """

    @abstractmethod
    def encode(self, plan: np.ndarray) -> np.ndarray:
        """
        Encode goal and return
        """
