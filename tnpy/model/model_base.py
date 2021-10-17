import abc
import numpy as np
from tnpy.operators import MPO


class ModelBase(abc.ABC):

    def __init__(self, N: int):
        """

        Args:
            N: System size.
        """
        self.N = N

    @abc.abstractmethod
    def _elem(self, site: int) -> np.ndarray:
        return NotImplemented

    @property
    def mpo(self) -> MPO:
        """
        Return matrix product operator (mpo) as a property of the model.

        Returns:
            mpo:
        """
        return MPO(self.N, self._elem)
