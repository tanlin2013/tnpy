import abc
import numpy as np
from tnpy.operators import MPO


class ModelBase(abc.ABC):

    def __init__(self, n: int):
        """

        Args:
            n: System size.
        """
        self._n = n

    @property
    def n(self) -> int:
        return self._n

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
