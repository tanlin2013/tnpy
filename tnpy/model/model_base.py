import abc
import numpy as np
from tnpy.operators import MatrixProductOperator


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
    def mpo(self) -> MatrixProductOperator:
        """
        Return matrix product operator (mpo) as a property of the model.

        Returns:
            mpo:
        """
        tensors = [self._elem(site) for site in range(self.n)]
        tensors[0] = tensors[0][0, :, :, :]
        tensors[-1] = tensors[-1][:, -1, :, :]
        return MatrixProductOperator(tensors)
