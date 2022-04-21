from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import quimb.tensor as qtn
from tensornetwork import Node
from tqdm import tqdm

from tnpy import logger
from tnpy.linalg import svd, LinearOperator
from tnpy.operators import MatrixProductOperator


class Direction(Enum):
    """
    Enumeration for specifying the direction (leftward or rightward).
    """
    RIGHTWARD = 1
    LEFTWARD = -1


class MatrixProductState(qtn.MatrixProductState):

    def __init__(self, *args, **kwargs):
        """
        Matrix product state (MPS) for a finite size system.

        Args:
            *args: Positional arguments for :class:`qtn.MatrixProductState`.
            **kwargs: Keyword arguments for :class:`qtn.MatrixProductState`.

        Note:
            This class inherits directly from :class:`qtn.MatrixProductState`,
            with some added features.
            But, note that the underlying tensors are in the shape 'lpr',
            which differs from the parent class :class:`qtn.MatrixProductState`.
        """
        super().__init__(*args, **kwargs)
        self.permute_arrays('lpr')

    @property
    def n_sites(self) -> int:
        return self.nsites

    @property
    def phys_dim(self) -> int:
        """
        Physical dimensions of local state.

        Returns:

        """
        return self[0].shape[0]

    @property
    def bond_dim(self) -> int:
        """
        The bond dimensions.

        Returns:

        Notes:
            The actual bond dimensions can be smaller
            than this one at the boundaries.
        """
        return self.max_bond()

    def __getitem__(self, site: int) -> qtn.Tensor:
        tensor, = self.select_tensors(self.site_tag(site))
        return tensor

    def permute_arrays(self, shape: str = 'lrp'):
        """
        Permute the indices of each tensor in this MPS to match ``shape``.
        This doesn't change how the overall object interacts with other tensor
        networks but may be useful for extracting the underlying arrays
        consistently. This is an inplace operation.

        Args:
            shape: A permutation of ``'lrp'`` specifying the desired order of
                the left, right, and physical indices respectively.
        """
        for i in self.sites:
            inds = {'p': self.site_ind(i)}
            if self.cyclic or i > 0:
                inds['l'] = self.bond(i, (i - 1) % self.n_sites)
            if self.cyclic or i < self.n_sites - 1:
                inds['r'] = self.bond(i, (i + 1) % self.n_sites)
            inds = [inds[s] for s in shape if s in inds]
            self[i].transpose_(*inds)

    def conj(self,
             mangle_inner: bool = False, mangle_outer: bool = False
             ) -> MatrixProductState:
        """
        Create a conjugated copy of this :class:`~MatrixProductState` instance.

        Args:
            mangle_inner: Whether to rename the inner indices,
                so that there will be no conflict with the original one
                when contracting the network.
            mangle_outer: Whether to rename the outer indices,
                so that there will be no conflict with the original one
                when contracting the network.

        Returns:
            conj_mps:
        """
        mps = super().conj()
        if mangle_inner:
            mps.reindex({
                ind: qtn.rand_uuid() for ind in mps.inner_inds()
            }, inplace=True)
        if mangle_outer:
            mps.reindex({
                ind: re.sub(r'^k(\d+)', r'b\1', ind) for ind in mps.outer_inds()
            }, inplace=True)
        return mps

    def save(self, filename: str):
        """
        Save the underlying tensors of this :class:`~MatrixProductState`
        in the order of indices ``'lpr'`` into file.
        Only accepts suffix ``'.hdf5'`` or ``'.npz'`` as file extension.

        Args:
            filename: Absolute path to file in local disk.

        Returns:

        """
        # @TODO: solve ordering issue
        tensor_datasets = {
            self.site_tag(site): self[site].data for site in range(self.nsites)
        }
        filepath = Path(filename)
        extension = filepath.suffix
        if extension == '.hdf5':
            with h5py.File(str(filepath), 'w') as f:
                for tag, array in tensor_datasets.items():
                    f.create_dataset(tag, data=array)
                f.close()
        elif extension == '.npz':
            np.savez(str(filepath), **tensor_datasets)
        else:
            raise ValueError(f"File extension {extension} is not supported.")

    @classmethod
    def load(cls, filename: str) -> MatrixProductState:
        """
        Initialize the underlying tensors of :class:`~MatrixProductState`
        from a file. Tensors in the file should take the order of indices
        ``'lpr'``. Only accepts suffix ``'.hdf5'`` or ``'.npz'`` as
        file extension.

        Args:
            filename: Absolute path to file in local disk.

        Returns:
            mps:
        """
        # @TODO: solve ordering issue
        filepath = Path(filename)
        extension = filepath.suffix
        if extension == '.hdf5':
            f = h5py.File(str(filepath), 'r')
            return cls([array[()] for array in f.values()])
        elif extension == '.npz':
            f = np.load(str(filepath))
            return cls([array for array in f.values()])
        else:
            raise ValueError(f"File extension {extension} is not supported.")

    @classmethod
    def random(cls,
               n: int, bond_dim: int, phys_dim: int, **kwargs
               ) -> MatrixProductState:
        """
        Create a randomly initialized :class:`~MatrixProductState`.

        Args:
            n: The system size.
            bond_dim: The bond dimensions.
            phys_dim: Physical dimensions of local state.

        Returns:

        """
        rand_mps = qtn.MPS_rand_state(n=n, bond_dim=bond_dim, phys_dim=phys_dim)
        rand_mps.compress()
        return cls([tensor.data for tensor in rand_mps], **kwargs)

    def split_tensor(self, site: int, direction: Direction):
        """
        Split the tensor at given ``site`` into two tensors,
        and multiply the one on left or right ``direction`` into its
        neighbouring tensor. This is an inplace operation.

        Args:
            site:
            direction:

        Returns:

        """
        if direction == direction.RIGHTWARD:
            psi = self[site].data if site == 0 else \
                self[site].data.reshape(self.phys_dim * self[site].shape[0], -1)
            u, s, vt = svd(psi, cutoff=self[site].shape[-1])
            self[site].modify(data=u.reshape(self[site].shape))
            residual = Node(np.diagflat(s) @ vt)
            neighbour = Node(self[site + 1].data)
            residual[1] ^ neighbour[0]
            self[site + 1].modify(data=(residual @ neighbour).tensor)
        elif direction == direction.LEFTWARD:
            psi = self[site].data if site == self.n_sites - 1 else \
                self[site].data.reshape(-1, self.phys_dim * self[site].shape[2])
            u, s, vt = svd(psi, cutoff=self[site].shape[0])
            self[site].modify(data=vt.reshape(self[site].shape))
            residual = Node(u @ np.diagflat(s))
            neighbour = Node(self[site - 1].data)
            neighbour[-1] ^ residual[0]
            self[site - 1].modify(data=(neighbour @ residual).tensor)
        else:
            raise KeyError(
                "MatrixProductState only supplies left or right direction."
            )

    def enlarge_bond_dim(self, new_bond_dim: int, method: str):
        return NotImplemented


class Environment:

    def __init__(self, mpo: MatrixProductOperator, mps: MatrixProductState):
        """
        The effective environment for :class:`~MatrixProductState`-based
        algorithms.

        Args:
            mpo: Input :class:`~MatrixProductOperator`.
            mps: Input :class:`~MatrixProductState`.
        """
        self._mpo = mpo
        self._mps = mps
        self._conj_mps = mps.conj(mangle_inner=True, mangle_outer=True)
        self._n_sites = mpo.nsites
        self._left, self._right = {}, {}
        for site in tqdm(
                range(1, self.n_sites),
                desc="Initializing left environments"
        ):
            self.update_left(site)
        for site in tqdm(
                range(self.n_sites - 2, -1, -1),
                desc="Initializing right environments"
        ):
            self.update_right(site)

    def close(self):
        """
        Delete the underlying data of left and right environments.

        Returns:

        """
        logger.info("Deleting left and right environments.")
        del self._left, self._right

    @property
    def mpo(self) -> MatrixProductOperator:
        return self._mpo

    @property
    def mps(self) -> MatrixProductState:
        return self._mps

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def left(self) -> Dict[int, qtn.Tensor]:
        """
        Get a dictionary where left environments are stored on the respective
        site.

        Returns:

        """
        return self._left

    @property
    def right(self) -> Dict[int, qtn.Tensor]:
        """
        Get a dictionary where right environments are stored on the respective
        site.

        Returns:

        """
        return self._right

    def update_left(self, site: int):
        """
        Update the left environment at ``site``.

        Args:
            site:

        Returns:

        """
        if site == 1:
            tn = self._mps[site - 1] & \
                  self._mpo[site - 1] & \
                  self._conj_mps[site - 1]
        else:
            tn = self._left[site - 1] & \
                  self._mps[site - 1] & \
                  self._mpo[site - 1] & \
                  self._conj_mps[site - 1]
        self._left[site] = tn.contract()

    def update_right(self, site: int):
        """
        Update the right environment at ``site``.

        Args:
            site:

        Returns:

        """
        if site == self.n_sites - 2:
            tn = self._mps[site + 1] & \
                  self._mpo[site + 1] & \
                  self._conj_mps[site + 1]
        else:
            tn = self._right[site + 1] & \
                  self._mps[site + 1] & \
                  self._mpo[site + 1] & \
                  self._conj_mps[site + 1]
        self._right[site] = tn.contract()

    def update(self, site: int, direction: Direction):
        """
        Alias to :func:`~Environment.update_left` or
        :func:`~Environment.update_right` depending on the given ``direction``.

        Args:
            site:
            direction:

        Returns:

        """
        if direction == Direction.RIGHTWARD:
            self.update_left(site + 1)
        elif direction == Direction.LEFTWARD:
            self.update_right(site - 1)

    def update_mps(self, site: int, data: np.ndarray):
        self._mps[site].modify(data=data)
        self._conj_mps[site].modify(data=data)

    def split_tensor(self, site: int, direction: Direction):
        self._mps.split_tensor(site, direction)
        self._conj_mps[site].modify(data=self._mps[site].data)
        if direction == direction.LEFTWARD:
            self._conj_mps[site - 1].modify(data=self._mps[site - 1].data)
        elif direction == direction.RIGHTWARD:
            self._conj_mps[site + 1].modify(data=self._mps[site + 1].data)

    def variance(self) -> float:
        tn1 = self._mps & self.mpo.square() & self._conj_mps
        tn2 = self._mps & self.mpo & self._conj_mps
        return tn1.contract() - tn2.contract() ** 2

    def one_site_full_matrix(self, site: int) -> np.ndarray:
        """
        Construct the effective matrix for variational solver,
        with one site remained un-contracted.

        Args:
            site:

        Returns:

        """
        if site == 0:
            tn = self.right[site] & self.mpo[site]
            fuse_map = {
                'k': [self.mpo[site].inds[-2], self.right[site].inds[0]],
                'b': [self.mpo[site].inds[-1], self.right[site].inds[-1]]
            }
        elif site == self.n_sites - 1:
            tn = self.left[site] & self.mpo[site]
            fuse_map = {
                'k': [self.left[site].inds[0], self.mpo[site].inds[-2]],
                'b': [self.left[site].inds[-1], self.mpo[site].inds[-1]]
            }
        else:
            tn = self.left[site] & self.right[site] & self.mpo[site]
            fuse_map = {
                'k': [
                    self.left[site].inds[0],
                    self.mpo[site].inds[-2],
                    self.right[site].inds[0]
                ],
                'b': [
                    self.left[site].inds[-1],
                    self.mpo[site].inds[-1],
                    self.right[site].inds[-1]
                ]
            }
        return tn.contract().fuse(fuse_map).data

    def one_site_matvec(self, site: int) -> LinearOperator:
        """
        Construct the effective linear operator (matrix-vector product) for
        variational solver, with one site remained un-contracted.

        Args:
            site:

        Returns:

        """
        def matvec(x: np.ndarray) -> np.ndarray:
            vec = qtn.Tensor(x.reshape(self.mps[site].shape),
                             inds=self.mps[site].inds)
            if site == 0:
                tn = self.right[site] & self.mpo[site] & vec
                output_inds = [
                    self.mpo[site].inds[-1],
                    self.right[site].inds[-1]
                ]
            elif site == self.n_sites - 1:
                tn = self.left[site] & self.mpo[site] & vec
                output_inds = [
                    self.left[site].inds[-1],
                    self.mpo[site].inds[-1]
                ]
            else:
                tn = self.left[site] & self.right[site] & self.mpo[site] & vec
                output_inds = [
                    self.left[site].inds[-1],
                    self.mpo[site].inds[-1],
                    self.right[site].inds[-1]
                ]
            return tn.contract(output_inds=output_inds).data.reshape(-1, 1)
        return LinearOperator(
            shape=(self.mps[site].size, self.mps[site].size),
            matvec=matvec
        )


class MatrixProductStateMeasurements:

    def __init__(self, mps: MatrixProductState):
        self._mps = mps

    def expectation_value(self, mpo: MatrixProductOperator = None) -> float:
        tn = self._mps.conj(mangle_inner=True) & self._mps if mpo is None else \
            self._mps.conj(
                mangle_inner=True, mangle_outer=True
            ) & mpo & self._mps
        return tn.contract()
