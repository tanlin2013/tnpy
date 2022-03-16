import re
import h5py
import numpy as np
import quimb.tensor as qtn
from tensornetwork import Node
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from tnpy import logger
from tnpy.linalg import svd, LinearOperator
from tnpy.operators import MatrixProductOperator
from typing import Dict


class Direction(Enum):
    rightward = 1
    leftward = -1


class MatrixProductState(qtn.MatrixProductState):

    def __init__(self, *args, **kwargs):
        super(MatrixProductState, self).__init__(*args, **kwargs)
        self.permute_arrays('lpr')

    @property
    def n_sites(self) -> int:
        return self.nsites

    @property
    def phys_dim(self) -> int:
        return self[0].shape[0]

    @property
    def bond_dim(self) -> int:
        return self.max_bond()

    def __getitem__(self, site: int) -> qtn.Tensor:
        tensor, = self.select_tensors(self.site_tag(site))
        return tensor

    def permute_arrays(self, shape='lrp'):
        """Permute the indices of each tensor in this MPS to match ``shape``.
        This doesn't change how the overall object interacts with other tensor
        networks but may be useful for extracting the underlying arrays
        consistently. This is an inplace operation.

        Parameters
        ----------
        shape : str, optional
            A permutation of ``'lrp'`` specifying the desired order of the
            left, right, and physical indices respectively.
        """
        for i in self.sites:
            inds = {'p': self.site_ind(i)}
            if self.cyclic or i > 0:
                inds['l'] = self.bond(i, (i - 1) % self.n_sites)
            if self.cyclic or i < self.n_sites - 1:
                inds['r'] = self.bond(i, (i + 1) % self.n_sites)
            inds = [inds[s] for s in shape if s in inds]
            self[i].transpose_(*inds)

    def conj(self, mangle_inner: bool = False, mangle_outer: bool = False) -> 'MatrixProductState':
        mps = super(MatrixProductState, self).conj()
        if mangle_inner:
            mps.reindex({ind: qtn.rand_uuid() for ind in mps.inner_inds()}, inplace=True)
        if mangle_outer:
            mps.reindex({
                ind: re.sub(r'^k([0-9]+)', r'b\1', ind) for ind in mps.outer_inds()
            }, inplace=True)
        return mps

    def save(self, filename: str, extension: str = 'npz'):
        """

        Args:
            filename:
            extension:

        Returns:

        """
        tensor_datasets = {self.site_tag(site): self[site].data for site in range(self.nsites)}
        filepath = Path(f"{filename}.{extension}")
        if extension == 'hdf5':
            with h5py.File(filepath, 'w') as f:
                for tag, array in tensor_datasets.items():
                    f.create_dataset(tag, data=array)
                f.close()
        elif extension == 'npz':
            np.savez(filepath, **tensor_datasets)
        else:
            raise KeyError(f"File extension {extension} is not supported.")

    @classmethod
    def load(cls, filename: str, extension: str = 'npz') -> 'MatrixProductState':
        """

        Args:
            filename:
            extension:

        Returns:

        """
        filepath = Path(f"{filename}.{extension}")
        if extension == 'hdf5':
            f = h5py.File(filepath, 'r')
            return cls([array[()] for array in f.values()])
        elif extension == 'npz':
            f = np.load(filepath)
            return cls([array for array in f.values()])
        else:
            raise KeyError(f"File extension {extension} is not supported.")

    @classmethod
    def random(cls, n: int, bond_dim: int, phys_dim: int, **kwargs) -> 'MatrixProductState':
        """

        Args:
            n:
            bond_dim:
            phys_dim:

        Returns:

        """
        rand_mps = qtn.MPS_rand_state(n=n, bond_dim=bond_dim, phys_dim=phys_dim)
        rand_mps.compress()
        return cls([tensor.data for tensor in rand_mps], **kwargs)

    def split_tensor(self, site: int, direction: Direction):
        if direction == direction.rightward:
            psi = self[site].data if site == 0 \
                else self[site].data.reshape(self.phys_dim * self[site].shape[0], -1)
            u, s, vt = svd(psi, cutoff=self[site].shape[-1])
            self[site].modify(data=u.reshape(self[site].shape))
            residual = Node(np.diagflat(s) @ vt)
            neighbour = Node(self[site + 1].data)
            residual[1] ^ neighbour[0]
            self[site + 1].modify(data=(residual @ neighbour).tensor)
        elif direction == direction.leftward:
            psi = self[site].data if site == self.n_sites - 1 \
                else self[site].data.reshape(-1, self.phys_dim * self[site].shape[2])
            u, s, vt = svd(psi, cutoff=self[site].shape[0])
            self[site].modify(data=vt.reshape(self[site].shape))
            residual = Node(u @ np.diagflat(s))
            neighbour = Node(self[site - 1].data)
            neighbour[-1] ^ residual[0]
            self[site - 1].modify(data=(neighbour @ residual).tensor)
        else:
            raise KeyError

    def enlarge_bond_dim(self):
        return NotImplemented


class Environment:

    def __init__(self, mpo: MatrixProductOperator, mps: MatrixProductState):
        self._mpo = mpo
        self._mps = mps
        self._conj_mps = mps.conj(mangle_inner=True, mangle_outer=True)
        self._n_sites = mpo.nsites
        self._left, self._right = {}, {}
        for site in tqdm(range(1, self.n_sites), desc="Initializing left environments"):
            self.update_left(site)
        for site in tqdm(range(self.n_sites - 2, -1, -1), desc="Initializing right environments"):
            self.update_right(site)

    def close(self):
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
        return self._left

    @property
    def right(self) -> Dict[int, qtn.Tensor]:
        return self._right

    def update_left(self, site: int):
        if site == 1:
            net = self._mps[site - 1] & \
                  self._mpo[site - 1] & \
                  self._conj_mps[site - 1]
        else:
            net = self._left[site - 1] & \
                  self._mps[site - 1] & \
                  self._mpo[site - 1] & \
                  self._conj_mps[site - 1]
        self._left[site] = net.contract()

    def update_right(self, site: int):
        if site == self.n_sites - 2:
            net = self._mps[site + 1] & \
                  self._mpo[site + 1] & \
                  self._conj_mps[site + 1]
        else:
            net = self._right[site + 1] & \
                  self._mps[site + 1] & \
                  self._mpo[site + 1] & \
                  self._conj_mps[site + 1]
        self._right[site] = net.contract()

    def update(self, site: int, direction: Direction):
        if direction == Direction.rightward:
            self.update_left(site + 1)
        elif direction == Direction.leftward:
            self.update_right(site - 1)

    def update_mps(self, site: int, data: np.ndarray):
        self._mps[site].modify(data=data)
        self._conj_mps[site].modify(data=data)

    def split_tensor(self, site: int, direction: Direction):
        self._mps.split_tensor(site, direction)
        if direction == direction.leftward:
            self._conj_mps[site].modify(data=self._mps[site].data)
            self._conj_mps[site - 1].modify(data=self._mps[site - 1].data)
        elif direction == direction.rightward:
            self._conj_mps[site].modify(data=self._mps[site].data)
            self._conj_mps[site + 1].modify(data=self._mps[site + 1].data)

    def one_site_full_matrix(self, site: int) -> np.ndarray:
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
                'k': [self.left[site].inds[0], self.mpo[site].inds[-2], self.right[site].inds[0]],
                'b': [self.left[site].inds[-1], self.mpo[site].inds[-1], self.right[site].inds[-1]]
            }
        return tn.contract().fuse(fuse_map).data

    def one_site_matvec(self, site: int) -> LinearOperator:
        def matvec(x: np.ndarray) -> np.ndarray:
            vec = qtn.Tensor(x.reshape(self.mps[site].shape), inds=self.mps[site].inds)
            if site == 0:
                tn = self.right[site] & self.mpo[site] & vec
                output_inds = [self.mpo[site].inds[-1], self.right[site].inds[-1]]
            elif site == self.n_sites - 1:
                tn = self.left[site] & self.mpo[site] & vec
                output_inds = [self.left[site].inds[-1], self.mpo[site].inds[-1]]
            else:
                tn = self.left[site] & self.right[site] & self.mpo[site] & vec
                output_inds = [self.left[site].inds[-1], self.mpo[site].inds[-1], self.right[site].inds[-1]]
            return tn.contract(output_inds=output_inds).data.reshape(-1, 1)
        return LinearOperator((self.mps[site].size, self.mps[site].size), matvec=matvec)
