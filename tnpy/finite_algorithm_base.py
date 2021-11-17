import logging
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tensornetwork import Node, conj
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
# from tnpy.linalg import svd
from tnpy.operators import MPO
from typing import List, Union


class FiniteAlgorithmBase:

    mps_cls = FiniteMPS

    def __init__(self, mpo: MPO, chi: Union[int, None] = None, init_method: str = 'random'):
        """

        Args:
            mpo:
            chi:
            init_method:
        """
        if init_method == 'random':
            assert isinstance(chi, int), "chi has to be given for random init"
        self.mpo = mpo
        self.d = mpo.physical_dimensions
        self.chi = chi
        self.left_envs = {}
        self.right_envs = {}
        self.left_norms = {}
        self.right_norms = {}
        self._mps = None
        self.mps = init_method

    def close(self):
        logging.info("Deleting left-/right- environments and norms")
        del self.left_envs
        del self.right_envs
        del self.left_norms
        del self.right_norms

    @property
    def N(self) -> int:
        return len(self.mpo.nodes)

    @property
    def dtype(self):
        return self.mpo[0].tensor.dtype

    @property
    def bond_dimensions(self):
        return self._mps.bond_dimensions

    @staticmethod
    def generate_bond_dimensions(N: int, d: int, chi: int) -> List[int]:
        if N % 2 == 0:
            D = [min(d ** i, chi) for i in range(1, N // 2)]
            D += [int(min(d ** (N / 2), chi))] + D[::-1]
        else:
            D = [min(d ** i, chi) for i in range(1, (N + 1) // 2)]
            D += D[::-1]
        return D

    @property
    def mps(self) -> FiniteMPS:
        return self._mps

    @mps.setter
    def mps(self, init_method: str = 'random') -> None:
        if init_method == "random":
            logging.info("Initializing finite MPS randomly")
            D = FiniteAlgorithmBase.generate_bond_dimensions(self.N, self.d, self.chi)
            self._mps = self.mps_cls.random([self.d]*self.N, D, self.dtype)
            # self.normalize_mps(direction=1, normalize_sv=True)
        elif Path(init_method).suffix:
            self.load_mps(init_method)
        else:
            raise KeyError("Invalid init method or file does not exist")
        self._init_envs()

    def load_mps(self, filename: str) -> None:
        logging.info(f"Initializing finite MPS from file: {filename}")
        path = Path(filename)
        if path.suffix == '.hdf5':
            # TODO: Not Implemented
            NotImplemented
        elif path.suffix == '.npz':
            npzfile = np.load(path)
            # @TODO: this has to be converted into object FiniteMPS
            self._mps = [npzfile['arr_%i' % i] for i in range(len(npzfile.files))]
            npzfile.close()
        else:
            raise FileNotFoundError(f"File {path} not exists or not supported.")

    def save_mps(self, filename: str) -> None:
        path = Path(filename)
        tensor_dataset = map(lambda node: node.tensor, self._mps.nodes)
        if path.suffix == '.hdf5':
            # @TODO: this requires more cares on axes
            with h5py.File(path, 'w') as hf:
                hf.create_dataset(str(path), tensor_dataset)
                hf.close()
        elif path.suffix == '.npz':
            np.savez(path, *tensor_dataset)
        else:
            raise NameError(f"File extension {path.suffix} is not supported.")

    def mps_shape(self, site: int):
        return self._mps.get_tensor(site).shape

    # def normalize_mps(self, direction, normalize_sv=False):
    #     if direction == 1:
    #         iterator = range(self.N-1)
    #     elif direction == -1:
    #         iterator = range(self.N-1, 0, -1)
    #
    #     for site in iterator:
    #         theta = self._mps.nodes[site].tensor
    #         if direction == 1:
    #             theta = theta.reshape(self.d * self.mps_shape(site)[0], -1)
    #         elif direction == -1:
    #             theta = theta.reshape(-1, self.d * self.mps_shape(site)[2])
    #         u, s, vt = np.linalg.svd(theta, full_matrices=False)
    #         if normalize_sv:
    #             s /= np.linalg.norm(s)
    #         if direction == 1:
    #             self._mps.nodes[site] = Node(u.reshape(self.mps_shape(site)))
    #             residual = Node(np.dot(np.diagflat(s), vt))
    #             G = self._mps.nodes[site+1]
    #             residual[1] ^ G[0]
    #             self._mps.nodes[site+1] = residual @ G
    #             self._update_left_env(site+1)
    #         elif direction == -1:
    #             self._mps.nodes[site] = Node(vt.reshape(self.mps_shape(site)))
    #             residual = Node(np.dot(u, np.diagflat(s)))
    #             G = self._mps.nodes[site-1]
    #             G[2] ^ residual[0]
    #             self._mps.nodes[site-1] = G @ residual
    #             self._update_right_env(site-1)

    def _init_envs(self):
        # @TODO: only need to do for one direction
        logging.info("Initializing left environments")
        for site in tqdm(range(1, self.N)):
            self._update_left_env(site)
        logging.info("Initializing right environments")
        for site in tqdm(range(self.N-2, -1, -1)):
            self._update_right_env(site)

    def _init_norms(self):
        # @TODO: only need to do for one direction
        logging.info("Initializing left norms")
        for site in tqdm(range(1, self.N)):
            self._update_left_norm(site)
        logging.info("Initializing right norms")
        for site in tqdm(range(self.N-2, -1, -1)):
            self._update_right_norm(site)

    def _update_left_env(self, site):
        W = self.mpo[site-1]
        M = Node(self._mps.get_tensor(site-1))
        M_conj = Node(conj(self._mps.get_tensor(site-1)))
        if site == 1:
            M[0] ^ M_conj[0]
            M[1] ^ W[1]
            W[2] ^ M_conj[1]
            self.left_envs[site] = M @ W @ M_conj
        else:
            L = self.left_envs[site-1]
            L[0] ^ M[0]
            L[1] ^ W[0]
            L[2] ^ M_conj[0]
            M[1] ^ W[2]
            M_conj[1] ^ W[3]
            self.left_envs[site] = L @ M @ W @ M_conj

    def _update_right_env(self, site):
        W = self.mpo[site+1]
        M = Node(self._mps.get_tensor(site+1))
        M_conj = Node(conj(self._mps.get_tensor(site+1)))
        if site == self.N-2:
            M[2] ^ M_conj[2]
            M[1] ^ W[1]
            W[2] ^ M_conj[1]
            self.right_envs[site] = M @ W @ M_conj
        else:
            R = self.right_envs[site+1]
            R[0] ^ M[2]
            R[1] ^ W[1]
            R[2] ^ M_conj[2]
            M[1] ^ W[2]
            M_conj[1] ^ W[3]
            self.right_envs[site] = R @ M @ W @ M_conj

    def _update_left_norm(self, site):
        M = self._mps.nodes[site-1]
        M_conj = conj(self._mps.nodes[site-1])
        if site == 1:
            M[0] ^ M_conj[0]
            M[1] ^ M_conj[1]
            self.left_norms[site] = M @ M_conj
        else:
            L = self.left_norms[site-1]
            L[0] ^ M[0]
            L[1] ^ M_conj[0]
            M[1] ^ M_conj[1]
            self.left_norms[site] = L @ M @ M_conj

    def _update_right_norm(self, site):
        M = self._mps.nodes[site+1]
        M_conj = conj(self._mps.nodes[site+1])
        if site == self.N-2:
            M[2] ^ M_conj[2]
            M[1] ^ M_conj[1]
            self.right_norms[site] = M @ M_conj
        else:
            R = self.right_norms[site+1]
            R[0] ^ M[2]
            R[1] ^ M_conj[2]
            M[1] ^ M_conj[1]
            self.right_norms[site] = R @ M @ M_conj
