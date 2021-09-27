import numpy as np
import logging
from itertools import count
from primme import eigsh
from tensornetwork import Node, Tensor
from dataclasses import dataclass, field
from tnpy.operators import MPO
from typing import Tuple, List, Union


@dataclass
class Projection:
    HarmonicRitz: str = 'primme_proj_harmonic'
    RefinedRitz: str = 'primme_proj_refined'
    RayleighRitz: str = 'primme_proj_RR'


@dataclass
class TreeNode:
    id: int
    node: Node
    gap: float = None
    left: 'TreeNode' = None
    right: 'TreeNode' = None


@dataclass
class GapCache:
    gap: List[float] = field(default_factory=list)
    evecs: List[np.ndarray] = field(default_factory=list)


class SDRG:

    def __init__(self, mpo: MPO, chi: int):
        self.mpo = mpo
        self._chi = chi
        _identity = np.identity(mpo.bond_dimensions)
        self._v_left = Node(_identity[0])
        self._v_right = Node(_identity[-1])
        self._tree = [
            TreeNode(id=site, node=self.mpo.nodes[site])
            for site in range(self.N)
        ]
        self.gap_cache = GapCache()

    @property
    def N(self) -> int:
        return len(self.mpo.nodes)

    @property
    def chi(self) -> int:
        return self._chi

    @property
    def v_left(self) -> Node:
        return self._v_left

    @property
    def v_right(self) -> Node:
        return self._v_right

    @property
    def tree(self) -> List[TreeNode]:
        return self._tree

    def block_hamiltonian(self, site: int) -> Union[Tensor, np.ndarray]:
        W1 = self.mpo.nodes[site]
        W2 = self.mpo.nodes[site + 1]
        if self.N == 2:
            W1[0] ^ W2[0]
            M = W1 @ W2
        elif site == 0:
            W1[0] ^ W2[0]
            W2[1] ^ self.v_right[0]
            M = W1 @ W2 @ self.v_right
        elif site == self.N - 2:
            self.v_left[0] ^ W1[0]
            W1[1] ^ W2[0]
            M = self.v_left @ W1 @ W2
        else:
            self.v_left[0] ^ W1[0]
            W1[1] ^ W2[0]
            W2[1] ^ self.v_right[0]
            M = self.v_left @ W1 @ W2 @ self.v_right
        shape = int(np.sqrt(M.tensor.size))
        return M.tensor.reshape(shape, shape)

    def largest_gap(self, evals: np.ndarray) -> Tuple[int, float]:
        gaps = np.diff(evals)
        max_idx = np.argmax(gaps)
        return max_idx, gaps[max_idx]

    def entanglement_rendering(self, evecs: np.ndarray) -> np.ndarray:
        def von_neumann_entropy(v: np.ndarray) -> float:
            singular_values = np.linalg.svd(v.reshape(2, -1))[1]
            ss = np.square(singular_values)
            return -1 * np.sum(ss @ np.log(ss))
        return np.array([von_neumann_entropy(v) for v in evecs.T])

    def eigen_solver(self, site: int,) -> Tuple[np.ndarray, np.ndarray]:
        block_ham = self.block_hamiltonian(site)
        evals, evecs = np.linalg.eigh(block_ham)
        if block_ham.shape[0] > self.chi:
            middle = block_ham.shape[0]
            evals = evals[middle - self.chi//2:middle + self.chi//2]
            evecs = evecs[:, middle - self.chi//2:middle + self.chi//2]
        return evals, evecs

    def spectrum_projector(self, site: int, evecs: np.ndarray) -> Tuple[Node, Node]:
        W1 = self.mpo.nodes[site]
        W2 = self.mpo.nodes[site + 1]
        V = Node(evecs.reshape((W1.tensor.shape[-1], W2.tensor.shape[-1], evecs.shape[1])))
        V_conj = V.copy(conjugate=True)
        if self.N == 2:
            W1[0] ^ W2[0]
            W1[1] ^ V_conj[0]
            W1[2] ^ V[0]
            W2[1] ^ V_conj[1]
            W2[2] ^ V[1]
            W = W1 @ W2 @ V_conj @ V
        elif site == 0:
            W1[0] ^ W2[0]
            W1[1] ^ V_conj[0]
            W1[2] ^ V[0]
            W2[2] ^ V_conj[1]
            W2[3] ^ V[1]
            W = W1 @ W2 @ V_conj @ V
        elif site == self.N - 2:
            W1[1] ^ W2[0]
            W1[2] ^ V_conj[0]
            W1[3] ^ V[0]
            W2[1] ^ V_conj[1]
            W2[2] ^ V[1]
            W = W1 @ W2 @ V_conj @ V
        else:
            W1[1] ^ W2[0]
            W1[2] ^ V_conj[0]
            W1[3] ^ V[0]
            W2[2] ^ V_conj[1]
            W2[3] ^ V[1]
            W = W1 @ W2 @ V_conj @ V
        return V, W

    def neighbouring_bonds(self, bond: int) -> List[int]:
        if bond == 0:
            neighbours = [bond]
        elif bond == self.N - 1:
            neighbours = [bond - 1]
        else:
            neighbours = [bond - 1, bond]
        return neighbours

    def run(self) -> None:
        if not self.gap_cache.gap:
            for bond in range(self.N - 1):
                evals, evecs = self.eigen_solver(bond)
                _, gap = self.largest_gap(evals)
                self.gap_cache.gap.append(gap)
                self.gap_cache.evecs.append(evecs)

        for step in count():
            max_gapped_bond = np.argmax(np.array(self.gap_cache.gap))
            logging.info(f"step {step}, merging bond {max_gapped_bond}/{self.N}")
            logging.info(f"entanglement entropy {self.entanglement_rendering(self.gap_cache.evecs[max_gapped_bond]).mean()}")
            V, W = self.spectrum_projector(max_gapped_bond, self.gap_cache.evecs[max_gapped_bond])
            self._tree[max_gapped_bond] = TreeNode(
                id=max_gapped_bond,
                node=V,
                gap=self.gap_cache.gap[max_gapped_bond],
                left=self.tree[max_gapped_bond],
                right=self.tree[max_gapped_bond + 1]
            )
            self._tree.pop(max_gapped_bond + 1)
            self.mpo.nodes[max_gapped_bond] = W
            self.mpo.nodes.pop(max_gapped_bond + 1)
            self.gap_cache.gap.pop(max_gapped_bond)
            self.gap_cache.evecs.pop(max_gapped_bond)
            if self.N == 1:
                break
            for bond in self.neighbouring_bonds(max_gapped_bond):
                evals, evecs = self.eigen_solver(bond)
                _, gap = self.largest_gap(evals)
                self.gap_cache.gap[bond] = gap
                self.gap_cache.evecs[bond] = evecs
