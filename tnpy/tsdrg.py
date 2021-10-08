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
    """
    The node object in binary tree.

    Attributes:
        id:
        node:
        gap:
        left:
        right:

    Notes:
        This class differs from tensornetwork.Node,
        which refers to the node in a tensor network.
        Please be aware of this bad naming.
    """
    id: int
    node: Node
    gap: float = None
    left: 'TreeNode' = None
    right: 'TreeNode' = None


@dataclass
class GapCache:
    gap: List[float] = field(default_factory=list)
    evecs: List[np.ndarray] = field(default_factory=list)


class TSDRG:

    def __init__(self, mpo: MPO, chi: int):
        self.mpo = mpo
        self._chi = chi
        self._N = len(mpo.nodes)
        self._tree = [
            TreeNode(id=site, node=self.mpo.nodes[site])
            for site in range(self.n_nodes)
        ]
        self.gap_cache = GapCache()

    @property
    def N(self) -> int:
        return self._N

    @property
    def n_nodes(self) -> int:
        return len(self.mpo.nodes)

    @property
    def chi(self) -> int:
        return self._chi

    @property
    def v_left(self) -> Node:
        return self.mpo.v_left

    @property
    def v_right(self) -> Node:
        return self.mpo.v_right

    @property
    def tree(self) -> List[TreeNode]:
        return self._tree

    def block_hamiltonian(self, site: int) -> Union[Tensor, np.ndarray]:
        """
        Construct the 2-site Hamiltonian from MPO,
        involving both the hopping term and one-site term.

        Args:
            site: The site i.
                The 2-site Hamiltonian is then constructed in site i and i+1.

        Returns:
            M: The 2-site Hamiltonian.
        """
        W1 = self.mpo.nodes[site]
        W2 = self.mpo.nodes[site + 1]
        if self.n_nodes == 2:
            W1[0] ^ W2[0]
            M = W1 @ W2
        elif site == 0:
            W1[0] ^ W2[0]
            W2[1] ^ self.v_right[0]
            M = W1 @ W2 @ self.v_right
        elif site == self.n_nodes - 2:
            self.v_left[0] ^ W1[0]
            W1[1] ^ W2[0]
            M = self.v_left @ W1 @ W2
        else:
            self.v_left[0] ^ W1[0]
            W1[1] ^ W2[0]
            W2[1] ^ self.v_right[0]
            M = self.v_left @ W1 @ W2 @ self.v_right
        M.reorder_axes([0, 2, 1, 3])
        shape = int(np.sqrt(M.tensor.size))
        return M.tensor.reshape(shape, shape)

    def head_node_hamiltonian(self, V: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        V_conj = V.copy(conjugate=True)
        V[0] ^ V_conj[0]
        V[1] ^ V_conj[1]
        M = V @ V_conj
        return M.tensor

    def highest_gap(self, evals: np.ndarray) -> float:
        """
        Return the largest gap in spectrum.

        Args:
            evals: The eigenvalues (energy spectrum).

        Returns:
            gap: The largest gap.
        """
        gaps = np.diff(evals)
        return gaps[self.chi - 1] if gaps.size > self.chi else gaps[-1]

    def entanglement_rendering(self, evecs: np.ndarray) -> np.ndarray:
        def von_neumann_entropy(v: np.ndarray) -> float:
            singular_values = np.linalg.svd(v.reshape(2, -1))[1]
            ss = np.square(singular_values)
            return -1 * np.sum(ss @ np.log(ss))
        return np.array([von_neumann_entropy(v) for v in evecs.T])

    def eigen_solver(self, matrix: Union[Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        evals, evecs = np.linalg.eigh(matrix)
        if matrix.shape[0] > self.chi:
            evals = evals[:self.chi]
            evecs = evecs[:, :self.chi]
        return evals, evecs

    def spectrum_projector(self, site: int, evecs: np.ndarray) -> Tuple[Node, Node]:
        """


        Args:
            site:
            evecs:

        Returns:

        """
        W1 = self.mpo.nodes[site]
        W2 = self.mpo.nodes[site + 1]
        V = Node(evecs.reshape((W1.tensor.shape[-1], W2.tensor.shape[-1], evecs.shape[1])))
        V_conj = V.copy(conjugate=True)
        if self.n_nodes == 2:
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
        elif site == self.n_nodes - 2:
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
        elif bond == self.n_nodes - 1:
            neighbours = [bond - 1]
        else:
            neighbours = [bond - 1, bond]
        return neighbours

    def run(self) -> None:
        """

        Returns:

        """
        # Initializing gap_cache if not exist
        if not self.gap_cache.gap:
            for site in range(self.n_nodes - 1):
                block_ham = self.block_hamiltonian(site)
                evals, evecs = self.eigen_solver(block_ham)
                gap = self.highest_gap(evals)
                self.gap_cache.gap.append(gap)
                self.gap_cache.evecs.append(evecs)

        for step in count(start=1):
            max_gapped_bond = np.argmax(np.array(self.gap_cache.gap))
            logging.info(f"step {step}, merging bond {max_gapped_bond}/{self.n_nodes}")
            V, W = self.spectrum_projector(max_gapped_bond, self.gap_cache.evecs[max_gapped_bond])
            self._tree.append(
                TreeNode(
                    id=self.N + step,
                    node=V,
                    gap=self.gap_cache.gap[max_gapped_bond],
                    left=self.tree[max_gapped_bond],
                    right=self.tree[max_gapped_bond + 1]
                )
            )
            self.mpo.nodes.pop(max_gapped_bond)
            self.mpo.nodes[max_gapped_bond] = W
            self.gap_cache.gap.pop(max_gapped_bond)
            self.gap_cache.evecs.pop(max_gapped_bond)
            if self.n_nodes == 1:
                logging.info('Reach head node of the tree')
                logging.info(f"Obtain ground state energy {evals}")
                assert step == self.N - 1, "step out of range"
                break
            for bond in self.neighbouring_bonds(max_gapped_bond):
                block_ham = self.block_hamiltonian(bond)
                evals, evecs = self.eigen_solver(block_ham)
                logging.info(f"{evals[0]}")
                gap = self.highest_gap(evals)
                self.gap_cache.gap[bond] = gap
                self.gap_cache.evecs[bond] = evecs
