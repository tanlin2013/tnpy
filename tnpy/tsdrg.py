import logging
import numpy as np
from functools import wraps
from itertools import count
from collections import deque
from copy import copy
from tensornetwork import ncon, Node, Tensor
from dataclasses import dataclass, field
from tnpy.operators import MPO
from typing import Tuple, List, Union, Callable, Iterator


@dataclass
class Projection:
    HarmonicRitz: str = 'primme_proj_harmonic'
    RefinedRitz: str = 'primme_proj_refined'
    RayleighRitz: str = 'primme_proj_RR'


@dataclass
class GapCache:
    """
    Helper class for storing the cache of energy gap in tSDRG algorithm.
    """
    gap: List[float] = field(default_factory=list)
    evecs: List[np.ndarray] = field(default_factory=list)


@dataclass
class TreeNode:
    """
    The node object in binary tree.

    Attributes:
        id: ID of this TreeNode.
        node: Container of :class:`~tensornetwork.Node`,
            which is supposed to take the ordering of axes `(1st physical bond, 2nd physical bond, virtual bond)`.
        gap:
        left:
        right:

    Notes:
        This class differs from :class:`~tensornetwork.Node`,
        which refers to the node in a tensor network.
        Please be aware of this bad naming.
    """
    id: int
    node: Node
    # TODO: is it really necessary to keep the gap here. we already have GapCache.
    gap: float = None
    left: 'TreeNode' = None
    right: 'TreeNode' = None

    def __eq__(self, other: 'TreeNode') -> bool:
        """
        Check 2 TreeNodes are identical.

        Args:
            other:

        Returns:

        Warnings:
            This is a weak condition, only the id of these 2 TreeNodes
            and the id of their 1st generation children (if not a leaf) are compared.
            It's not guaranteed that the value of :attr:`~TreeNode.node` will be the same,
            as well as their posterity.
        """
        if self.is_leaf and other.is_leaf:
            return self.id == other.id
        elif self.is_leaf or other.is_leaf:
            return False
        return self.id == other.id and self.left.id == other.left.id and self.right.id == other.right.id

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def shape(self) -> Tuple[int]:
        return self.node.tensor.shape


def check_root(func: Callable) -> Callable:
    """
    Helper function for checking the root in :class:`~TensorTree`.
    This is supposed to be applied on the member function of :class:`~TensorTree` as a decorator.

    Args:
        func:

    Returns:

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert isinstance(args[0], TensorTree), "This is supposed to be used on class member of TensorTree."
        if not args[0].has_root:
            raise RuntimeError("Cannot find root in tree.")
        return func(*args, **kwargs)
    return wrapper


class TensorTree:

    def __init__(self, leaves: List[TreeNode]):
        """
        The bottom-up binary tree where each node contains a tensor.

        Args:
            leaves:
        """
        self._leaves = {leaf.id: leaf for leaf in leaves}
        self._tree = copy(self._leaves)
        self._horizon = list(self._tree.keys())
        self._root = None
        self._top_tensor = None

    def __iter__(self) -> Iterator:
        return iter(self._tree.values())

    @property
    def n_nodes(self) -> int:
        """
        Total number of nodes in tree.

        Returns:
            n_nodes:
        """
        return len(self._tree)

    @property
    def n_layers(self) -> int:
        def max_depth(current_node: TreeNode) -> int:
            if current_node is not None:
                left_depth = max_depth(current_node.left)
                right_depth = max_depth(current_node.right)
                return max(left_depth, right_depth) + 1
            return 0
        return max_depth(self.root)

    @property
    def n_leaves(self) -> int:
        return len(self._leaves)

    @property
    def root(self) -> Union[None, TreeNode]:
        return self._root

    @property
    def has_root(self) -> bool:
        return self.root is not None

    @property
    def horizon(self) -> List[int]:
        return self._horizon

    @property
    def top_tensor(self):
        return self._top_tensor

    @top_tensor.setter
    def top_tensor(self, top_tensor):
        self._top_tensor = top_tensor

    def append(self, node: TreeNode) -> None:
        self._tree[node.id] = node
        self._horizon[self._horizon.index(node.left.id)] = node.id
        self._horizon.remove(node.right.id)
        if self.n_nodes == 2 * self.n_leaves - 1:
            self._root = node

    def node(self, node_id: int) -> TreeNode:
        return self._tree[node_id]

    def leaf(self, node_id: int) -> TreeNode:
        return self._leaves[node_id]

    @check_root
    def find_path(self, node_id: int) -> List[int]:
        """
        Find the path to targeted node starting from the root.

        Args:
            node_id:

        Returns:
            path: ID of nodes to targeted node, including targeted node itself.

        Raises:
            If path is empty, raises KeyError for input ``node_id``.

        See Also:
            The function :py:func:`TensorTree.ancestor` does the same job as this function,
            while the returning type is List[:class:`~TreeNode`] there.
        """
        def find_node(current_node: TreeNode, trial_path: List[int]) -> bool:
            """
            Find the node with depth-first search.

            Args:
                current_node: Current node in iterative depth-first searching
                trial_path: List for recording the trial path to every leaf.

            Returns:
                found: Return True if the given ``node_id`` is found in tree, else False.
            """
            if current_node is not None:
                trial_path.append(current_node.id)
                if current_node.id == node_id:
                    return True
                elif find_node(current_node.left, trial_path) or\
                        find_node(current_node.right, trial_path):
                    return True
                trial_path.pop()
            return False

        path = []
        if not find_node(self.root, path):
            raise KeyError("The given node_id is not found in tree.")
        return path

    def ancestor(self, node_id: int) -> List[TreeNode]:
        """
        Find all the ancestors of targeted node in the order starting from the root.

        Args:
            node_id:

        Returns:
            ancestor:

        See Also:
            The function :func:`~TensorTree.find_path` does the same job as this function,
            while the returning type is List[int] there.
        """
        path = self.find_path(node_id)
        return [self.node(idx) for idx in path][:-1]

    @check_root
    def common_ancestor(self, node_id1: int, node_id2: int) -> List[int]:
        """
        Find all common ancestor of two given nodes in the order starting from the root.

        Args:
            node_id1:
            node_id2:

        Returns:

        """
        path1 = self.find_path(node_id1)
        path2 = self.find_path(node_id2)
        first_unmatched = next(idx for idx, (x, y) in enumerate(zip(path1, path2)) if x != y)
        return path1[:first_unmatched]

    def filter_out_leaf(self, node_ids: List[int]) -> List[int]:
        return [node_id for node_id in node_ids if not self.node(node_id).is_leaf]

    @check_root
    def contract_nodes(self, node_ids: List[int], open_bonds: List[Tuple[int, str]] = None) -> Tensor:
        node_queue = deque([self.root])
        contractible = {
            f"{node_id}": {
                'tensor': self.node(node_id).node.tensor,
                'bonds': [f'-Node{node_id}Sub0', f'-Node{node_id}Sub1', f'-Node{node_id}Sub2']
            } for node_id in node_ids
        }
        contractible.update(
            {
                f"conj{node_id}": {
                    'tensor': self.node(node_id).node.copy().tensor,
                    'bonds': [f'-ConjNode{node_id}Sub0', f'-ConjNode{node_id}Sub1', f'-ConjNode{node_id}Sub2']
                } for node_id in node_ids
            }
        )
        if not all(isinstance(elem, tuple) and list(map(type, elem)) == [int, str] for elem in open_bonds):
            raise TypeError("open_bonds takes the type List[Tuple[ins, str]].")

        def update_contractible(parent: TreeNode, on: str):
            child = getattr(parent, on)
            on_bond_idx = 0 if on == 'left' else 1
            if child.id in node_ids:
                assert not child.is_leaf, "node_ids contains leaves"
                contractible[f"{parent.id}"]['bonds'][on_bond_idx] = f"Node{parent.id}ToNode{child.id}"
                contractible[f"{child.id}"]['bonds'][2] = f"Node{parent.id}ToNode{child.id}"
                contractible[f"conj{parent.id}"]['bonds'][on_bond_idx] = f"ConjNode{parent.id}ToConjNode{child.id}"
                contractible[f"conj{child.id}"]['bonds'][2] = f"ConjNode{parent.id}ToConjNode{child.id}"
                if (child.id, on) not in open_bonds:
                    node_queue.append(child)
            elif (parent.id, on) not in open_bonds:
                contractible[f"conj{parent.id}"]['bonds'][on_bond_idx] = f"Node{parent.id}Sub{on_bond_idx}ToConjNode{parent.id}Sub{on_bond_idx}"
                contractible[f"{parent.id}"]['bonds'][on_bond_idx] = f"Node{parent.id}Sub{on_bond_idx}ToConjNode{parent.id}Sub{on_bond_idx}"

        while len(node_queue) > 0:
            current_node = node_queue.popleft()
            update_contractible(current_node, on='left')
            update_contractible(current_node, on='right')

        target_tensors = [node['tensor'] for node in contractible.values()]
        contract_bonds = [tuple(node['bonds']) for node in contractible.values()]
        return ncon(target_tensors, contract_bonds)

    @check_root
    def plot(self):
        NotImplemented


class TSDRG:

    def __init__(self, mpo: MPO, chi: int):
        self.mpo = mpo
        self._chi = chi
        self._N = len(mpo.nodes)
        self._tree = TensorTree(
            [
                TreeNode(id=site, node=self.mpo.nodes[site])
                for site in range(self.n_nodes)
            ]
        )
        self.gap_cache = GapCache()
        self._init_gap_cache()

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
    def tree(self) -> TensorTree:
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

    def truncation_gap(self, evals: np.ndarray) -> float:
        """
        Return the gap upon :attr:`~TSDRG.chi` eigenvalues kept.

        Args:
            evals: The eigenvalues (energy spectrum).

        Returns:
            gap: The truncation gap, evals[chi+1] - evals[chi].
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
        elif site == 0:
            W1[0] ^ W2[0]
            W1[1] ^ V_conj[0]
            W1[2] ^ V[0]
            W2[2] ^ V_conj[1]
            W2[3] ^ V[1]
        elif site == self.n_nodes - 2:
            W1[1] ^ W2[0]
            W1[2] ^ V_conj[0]
            W1[3] ^ V[0]
            W2[1] ^ V_conj[1]
            W2[2] ^ V[1]
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

    def _init_gap_cache(self) -> None:
        for site in range(self.n_nodes - 1):
            evals, evecs = self.eigen_solver(self.block_hamiltonian(site))
            self.gap_cache.gap.append(self.truncation_gap(evals))
            self.gap_cache.evecs.append(evecs)

    def run(self) -> None:
        """

        Returns:

        """
        for step in count(start=1):
            max_gapped_bond = np.argmax(np.array(self.gap_cache.gap))
            V, W = self.spectrum_projector(max_gapped_bond, self.gap_cache.evecs[max_gapped_bond])
            horizon = self.tree.horizon
            logging.info(f"step {step}, merging TreeNode({horizon[max_gapped_bond]}) "
                         f"and TreeNode({horizon[max_gapped_bond + 1]}) to TreeNode({self.N + step})")
            self._tree.append(
                TreeNode(
                    id=self.N + step,
                    node=V,
                    gap=self.gap_cache.gap[max_gapped_bond],
                    left=self.tree.node(horizon[max_gapped_bond]),
                    right=self.tree.node(horizon[max_gapped_bond + 1])
                )
            )
            self.mpo.nodes.pop(max_gapped_bond)
            self.mpo.nodes[max_gapped_bond] = W
            self.gap_cache.gap.pop(max_gapped_bond)
            self.gap_cache.evecs.pop(max_gapped_bond)
            if self.n_nodes == 1:
                logging.info('Reach head node of the tree')
                # TODO: [Bug] evals may not be assigned
                logging.info(f"Obtain ground state energies {evals}")
                assert step == self.N - 1, "step out of range"
                break
            for bond in self.neighbouring_bonds(max_gapped_bond):
                evals, evecs = self.eigen_solver(self.block_hamiltonian(bond))
                self.gap_cache.gap[bond] = self.truncation_gap(evals)
                self.gap_cache.evecs[bond] = evecs
