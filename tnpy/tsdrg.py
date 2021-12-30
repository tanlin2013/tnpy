import logging
import numpy as np
from functools import wraps
from itertools import count
from collections import deque
from copy import copy
from graphviz import Digraph
from tensornetwork import ncon, Node, Tensor
from dataclasses import dataclass, field
from tnpy.operators import MPO
from typing import Tuple, List, Union, Callable, Iterator, Dict


@dataclass
class TreeNode:
    """
    The node object in binary tree.

    Attributes:
        id: ID of this TreeNode.
        node: Container of :class:`~tensornetwork.Node`,
            which is supposed to take the ordering of axes `(1st physical bond, 2nd physical bond, virtual bond)`.
        left:
        right:

    Notes:
        This class differs from :class:`~tensornetwork.Node`,
        which refers to the node in a tensor network.
        Please be aware of this bad naming.
    """
    id: int
    node: Node
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

    def __iter__(self) -> Iterator:
        return iter(self._tree.values())

    @property
    def keys(self) -> List:
        return list(self._tree.keys())

    @property
    def leaves(self) -> Dict:
        return self._leaves

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
        try:
            first_unmatched = next(idx for idx, (x, y) in enumerate(zip(path1, path2)) if x != y)
            return path1[:first_unmatched]
        except StopIteration:
            assert path1 == path2
            return path1[:-1]

    def lowest_common_ancestor(self, node_id1: int, node_id2: int) -> int:
        return self.common_ancestor(node_id1, node_id2)[-1]

    def filter_out_leaf(self, node_ids: List[int]) -> List[int]:
        return [node_id for node_id in node_ids if not self.node(node_id).is_leaf]

    class Contractible:
        """
        Accessory helper class for contracting TreeNodes.
        """

        def __init__(self, parent_cls: 'TensorTree', node_ids: List[int], open_bonds: List[Tuple[int, str]]):
            self.parent_cls = parent_cls
            self.open_bonds = open_bonds
            self.input_checker(node_ids, open_bonds)
            self._contractible = {
                k: v for node_id in node_ids for k, v in zip(
                    [f"{node_id}", f"conj{node_id}"],
                    [self.cell(node_id), self.cell(node_id, conj=True)]
                )
            }

        def __getitem__(self, node_id: str):
            return self._contractible[node_id]['bonds']

        @property
        def tensors(self) -> List[Tensor]:
            return [node['tensor'] for node in self._contractible.values()]

        @property
        def network_structure(self) -> List[Tuple[str, str, str]]:
            return [tuple(node['bonds']) for node in self._contractible.values()]

        @property
        def out_order(self) -> List[str]:
            edge_bonds = [self.syntax(*open_bond) for open_bond in self.open_bonds]
            conj_edge_bonds = [self.syntax(*open_bond, conj=True) for open_bond in self.open_bonds]
            return [self.syntax(self.parent_cls.root.id, 2)] + edge_bonds +\
                   [self.syntax(self.parent_cls.root.id, 2, conj=True)] + conj_edge_bonds

        @staticmethod
        def syntax(node_id: int, idx: Union[int, str], conj: bool = False) -> str:
            if type(idx) is str:
                idx = 0 if idx == 'left' else 1
            prefix = "Conj" if conj else ""
            return f"-{prefix}Node{node_id}Sub{idx}"

        def cell(self, node_id: int, conj: bool = False) -> Dict:
            def _tensor_setter() -> Tensor:
                node = self.parent_cls.node(node_id).node
                return node.copy().tensor if conj else node.tensor

            def _bonds_setter() -> List[str]:
                return [self.syntax(node_id, idx, conj) for idx in range(3)]
            return {'tensor': _tensor_setter(), 'bonds': _bonds_setter()}

        def input_checker(self, node_ids: List[int], open_bonds: List[Tuple[int, str]]):
            if len(node_ids) > len(set(node_ids)):
                raise KeyError("Every id in node_ids must be given uniquely.")
            for elem in open_bonds:
                if not isinstance(elem, tuple) and list(map(type, elem)) == [int, str]:
                    raise TypeError("Argument `open_bonds` takes the type List[Tuple[int, str]].")
                if elem[1] not in ['left', 'right']:
                    raise KeyError("Second argument of `open_bonds` must be either `left` or `right`.")
                if getattr(self.parent_cls.node(elem[0]), elem[1]).id in node_ids:
                    raise KeyError("Child node of `open_bonds` cannot present in `node_ids`.")

    @check_root
    def contract_nodes(self, node_ids: List[int], open_bonds: List[Tuple[int, str]] = None,
                       return_out_order: bool = False) -> Union[Tensor, Tuple[Tensor, List[str]]]:
        """

        Args:
            node_ids:
            open_bonds:
            return_out_order:

        Returns:
            contracted_network:
            out_order:
        """
        if open_bonds is None:
            open_bonds = []
        node_queue = deque([self.root])
        contractible = self.Contractible(self, node_ids, open_bonds)

        def update_contractible(parent: TreeNode, on: str):
            child = getattr(parent, on)
            on_bond_idx = 0 if on == 'left' else 1
            if child.id in node_ids:
                assert not child.is_leaf, "node_ids contains leaves"
                contractible[f"{parent.id}"][on_bond_idx] = f"Node{parent.id}ToNode{child.id}"
                contractible[f"{child.id}"][2] = f"Node{parent.id}ToNode{child.id}"
                contractible[f"conj{parent.id}"][on_bond_idx] = f"ConjNode{parent.id}ToConjNode{child.id}"
                contractible[f"conj{child.id}"][2] = f"ConjNode{parent.id}ToConjNode{child.id}"
                if (child.id, 'left') or (child.id, 'right') not in open_bonds:
                    node_queue.append(child)
            elif (parent.id, on) not in open_bonds:
                contractible[f"conj{parent.id}"][on_bond_idx] = f"Node{parent.id}Sub{on_bond_idx}" \
                                                                f"ToConjNode{parent.id}Sub{on_bond_idx}"
                contractible[f"{parent.id}"][on_bond_idx] = f"Node{parent.id}Sub{on_bond_idx}" \
                                                            f"ToConjNode{parent.id}Sub{on_bond_idx}"

        while len(node_queue) > 0:
            current_node = node_queue.popleft()
            update_contractible(current_node, on='left')
            update_contractible(current_node, on='right')

        # TODO: con_order should be given
        out_tensor = ncon(
            contractible.tensors,
            contractible.network_structure,
            out_order=contractible.out_order
        )
        if return_out_order:
            return out_tensor, contractible.out_order
        return out_tensor

    @check_root
    def plot(self) -> Digraph:
        def find_child(node: TreeNode):
            for attr in ['left', 'right']:
                if getattr(node, attr) is not None:
                    if not getattr(node, attr).is_leaf:
                        graph.node(f'{getattr(node, attr).id}', shape='triangle', style='rounded')
                    graph.edge(
                        f'{node.id}', f'{getattr(node, attr).id}',
                        splines='ortho', minlen='2',
                        headport='n', tailport='_', arrowhead='inv', constraint='true'
                    )
                    find_child(getattr(node, attr))

        graph = Digraph()
        graph.node('head', style='invis')
        graph.edge('head', f'{self.root.id}', headport='n', arrowhead="inv")
        graph.node(f'{self.root.id}', shape='triangle', rank='max', style='rounded')
        find_child(self.root)
        with graph.subgraph(name='cluster_0') as sg:
            sg.attr(style='invis')
            for k, v in enumerate(self._leaves.keys()):
                sg.node(f'{v}', shape='box', rank='sink', style='rounded')
                if k < len(self._leaves.keys()) - 1:
                    sg.edge(f'{v}', f'{v+1}', splines='ortho', minlen='0', arrowhead="none", constraint='true')
        logging.debug(graph)
        return graph


class TSDRG:

    @dataclass
    class GapCache:
        """
        Helper class for storing the cache of energy gap in tSDRG algorithm.
        """
        parent_cls: 'TSDRG'
        gap: List[float] = field(default_factory=list)
        evecs: List[np.ndarray] = field(default_factory=list)
        evals: np.ndarray = None

        def __post_init__(self):
            for site in range(self.parent_cls.n_nodes - 1):
                evals, evecs = self.parent_cls.eigen_solver(self.parent_cls.block_hamiltonian(site))
                self.gap.append(self.parent_cls.truncation_gap(evals))
                self.evecs.append(evecs)
                if self.parent_cls.n_nodes == 2:
                    self.evals = evals

    def __init__(self, mpo: MPO, chi: int):
        self.mpo = mpo
        self._chi = chi
        self._N = len(mpo.nodes)
        self._physical_dimensions = mpo.physical_dimensions
        assert self._N >= 2, "There must be more than 2 sites by definition."
        self._tree = TensorTree(
            [
                TreeNode(id=site, node=node)
                for site, node in enumerate(self.mpo)
            ]
        )
        self.gap_cache = self.GapCache(self)

    @property
    def N(self) -> int:
        return self._N

    @property
    def physical_dimensions(self) -> int:
        return self._physical_dimensions

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
        W1 = self.mpo[site]
        W2 = self.mpo[site + 1]
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
            V: The isometric tensor
            W: The coarse-grained MPO
        """
        def network_structure() -> List:
            if self.n_nodes == 2:
                return [(1, 'a1', 'a2'), (1, 'b1', 'b2'), ('a1', 'b1', -3), ('a2', 'b2', -4)]
            elif site == 0:
                return [(1, 'a1', 'a2'), (1, -2, 'b1', 'b2'), ('a1', 'b1', -3), ('a2', 'b2', -4)]
            elif site == self.n_nodes - 2:
                return [(-1, 1, 'a1', 'a2'), (1, 'b1', 'b2'), ('a1', 'b1', -3), ('a2', 'b2', -4)]
            return [(-1, 1, 'a1', 'a2'), (1, -2, 'b1', 'b2'), ('a1', 'b1', -3), ('a2', 'b2', -4)]
        W1 = self.mpo[site].tensor
        W2 = self.mpo[site + 1].tensor
        V = Node(evecs.reshape((W1.shape[-1], W2.shape[-1], evecs.shape[1])))
        W = Node(ncon(
            [W1, W2, V.copy(conjugate=True).tensor, V.tensor],
            network_structure()
        ))
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
        evals = self.gap_cache.evals if self.n_nodes == 2 else None
        for step in count(start=1):
            max_gapped_bond = np.argmax(np.array(self.gap_cache.gap))
            V, W = self.spectrum_projector(max_gapped_bond, self.gap_cache.evecs[max_gapped_bond])
            horizon = self.tree.horizon
            logging.info(f"step {step}, merging TreeNode({horizon[max_gapped_bond]}) "
                         f"and TreeNode({horizon[max_gapped_bond + 1]}) to TreeNode({self.N + step - 1})")
            self._tree.append(
                TreeNode(
                    id=self.N + step - 1,
                    node=V,
                    left=self.tree.node(horizon[max_gapped_bond]),
                    right=self.tree.node(horizon[max_gapped_bond + 1])
                )
            )
            self.mpo.nodes.pop(max_gapped_bond)
            self.mpo.nodes[max_gapped_bond] = W
            self.gap_cache.gap.pop(max_gapped_bond)
            self.gap_cache.evecs.pop(max_gapped_bond)
            if self.n_nodes == 1:
                logging.info('Reach the root in tree')
                logging.info(f"Obtain ground state energies {evals}")
                assert step == self.N - 1, "step out of range"
                break
            for bond in self.neighbouring_bonds(max_gapped_bond):
                evals, evecs = self.eigen_solver(self.block_hamiltonian(bond))
                self.gap_cache.gap[bond] = self.truncation_gap(evals)
                self.gap_cache.evecs[bond] = evecs

    def reduced_density_matrix(self, on_site: int, energy_level: int = 0) -> Union[Tensor, np.ndarray]:
        assert 0 <= on_site < self.N
        assert 0 <= energy_level < self.physical_dimensions ** self.chi

        def descendant_chirality(ancestor: int, descendant: int) -> Tuple[int, str]:
            path = self.tree.find_path(descendant)
            return (ancestor, 'left') if self.tree.node(ancestor).left.id in path else (ancestor, 'right')
        iterator = range(on_site+1) if (on_site + 1) / (self.N - on_site) < 1 else range(on_site, self.N)
        open_bonds = [
            descendant_chirality(
                self.tree.lowest_common_ancestor(node_id, on_site),
                node_id
            ) for node_id in iterator
        ]
        rho = self.tree.contract_nodes(
            self.tree.find_path(on_site)[:-1],
            open_bonds=list(set(open_bonds))
        ).reshape((
            min(self.physical_dimensions ** self.N, self.chi),
            self.physical_dimensions ** min(on_site + 1, self.N - on_site),
            min(self.physical_dimensions ** self.N, self.chi),
            self.physical_dimensions ** min(on_site + 1, self.N - on_site)
        ))
        return rho[energy_level, :, energy_level, :]

    def entanglement_entropy(self, on_site: int, energy_level: int = 0) -> float:
        ss = np.linalg.svd(self.reduced_density_matrix(on_site, energy_level), compute_uv=False)
        return -1 * ss @ np.log(ss)

    def correlation_function(self):
        NotImplemented

    def energies(self, mpo: MPO = None) -> Union[Tensor, np.ndarray]:
        node_queue = deque([self.tree.root])
        node_ids = self.tree.filter_out_leaf(self.tree.keys)
        contractible = self.tree.Contractible(self.tree, node_ids, [])

        def update_contractible(parent: TreeNode, on: str):
            child = getattr(parent, on)
            on_bond_idx = 0 if on == 'left' else 1
            if child.is_leaf:
                contractible[f"{parent.id}"][on_bond_idx] = f"NodeToMPO{child.id}"
                contractible[f"conj{parent.id}"][on_bond_idx] = f"ConjNodeToMPO{child.id}"
            else:
                contractible[f"{parent.id}"][on_bond_idx] = f"Node{parent.id}ToNode{child.id}"
                contractible[f"{child.id}"][2] = f"Node{parent.id}ToNode{child.id}"
                contractible[f"conj{parent.id}"][on_bond_idx] = f"ConjNode{parent.id}ToConjNode{child.id}"
                contractible[f"conj{child.id}"][2] = f"ConjNode{parent.id}ToConjNode{child.id}"
                node_queue.append(child)

        def mpo_network(node_id: int):
            net = (
                f"MPO{node_id - 1}ToMPO{node_id}",
                f"MPO{node_id}ToMPO{node_id + 1}",
                f"NodeToMPO{node_id}",
                f"ConjNodeToMPO{node_id}"
            )
            if node_id == 0:
                return net[1:]
            elif node_id == self.N - 1:
                return (net[0], *net[2:])
            return net

        while len(node_queue) > 0:
            current_node = node_queue.popleft()
            update_contractible(current_node, on='left')
            update_contractible(current_node, on='right')

        mpo_tensors = [node.tensor for node in mpo] if mpo is not None \
            else [tree_node.node.tensor for tree_node in self.tree.leaves.values()]
        mpo_network_structure = [mpo_network(node_id) for node_id in self.tree.leaves.keys()]

        # TODO: con_order should be given
        return ncon(
            contractible.tensors + mpo_tensors,
            contractible.network_structure + mpo_network_structure,
            out_order=contractible.out_order
        )
