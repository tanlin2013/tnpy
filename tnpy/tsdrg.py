import re
import numpy as np
import quimb.tensor as qtn
from dataclasses import dataclass, field
from functools import wraps
from itertools import count, islice
from typing import Union, List, Dict, Iterator, Callable, Sequence, Tuple
from graphviz import Digraph
from tnpy import logger
from tnpy.operators import MatrixProductOperator


class Node(qtn.Tensor):

    def __init__(self, node_id: int, left: 'Node' = None, right: 'Node' = None,
                 *args, **kwargs):
        """
        The node of binary tree, while data structure is inherited from :class:`~quimb.Tensor`.

        Args:
            node_id:
            left:
            right:
            *args:
            **kwargs:
        """
        super(Node, self).__init__(*args, **kwargs)
        self._node_id = node_id
        self._left = left
        self._right = right

    def __repr__(self):
        left_node_id = None if self.left is None else self.left.node_id
        right_node_id = None if self.right is None else self.right.node_id
        return f'Node(id={self.node_id}, ' \
               f'left_id={left_node_id}, ' \
               f'right_id={right_node_id}, ' \
               f'shape={tuple(map(int, self.data.shape))}, ' \
               f'inds={self.inds})'

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def left(self) -> 'Node':
        return self._left

    @property
    def right(self) -> 'Node':
        return self._right

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class TensorTree:

    @dataclass
    class Syntax:
        node: str = 'Node'
        conj_node: str = 'ConjNode'
        level_idx: str = 'LevelIdx'

    def __init__(self, mpo: MatrixProductOperator):
        """
        The bottom-up binary tree where each node contains a tensor.

        Args:
            mpo:
        """
        self._root_id = None
        self._tree = {
            site: Node(
                node_id=site,
                data=tensor.data,
                inds=tensor.inds,
                tags=tensor.tags
            )
            for site, tensor in enumerate(mpo)
        }
        self._n_leaves = mpo.nsites
        self._horizon = list(range(self._n_leaves))

    def __iter__(self) -> Iterator:
        return iter(self._tree.values())

    def __getitem__(self, node_id: int) -> Node:
        return self._tree[node_id]

    def __setitem__(self, node_id: int, node: Node):
        self._tree[node_id] = node

    def __repr__(self):
        return 'TensorTree([\n\t{}\n])'.format(', \n\t'.join(repr(node) for node in self._tree.values()))

    @property
    def root_id(self) -> int:
        return self._root_id

    @property
    def root(self) -> Node:
        return self[self._root_id]

    @property
    def leaves(self) -> Dict[int, Node]:
        return dict(islice(self._tree.items(), self._n_leaves))

    @property
    def horizon(self) -> List[int]:
        return self._horizon

    @property
    def n_nodes(self) -> int:
        """
        Total number of nodes in tree.

        Returns:
            n_nodes:
        """
        return len(self._tree)

    @property
    def n_leaves(self) -> int:
        return self._n_leaves

    def fuse(self, left_id: int, right_id: int, new_id: int, data: np.ndarray):
        if self.root_id is not None:
            raise RuntimeError("Can't perform fuse on a grown tree.")
        left_ind = f'{self.Syntax.node}{new_id}-{self.Syntax.node}{left_id}' \
            if not self[left_id].is_leaf else f'k{left_id}'
        right_ind = f'{self.Syntax.node}{new_id}-{self.Syntax.node}{right_id}' \
            if not self[right_id].is_leaf else f'k{right_id}'
        self[new_id] = Node(
            node_id=new_id,
            left=self[left_id],
            right=self[right_id],
            data=data,
            inds=(left_ind, right_ind, f'{self.Syntax.node}{new_id}{self.Syntax.level_idx}'),
            tags=f'{self.Syntax.node}{new_id}'
        )
        self[left_id].reindex(
            {f'{self.Syntax.node}{left_id}{self.Syntax.level_idx}':
                f'{self.Syntax.node}{new_id}-{self.Syntax.node}{left_id}'}, inplace=True
        )
        self[right_id].reindex(
            {f'{self.Syntax.node}{right_id}{self.Syntax.level_idx}':
                f'{self.Syntax.node}{new_id}-{self.Syntax.node}{right_id}'}, inplace=True
        )
        self._horizon[self._horizon.index(left_id)] = new_id
        self._horizon.remove(right_id)
        if self.n_nodes == 2 * self.n_leaves - 1:
            self._root_id = new_id

    def check_root(func: Callable) -> Callable:
        """
        Accessory decorator for checking the existence of root in :class:`~TensorTree`.

        Returns:

        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._root_id is None:
                raise KeyError("Can't find root in tree.")
            return func(self, *args, **kwargs)
        return wrapper

    @property
    @check_root
    def n_layers(self) -> int:
        def max_depth(current_node: Node) -> int:
            if current_node is not None:
                left_depth = max_depth(current_node.left)
                right_depth = max_depth(current_node.right)
                return max(left_depth, right_depth) + 1
            return 0
        return max_depth(self.root)

    @check_root
    def tensor_network(self, node_ids: Sequence[int] = None, conj: bool = False,
                       mangle_outer: bool = True, with_leaves: bool = False) -> qtn.TensorNetwork:
        node_ids = list(self._tree.keys()) if node_ids is None else node_ids
        nodes = [self[node_id] for node_id in node_ids if not self[node_id].is_leaf] if not with_leaves \
            else [self[node_id] for node_id in node_ids]
        net = qtn.TensorNetwork(nodes)
        if conj:
            conj_net = net.reindex(
                {f'{self.Syntax.node}{self._root_id}{self.Syntax.level_idx}':
                    f'{self.Syntax.conj_node}{self._root_id}{self.Syntax.level_idx}'}
            )
            conj_net = conj_net.retag(
                {tag: tag.replace(self.Syntax.node, self.Syntax.conj_node)
                 for node in conj_net for tag in node.tags}
            )
            if mangle_outer:
                conj_net = conj_net.reindex(
                    {ind: re.sub(r'^k([0-9]+)', r'b\1', ind)
                     for node in conj_net for ind in node.inds if ind.startswith('k')}
                )
            return conj_net
        return net

    @check_root
    def find_path(self, node_id: int, return_itself: bool = False) -> List[int]:
        """
        Find the path to targeted node starting from the root.

        Args:
            node_id:
            return_itself:

        Returns:
            path: ID of nodes to targeted node, including targeted node itself.

        Raises:
            If path is empty, raises KeyError for input ``node_id``.
        """
        def find_node(current_node: Node, trial_path: List[int]) -> bool:
            """
            Find the node with depth-first search.

            Args:
                current_node: Current node in iterative depth-first searching
                trial_path: List for recording the trial path to the targeted leaf.

            Returns:
                found: Return True if the given ``node_id`` is found in tree, else False.
            """
            if current_node is not None:
                trial_path.append(current_node.node_id)
                if current_node.node_id == node_id:
                    return True
                elif find_node(current_node.left, trial_path) or \
                        find_node(current_node.right, trial_path):
                    return True
                trial_path.pop()
            return False

        path = []
        if not find_node(self.root, path):
            raise KeyError("The given node_id is not found in tree.")
        return path if return_itself else path[:-1]

    @check_root
    def common_ancestor(self, node_id1: int, node_id2: int, lowest: bool = False) -> Union[int, List[int]]:
        """
        Find all common ancestor of two given nodes in the order starting from the root.

        Args:
            node_id1:
            node_id2:
            lowest:

        Returns:

        """
        path1 = self.find_path(node_id1, return_itself=True)
        path2 = self.find_path(node_id2, return_itself=True)
        try:
            first_unmatched = next(idx for idx, (x, y) in enumerate(zip(path1, path2)) if x != y)
            common_path = path1[:first_unmatched]
        except StopIteration:
            assert path1 == path2
            common_path = path1[:-1]
        return common_path if not lowest else common_path[-1]

    @check_root
    def plot(self, view: bool = False) -> Digraph:
        def find_child(node: Node):
            for attr in ['left', 'right']:
                if getattr(node, attr) is not None:
                    if not getattr(node, attr).is_leaf:
                        graph.node(f'{getattr(node, attr).node_id}', shape='triangle', style='rounded')
                    graph.edge(
                        f'{node.node_id}', f'{getattr(node, attr).node_id}',
                        splines='ortho', minlen='2',
                        headport='n', tailport='_', arrowhead='inv', constraint='true'
                    )
                    find_child(getattr(node, attr))

        graph = Digraph()
        graph.node('head', style='invis')
        graph.edge('head', f'{self.root.node_id}', headport='n', arrowhead="inv")
        graph.node(f'{self.root.node_id}', shape='triangle', rank='max', style='rounded')
        find_child(self.root)
        with graph.subgraph(name='cluster_0') as sg:  # for constraining mpo inside an invisible box
            sg.attr(style='invis')
            for k, v in enumerate(self.leaves.keys()):
                sg.node(f'{v}', shape='box', rank='sink', style='rounded')
                if k < len(self.leaves.keys()) - 1:
                    sg.edge(f'{v}', f'{v + 1}', splines='ortho', minlen='0', arrowhead="none", constraint='true')
        logger.debug(graph)
        if view:
            graph.render(format='png', view=True)
        return graph


class TreeTensorNetworkSDRG:

    @dataclass
    class GapCache:
        """
        Helper class for caching the energy gap in tSDRG algorithm.
        """
        tsdrg: 'TreeTensorNetworkSDRG'
        evecs: List[np.ndarray] = field(default_factory=list)
        gap: List[float] = field(default_factory=list)

        def __post_init__(self):
            for site in range(self.tsdrg.n_sites - 1):
                evals, evecs = self.tsdrg.eigen_solver(self.tsdrg.block_hamiltonian(site))
                self.evecs.append(evecs)
                self.gap.append(self.tsdrg.truncation_gap(evals))
                self.tsdrg._evals = evals if self.tsdrg.n_sites == 2 else None

        def neighbouring_bonds(self, bond: int) -> List[int]:
            if bond == 0:
                neighbours = [bond]
            elif bond == len(self.tsdrg._fused_mpo_cache) - 1:
                neighbours = [bond - 1]
            else:
                neighbours = [bond - 1, bond]
            return neighbours

        def update(self, max_gapped_bond: int):
            self.gap.pop(max_gapped_bond)
            self.evecs.pop(max_gapped_bond)
            for bond in self.neighbouring_bonds(max_gapped_bond):
                evals, evecs = self.tsdrg.eigen_solver(self.tsdrg.block_hamiltonian(bond))
                self.gap[bond] = self.tsdrg.truncation_gap(evals)
                self.evecs[bond] = evecs
                self.tsdrg._evals = evals

    def __init__(self, mpo: MatrixProductOperator, chi: int):
        self._mpo = mpo
        self._chi = chi
        self._tree = TensorTree(mpo)
        self._fused_mpo_cache = [tensor for tensor in mpo]
        self._gap_cache = self.GapCache(self)
        self._evals = None

    @property
    def mpo(self) -> MatrixProductOperator:
        return self._mpo

    @property
    def chi(self) -> int:
        return self._chi

    @property
    def tree(self) -> TensorTree:
        return self._tree

    @property
    def n_sites(self) -> int:
        return self.mpo.nsites

    @property
    def evals(self) -> np.ndarray:
        return self._evals

    def eigen_solver(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: perhaps consider iterative solver?
        evals, evecs = np.linalg.eigh(matrix)
        if matrix.shape[0] > self.chi:
            evals = evals[:self.chi]
            evecs = evecs[:, :self.chi]
        return evals, evecs

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

    def block_hamiltonian(self, locus: int) -> np.ndarray:
        """
        Construct the 2-site Hamiltonian from coarse-graining MPO.

        Args:
            locus: The site in coarse-grained system during the RG process.

        Returns:
            block_ham: The 2-site Hamiltonian.
        """
        mpo1, mpo2 = self._fused_mpo_cache[locus], self._fused_mpo_cache[locus + 1]
        if len(mpo1.inds) == 4:
            mpo1 = mpo1.isel({mpo1.inds[0]: 0})
        if len(mpo2.inds) == 4:
            mpo2 = mpo2.isel({mpo2.inds[1]: -1})
        ham = mpo1 @ mpo2
        ham.fuse({'0': [ham.inds[0], ham.inds[2]], '1': [ham.inds[1], ham.inds[3]]}, inplace=True)
        return ham.data

    def spectrum_projector(self, locus: int, evecs: np.ndarray) -> np.ndarray:
        """
        Coarse-grain the MPO with given evecs which is used to form a projector.

        Args:
            locus: The site in coarse-grained system during the RG process.
            evecs:

        Returns:
            projector:
        """
        mpo1, mpo2 = self._fused_mpo_cache[locus], self._fused_mpo_cache[locus + 1]
        projector = qtn.Tensor(
            evecs.reshape((mpo1.shape[-1], mpo2.shape[-1], evecs.shape[1])),
            inds=(mpo1.inds[-2], mpo2.inds[-2], '-1')
        )
        self._fused_mpo_cache.pop(locus + 1)
        rand_id = qtn.rand_uuid()
        self._fused_mpo_cache[locus] = qtn.TensorNetwork([
            mpo1,
            mpo2,
            projector,
            projector.copy().reindex({
                mpo1.inds[-2]: mpo1.inds[-1],
                mpo2.inds[-2]: mpo2.inds[-1],
                '-1': '-2'
            })
        ]).contract().reindex({'-1': f'k{rand_id}', '-2': f'b{rand_id}'})
        return projector.data

    def run(self):
        for step in count(start=1):
            max_gapped_bond = int(np.argmax(np.array(self._gap_cache.gap)))
            logger.info(f"On step {step}, "
                        f"merging Node({self.tree.horizon[max_gapped_bond]}) "
                        f"and Node({self.tree.horizon[max_gapped_bond + 1]}) "
                        f"into Node({self.n_sites + step - 1}).")
            self.tree.fuse(
                left_id=self.tree.horizon[max_gapped_bond],
                right_id=self.tree.horizon[max_gapped_bond + 1],
                new_id=self.n_sites + step - 1,
                data=self.spectrum_projector(max_gapped_bond, self._gap_cache.evecs[max_gapped_bond])
            )
            if len(self._fused_mpo_cache) == 1:
                logger.info('Reaching the root in tree.')
                logger.info(f"Obtain ground state energies, {repr(self.evals)}.")
                assert step == self.n_sites - 1, "step is out of range."
                break
            self._gap_cache.update(max_gapped_bond)

    @property
    def measurements(self) -> 'TreeTensorNetworkMeasurements':
        if not len(self._fused_mpo_cache) == 1:
            raise RuntimeError("tSDRG algorithm hasn't been executed yet.")
        return TreeTensorNetworkMeasurements(self._tree)


class TreeTensorNetworkMeasurements:

    def __init__(self, tree: TensorTree):
        self._tree = tree

    @property
    def tree(self):
        return self._tree

    def loop_simplify(self, bra: qtn.TensorNetwork, ket: qtn.TensorNetwork,
                      mpo: MatrixProductOperator = None) -> qtn.Tensor:
        net = (bra & ket) if mpo is None else (bra & mpo & ket)
        if mpo is not None:
            for node_id in range(self.tree.n_leaves, self.tree.n_nodes):
                net = net.contract([
                    f"{TensorTree.Syntax.node}{node_id}",
                    list(self.tree[node_id].left.tags)[0],
                    list(self.tree[node_id].right.tags)[0],
                    f"{TensorTree.Syntax.conj_node}{node_id}"
                ])
        else:
            net = net.contract(all)
        return net

    def sandwich(self, mpo: MatrixProductOperator = None) -> qtn.Tensor:
        ket = self.tree.tensor_network()
        bra = self.tree.tensor_network(conj=True, mangle_outer=False) if mpo is None \
            else self.tree.tensor_network(conj=True)
        return self.loop_simplify(bra, ket, mpo)

    def expectation_value(self, mpo: MatrixProductOperator) -> np.ndarray:
        exp_val = self.sandwich(mpo).data
        if not np.allclose(np.zeros(exp_val.shape), exp_val - np.diagflat(np.diag(exp_val)), atol=1e-12):
            logger.warning("Expectation value may contain large off-diagonal elements.")
        return np.diag(exp_val)

    def _min_surface(self, bipartite_site: int) -> Tuple[int, Dict, Dict]:
        min_side = 0 if (bipartite_site + 1) / (self.tree.n_leaves - bipartite_site) < 1 else 1
        iterator = range(bipartite_site + 1) if min_side == 0 \
            else range(bipartite_site + 1, self.tree.n_leaves)
        bipartite_site = bipartite_site + 1 if min_side == 0 else bipartite_site
        ket_inds_map, bra_inds_map = {}, {}
        for leaf_id in iterator:
            lca_id = self.tree.common_ancestor(leaf_id, bipartite_site, lowest=True)
            ket_inds_map[f'{self.tree[lca_id].inds[min_side]}'] = f'rho_k{leaf_id}'
            bra_inds_map[f'{self.tree[lca_id].inds[min_side]}'] = f'rho_b{leaf_id}'
        return min_side, ket_inds_map, bra_inds_map

    def reduced_density_matrix(self, site: int, level_idx: int) -> np.ndarray:
        if not 0 <= site < self.tree.n_leaves - 1:
            raise ValueError("Parameter `site` for bi-partition has to be within the system size.")
        if not 0 <= level_idx < self.tree.root.shape[2]:
            raise ValueError("Parameter `level_idx` has to be lower than truncation dimension.")
        min_side, on_min_ket_surface, on_min_bra_surface = self._min_surface(site)
        to_site = site + 1 if min_side == 0 else site
        node_ids = self.tree.find_path(to_site)
        ket = self.tree.tensor_network(node_ids).reindex(on_min_ket_surface)
        ket.isel(
            {f'{TensorTree.Syntax.node}{self.tree.root_id}{TensorTree.Syntax.level_idx}': level_idx},
            inplace=True
        )
        bra = self.tree.tensor_network(node_ids, conj=True, mangle_outer=False).reindex(on_min_bra_surface)
        bra.isel(
            {f'{TensorTree.Syntax.conj_node}{self.tree.root_id}{TensorTree.Syntax.level_idx}': level_idx},
            inplace=True
        )
        net = (ket & bra).contract(
            [elem for node_id in node_ids
             for elem in (f'{TensorTree.Syntax.node}{node_id}', f'{TensorTree.Syntax.conj_node}{node_id}')]
        )
        return net.to_dense([*on_min_ket_surface.values()], [*on_min_bra_surface.values()])

    def entanglement_entropy(self, site: int, level_idx: int, nan_to_num: bool = False) -> float:
        rho = np.linalg.eigvalsh(self.reduced_density_matrix(site, level_idx))[::-1]
        entropy = -1 * rho @ np.log(rho)
        return np.nan_to_num(entropy) if nan_to_num else entropy

    def one_point_function(self):
        NotImplemented

    def two_point_function(self):
        NotImplemented
