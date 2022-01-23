import logging
import re
import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator as MPO
from functools import wraps
from itertools import count, islice
from dataclasses import dataclass, field
from typing import Union, List, Dict, Iterator, Callable, Sequence, Tuple


class Node(qtn.Tensor):

    def __init__(self, node_id: int, left: 'Node' = None, right: 'Node' = None,
                 *args, **kwargs):
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

    def __init__(self, mpo: MPO):
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
        left_ind = f'Node{new_id}-Node{left_id}' if not self[left_id].is_leaf else f'k{left_id}'
        right_ind = f'Node{new_id}-Node{right_id}' if not self[right_id].is_leaf else f'k{right_id}'
        self[new_id] = Node(
            node_id=new_id,
            left=self[left_id],
            right=self[right_id],
            data=data,
            inds=(left_ind, right_ind, f'Node{new_id}LevelIdx'),
            tags=f'Node{new_id}'
        )
        self[left_id].reindex({f'Node{left_id}LevelIdx': f'Node{new_id}-Node{left_id}'}, inplace=True)
        self[right_id].reindex({f'Node{right_id}LevelIdx': f'Node{new_id}-Node{right_id}'}, inplace=True)
        self._horizon[self._horizon.index(left_id)] = new_id
        self._horizon.remove(right_id)
        if self.n_nodes == 2 * self.n_leaves - 1:
            self._root_id = new_id

    def check_root(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._root_id is None:
                raise KeyError("Cannot find root in tree.")
            return func(self, *args, **kwargs)
        return wrapper

    @check_root
    def tensor_network(self, node_ids: Sequence[int] = None, conj: bool = False,
                       mangle_outer: bool = True, with_leaves: bool = False) -> qtn.TensorNetwork:
        node_ids = list(self._tree.keys()) if node_ids is None else node_ids
        nodes = [self[node_id] for node_id in node_ids if not self[node_id].is_leaf] if not with_leaves \
            else [self[node_id] for node_id in node_ids]
        net = qtn.TensorNetwork(nodes)
        if conj:
            conj_net = net.copy(deep=True)
            conj_net.reindex({f'Node{self._root_id}LevelIdx': f'ConjNode{self._root_id}LevelIdx'}, inplace=True)
            conj_net.retag({tag: tag.replace('Node', 'ConjNode')
                            for node in conj_net for tag in node.tags}, inplace=True)
            if mangle_outer:
                conj_net.reindex({ind: re.sub(r'^k([0-9]+)', r'b\1', ind)
                                  for node in conj_net for ind in node.inds
                                  if ind.startswith('k')}, inplace=True)
            return conj_net
        return net

    @check_root
    def find_path(self, node_id: int):
        print('get', node_id)
        return


class TreeTensorNetworkSDRG:

    @dataclass
    class GapCache:
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

    def __init__(self, mpo: MPO, chi: int):
        self._mpo = mpo
        self._chi = chi
        self._tree = TensorTree(mpo)
        self._fused_mpo_cache = [tensor for tensor in mpo]
        self._gap_cache = self.GapCache(self)
        self._evals = None

    @property
    def mpo(self) -> MPO:
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
        evals, evecs = np.linalg.eigh(matrix)
        if matrix.shape[0] > self.chi:
            evals = evals[:self.chi]
            evecs = evecs[:, :self.chi]
        return evals, evecs

    def truncation_gap(self, evals: np.ndarray) -> float:
        gaps = np.diff(evals)
        return gaps[self.chi - 1] if gaps.size > self.chi else gaps[-1]

    def block_hamiltonian(self, locus: int) -> np.ndarray:
        ham = self._fused_mpo_cache[locus] @ self._fused_mpo_cache[locus + 1]
        if len(self._fused_mpo_cache) > 2:
            if locus == 0:
                ham.isel({ham.inds[2]: -1}, inplace=True)
            elif locus == len(self._fused_mpo_cache) - 2:
                ham.isel({ham.inds[0]: 0}, inplace=True)
            else:
                ham.isel({ham.inds[0]: 0, ham.inds[3]: -1}, inplace=True)
        ham.fuse({'0': [ham.inds[0], ham.inds[2]], '1': [ham.inds[1], ham.inds[3]]}, inplace=True)
        return ham.data

    def spectrum_projector(self, locus: int, evecs: np.ndarray) -> np.ndarray:
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
            logging.info(f"On step {step}, "
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
                logging.info('Reaching the root in tree.')
                logging.info(f"Obtain ground state energies, {repr(self.evals)}.")
                assert step == self.n_sites - 1, "step is out of range."
                break
            self._gap_cache.update(max_gapped_bond)

    @property
    def measurements(self) -> 'TreeTensorNetworkMeasurements':
        assert len(self._fused_mpo_cache) == 1, "tSDRG algorithm hasn't been executed yet."
        return TreeTensorNetworkMeasurements(self._tree)


class TreeTensorNetworkMeasurements:

    def __init__(self, tree: TensorTree):
        self._tree = tree

    def sandwich(self, mpo: MPO = None) -> qtn.Tensor:
        bra = self._tree.tensor_network()
        ket = self._tree.tensor_network(conj=True, mangle_outer=False) if mpo is None \
            else self._tree.tensor_network(conj=True)
        net = (bra & ket) if mpo is None else (bra & mpo & ket)
        return net.contract(all)

    def expectation_value(self, mpo: MPO) -> np.ndarray:
        exp_val = TreeTensorNetworkMeasurements(self._tree).sandwich(mpo).data
        assert np.allclose(
            np.zeros(exp_val.shape),
            exp_val - np.diagflat(np.diag(exp_val)),
            atol=1e-12
        ), "Expectation value contains large off-diagonal elements."
        return np.diag(exp_val)

    def reduced_density_matrix(self):
        NotImplemented

    def entanglement_entropy(self):
        NotImplemented

    def one_point_function(self):
        NotImplemented

    def two_point_function(self):
        NotImplemented
