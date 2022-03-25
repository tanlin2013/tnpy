import time
import numpy as np
from quimb.tensor import rand_uuid
from datetime import timedelta
from itertools import cycle
from tnpy import logger
from tnpy.linalg import eigh, eigshmv
from tnpy.matrix_product_state import (
    MatrixProductState,
    Environment,
    Direction,
    MatrixProductStateMeasurements
)
from tnpy.operators import MatrixProductOperator
from typing import Tuple, List
from enum import Enum


class StoppingCriterion(Enum):
    """
    Enumeration for the stopping criterion of sweeping procedure.
    """
    by_energy = 1
    by_variance = 2


class FiniteDMRG:

    def __init__(self, mpo: MatrixProductOperator,
                 bond_dim: int,
                 block_size: int = 1,
                 mps: MatrixProductState = None,
                 exact_solver_dim: int = 200):
        """
        The Density Matrix Renormalization Group (DMRG) algorithm of a finite size system.

        Args:
            mpo: Matrix product operator.
            bond_dim: Bond dimensions.
            block_size: Block size of each local update.
            mps: (Optional) Initial guess of matrix product state.
            exact_solver_dim: A switch point at which the exact eigensolver should be used
                below certain matrix size.

        Examples:
            To start the algorithm, one can call :func:`~FiniteDMRG.run`,

                fdmrg = FiniteDMRG(mpo, bond_dim=20)
                fdmrg.run(tol=1e-8)

            The optimized matrix product state can then be retrieved with :attr:`~FiniteDMRG.mps`.
        """
        self._n_sites = mpo.nsites
        self._bond_dim = bond_dim
        self._phys_dim = mpo.phys_dim(0)
        self._block_size = block_size
        self._exact_solver_dim = exact_solver_dim
        mps = MatrixProductState.random(
            n=self.n_sites, bond_dim=self.bond_dim, phys_dim=self.phys_dim
        ) if mps is None else mps
        self._env = Environment(mpo=mpo, mps=mps)
        self._energies = [np.nan]
        self._variances = [np.nan]

    @property
    def bond_dim(self) -> int:
        return self._bond_dim

    @property
    def mps(self) -> MatrixProductState:
        return self._env.mps

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def phys_dim(self) -> int:
        return self._phys_dim

    def variance(self) -> float:
        return self._env.variance()

    def one_site_solver(self, site: int, tol: float = 1e-8, **kwargs) -> Tuple[float, np.ndarray]:
        """

        Args:
            site:
            tol: Tolerance to the eigensolver.
            **kwargs: Keyword arguments to primme eigensolver.

        Returns:

        """
        v0 = self.mps[site].data.reshape(-1, 1)
        if v0.size < self._exact_solver_dim:
            return eigh(self._env.one_site_full_matrix(site))
        return eigshmv(
            self._env.one_site_matvec(site),
            v0=v0, tol=tol, **kwargs
        )

    def two_site_solver(self, site: int, tol: float = 1e-8, **kwargs):
        return NotImplemented

    def modified_density_matrix(self, site: int, alpha: float = 0):
        """
        For the single-site DMRG :func:`~FiniteDMRG.one_site_solver`,
        the modified density matrix can be used against trapping in local minimum.

        Args:
            site:
            alpha: The small parameter.

        Returns:
            modified_dm:

        References:
            `S. R. White,
            Density matrix renormalization group algorithms with a single center site,
            Phys. Rev. B 72, 180403 (2005).
            <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.72.180403>`
        """
        return NotImplemented

    def sweep(self, direction: Direction = Direction.rightward, tol: float = 1e-8, **kwargs) -> float:
        """
        Perform a single sweep on the given ``direction``.

        Args:
            direction: The left or right direction on which the sweep to perform.
            tol: Tolerance to the eigensolver.
            **kwargs: Keyword arguments to primme eigensolver.

        Returns:
            energy: Variationally optimized energy after this sweep.
        """
        iter_sites = range(self.n_sites - 1) if direction == Direction.rightward \
            else range(self.n_sites - 1, 0, -1)
        energy = None
        for site in iter_sites:
            energy, psi = self.one_site_solver(site, tol, **kwargs)
            logger.info(f"Sweeping to site [{site + 1}/{self.n_sites}], E0 = {energy}")
            self._env.update_mps(site, data=psi.reshape(self.mps[site].shape))
            self._env.split_tensor(site, direction=direction)
            self._env.update(site, direction=direction)
        return energy

    def _converged(self, stopping_criterion: StoppingCriterion,
                   tol: float, n_sweep: int, max_sweep: int) -> bool:
        """
        Helper function for checking the convergence.

        Args:
            stopping_criterion:
            tol:
            n_sweep:
            max_sweep:

        Returns:

        """
        def rule(criteria: float, label: str) -> bool:
            logger.info(f"and lowers {label} by {criteria:e}.")
            if abs(criteria) < tol:
                logger.info(f"Reaching set tolerance {tol}, stop sweeping.")
                return True
            elif n_sweep == max_sweep:
                # @TODO: dump mps to file and raise error
                logger.warning(f"Maximum number of sweeps {max_sweep} is reached, "
                               f"yet {label} = {criteria:e} is still greater than tol = {tol}.")
            elif abs(criteria) > tol and criteria < 0:
                logger.warning(f"Might be trapped in local minimum in this sweep, "
                               f"got {label} = {criteria:e}, skip and proceed.")
            return False

        if stopping_criterion == StoppingCriterion.by_energy:
            energy_gradient = self._energies[-2] - self._energies[-1]
            return rule(energy_gradient, label='dE')
        elif stopping_criterion == StoppingCriterion.by_variance:
            self._variances.append(self._env.variance())
            variance_gradient = self._variances[-2] - self._variances[-1]
            return rule(variance_gradient, label='dVar')
        return False

    def run(self, tol: float = 1e-7, max_sweep: int = 100,
            stopping_criterion: StoppingCriterion = StoppingCriterion.by_energy, **kwargs) -> List[float]:
        """
        By calling this method, :class:`~FiniteDMRG` will start the sweeping procedure
        until the given tolerance is reached or touching the maximally allowed number of sweeps.

        Args:
            tol: Required precision to the variationally optimized energy.
            max_sweep: Maximum number of sweeps.
            stopping_criterion: By which value to examine the convergence.
            **kwargs: Keyword arguments to primme eigensolver.

        Returns:
            energies: A record of the energies computed on each sweep.
        """
        clock = [time.process_time()]
        logger.info(f"Set tolerance = {tol}, up to maximally {max_sweep} sweeps.")
        for n_sweep, direction in zip(range(1, max_sweep + 1), cycle([Direction.rightward, Direction.leftward])):
            logger.info(f"<==== In sweep epoch [{n_sweep}/{max_sweep}] ====>")
            energy = self.sweep(direction, tol=tol, **kwargs)
            clock.append(time.process_time())
            self._energies.append(energy)
            logger.info(f"This sweep takes {timedelta(seconds=np.diff(clock[-2:])[0])},")
            if self._converged(stopping_criterion, tol, n_sweep, max_sweep):
                break
        elapsed_time = np.mean(np.sort(np.diff(clock))[:3])
        logger.info(f"Summary - {n_sweep} sweeps, "
                    f"best of {min(3, n_sweep)} - {timedelta(seconds=elapsed_time)} per sweep.")
        return self._energies[1:]

    @property
    def measurements(self) -> MatrixProductStateMeasurements:
        if len(self._energies) == 1:
            raise RuntimeError("FiniteDMRG is probably not executed yet.")
        return MatrixProductStateMeasurements(self.mps)


class ShiftInvertDMRG(FiniteDMRG):

    def __init__(self, mpo: MatrixProductOperator,
                 bond_dim: int,
                 offset: float = 0,
                 block_size: int = 1,
                 mps: MatrixProductState = None,
                 exact_solver_dim: int = 200):
        """
        The DMRG algorithm for optimizing the shift-invert spectrum of a finite size system.

        Args:
            mpo:
            bond_dim:
            offset:
            block_size:
            mps:
            exact_solver_dim:
        """
        super(ShiftInvertDMRG, self).__init__(
            mpo, bond_dim=bond_dim, block_size=block_size,
            mps=mps, exact_solver_dim=exact_solver_dim
        )
        self._env2 = Environment(mpo=mpo.square(), mps=self.mps)
        self._offset = offset
        self._restored_mps = None

    def restore_energy(self, energy: float) -> float:
        """

        Args:
            energy:

        Returns:

        """
        return 1 / energy + self._offset

    @property
    def restored_mps(self) -> MatrixProductState:
        return self._restored_mps

    def _restore_mps(self):
        def gen_array(site: int) -> np.ndarray:
            net = self._env.mpo[site] & conj_mps[site]
            virtual_inds = self._env.mpo[site].inds[:-2]
            phys_ind = self._env.mpo[site].inds[-2]
            if site == 0:
                fuse_map = {
                    rand_uuid(): [conj_mps[site].inds[1], virtual_inds[0]],
                    phys_ind: [phys_ind]
                }
            elif site == self.n_sites - 1:
                fuse_map = {
                    phys_ind: [phys_ind],
                    rand_uuid(): [conj_mps[site].inds[0], virtual_inds[0]]
                }
            else:
                fuse_map = {
                    rand_uuid(): [conj_mps[site].inds[0], virtual_inds[0]],
                    phys_ind: [phys_ind],
                    rand_uuid(): [conj_mps[site].inds[2], virtual_inds[1]],
                }
            return net.contract().fuse(fuse_map).data
        conj_mps = self.mps.conj(mangle_outer=True)
        self._restored_mps = MatrixProductState(
            [gen_array(site) for site in range(self.n_sites)], shape='lpr'
        )

    def one_site_solver(self, site: int, tol: float = 1e-8, **kwargs) -> Tuple[float, np.ndarray]:
        v0 = self.mps[site].data.reshape(-1, 1)
        if v0.size < self._exact_solver_dim:
            return eigh(
                self._env.one_site_full_matrix(site),
                b=self._env2.one_site_full_matrix(site),
                backend='scipy'
            )
        return eigshmv(
            self._env.one_site_matvec(site),
            M=self._env2.one_site_matvec(site),
            v0=v0, tol=tol, **kwargs
        )

    def sweep(self, direction: Direction = Direction.rightward, tol: float = 1e-8, **kwargs) -> float:
        iter_sites = range(self.n_sites - 1) if direction == Direction.rightward \
            else range(self.n_sites - 1, 0, -1)
        energy = None
        for site in iter_sites:
            energy, psi = self.one_site_solver(site, tol, **kwargs)
            logger.info(f"Sweeping to site [{site + 1}/{self.n_sites}], "
                        f"E0 = {self.restore_energy(energy)}")
            self._env.update_mps(site, data=psi.reshape(self.mps[site].shape))
            self._env.split_tensor(site, direction=direction)
            self._env.update(site, direction=direction)
            self._env2.update_mps(site, data=self.mps[site].data)
            if direction == Direction.leftward:
                self._env2.update_mps(site - 1, data=self.mps[site - 1].data)
            elif direction == Direction.rightward:
                self._env2.update_mps(site + 1, data=self.mps[site + 1].data)
            self._env2.update(site, direction=direction)
        return energy

    def run(self, tol: float = 1e-7, max_sweep: int = 100,
            stopping_criterion: StoppingCriterion = StoppingCriterion.by_energy, **kwargs) -> List[float]:
        energies = super(ShiftInvertDMRG, self).run(tol, max_sweep, stopping_criterion, **kwargs)
        self._restore_mps()
        return [self.restore_energy(energy) for energy in energies]

    @property
    def measurements(self) -> MatrixProductStateMeasurements:
        if len(self._energies) == 1:
            raise RuntimeError("FiniteDMRG is probably not executed yet.")
        return MatrixProductStateMeasurements(self.restored_mps)

    @property
    def variance(self) -> float:
        net = self.restored_mps & \
              self._env.mpo.square() & \
              self.restored_mps.conj(mangle_inner=True, mangle_outer=True)
        return net.contract() - self._energies[-1] ** 2 - self._offset ** 2
