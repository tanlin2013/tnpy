import time
from enum import Enum
from datetime import timedelta
from itertools import cycle
from functools import partial
from typing import Tuple, List

import numpy as np
from quimb.tensor import rand_uuid

from tnpy import logger
from tnpy.linalg import eigh, eigshmv
from tnpy.operators import MatrixProductOperator
from tnpy.matrix_product_state import (
    MatrixProductState,
    Environment,
    Direction,
    MatrixProductStateMeasurements
)


class Metric(Enum):
    """
    Enumeration for which value the stopping criterion will examine in
    sweeping procedure.
    """
    ENERGY = 1
    VARIANCE = 2


class FiniteDMRG:

    def __init__(self, mpo: MatrixProductOperator,
                 bond_dim: int,
                 block_size: int = 1,
                 mps: MatrixProductState = None,
                 exact_solver_dim: int = 200):
        """
        The Density Matrix Renormalization Group (DMRG) algorithm of a
        finite size system.

        Args:
            mpo: Matrix product operator.
            bond_dim: Bond dimensions.
            block_size: Block size of each local update.
            mps: (Optional) Initial guess of :class:`~MatrixProductState`.
            exact_solver_dim: A switch point at which the exact eigensolver
                should be used below certain matrix size.

        Examples:
            To start the algorithm, one can call :func:`~FiniteDMRG.run`,

                fdmrg = FiniteDMRG(mpo, bond_dim=20)
                fdmrg.run(tol=1e-8)

            The optimized :class:`~MatrixProductState` can then be retrieved
            from :attr:`~FiniteDMRG.mps`.
        """
        self._n_sites = mpo.n_sites
        self._bond_dim = bond_dim
        self._phys_dim = mpo.phys_dim
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

    def one_site_solver(self, site: int, tol: float = 1e-8,
                        **kwargs) -> Tuple[float, np.ndarray]:
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

    def perturb_wave_function(self, site: int, alpha: float = 1e-5):
        """
        Perturb the local tensor of wave function by an amount of ``alpha``.
        For the single-site DMRG :func:`~FiniteDMRG.one_site_solver`,
        the modified density matrix can be used against trapping in local
        minimums. This is an in-place operation.

        Args:
            site:
            alpha: The small parameter for perturbation.

        References:
            `S. R. White,
            Density matrix renormalization group algorithms with a single
            center site, Phys. Rev. B 72, 180403 (2005).
            <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.72.180403>`_
            `U. Schollwoeck,
            The density-matrix renormalization group in the age of matrix
            product states, arXiv:1008.3477 (2011).
            <https://arxiv.org/abs/1008.3477>`_
        """
        # TODO: dynamically adjust alpha
        psi = self.mps[site].data.flatten()
        psi += alpha * self._env.one_site_matvec(site).matvec(psi)
        self._env.update_mps(site, data=psi.reshape(self.mps[site].shape))

    def sweep(self, direction: Direction = Direction.RIGHTWARD,
              tol: float = 1e-8, **kwargs) -> float:
        """
        Perform a single sweep on the given ``direction``.

        Args:
            direction: The left or right direction on which the sweep to
                perform.
            tol: Tolerance to the eigensolver.
            **kwargs: Keyword arguments to primme eigensolver.

        Returns:
            energy: Variationally optimized energy after this sweep.
        """
        iter_sites = range(self.n_sites - 1) \
            if direction == Direction.RIGHTWARD \
            else range(self.n_sites - 1, 0, -1)
        energy = None
        for site in iter_sites:
            energy, psi = self.one_site_solver(site, tol, **kwargs)
            logger.info(
                f"Sweeping to site [{site + 1}/{self.n_sites}], E0 = {energy}"
            )
            self._env.update_mps(site, data=psi.reshape(self.mps[site].shape))
            self.perturb_wave_function(site)
            self._env.split_tensor(site, direction=direction)
            self._env.update(site, direction=direction)
        return energy

    def _converged(self, n_sweep: int, tol: float, max_sweep: int,
                   metric: Metric) -> bool:
        """
        Helper function for checking the convergence.

        Args:
            n_sweep:
            tol:
            max_sweep:
            metric:

        Returns:

        """
        def stopping_criterion(gradient: float) -> bool:
            logger.info(
                f"Metric {metric.name} is lowered by {gradient:e} "
                f"in this sweep."
            )
            if abs(gradient) < tol:
                logger.info(f"Reaching set tolerance {tol}, stop sweeping.")
                return True
            elif n_sweep == max_sweep:
                # @TODO: dump mps to file and raise error
                logger.warning(
                    f"Maximum number of sweeps {max_sweep} is reached, "
                    f"yet {metric.name} gradient = {gradient:e} is still "
                    f"greater than tol = {tol}."
                )
            elif abs(gradient) > tol and gradient < 0:
                # TODO: this is not the case for shift-invert method
                logger.warning(
                    f"Might be trapped in local minimum in this sweep, "
                    f"got {metric.name} gradient = {gradient:e}, "
                    f"skip and proceed."
                )
            return False

        return stopping_criterion(np.diff(self._variances[-2:])[0]) \
            if metric == Metric.VARIANCE \
            else stopping_criterion(np.diff(self._energies[-2:])[0])

    def run(self, tol: float = 1e-8, max_sweep: int = 100,
            metric: Metric = Metric.ENERGY, **kwargs) -> List[float]:
        """
        By calling this method, :class:`~FiniteDMRG` will start the sweeping
        procedure until the given tolerance is reached or touching the
        maximally allowed number of sweeps.

        Args:
            tol: Required precision to the variationally optimized energy.
            max_sweep: Maximum number of sweeps.
            metric: By which value to examine the convergence.
            **kwargs: Keyword arguments to primme eigensolver.

        Returns:
            energies: A record of energies computed on each sweep.
        """
        clock = [time.process_time()]
        converged = partial(self._converged, tol=tol,
                            max_sweep=max_sweep, metric=metric)
        logger.info(f"Set tolerance = {tol} to metric {metric.name},"
                    f" up to maximally {max_sweep} sweeps.")
        for n_sweep, direction in zip(
                range(1, max_sweep + 1),
                cycle([Direction.RIGHTWARD, Direction.LEFTWARD])
        ):
            logger.info(f"<==== In sweep epoch [{n_sweep}/{max_sweep}] ====>")
            energy = self.sweep(direction, tol=tol, **kwargs)
            clock.append(time.process_time())
            self._energies.append(energy)
            self._variances.append(self._env.variance())
            logger.info(
                f"Last sweep took {timedelta(seconds=np.diff(clock[-2:])[0])}."
            )
            if converged(n_sweep):
                break
        elapsed_time = np.mean(np.sort(np.diff(clock))[:3])
        logger.info(
            f"Summary - {n_sweep} sweeps, best of {min(3, n_sweep)} - "
            f"{timedelta(seconds=elapsed_time)} per sweep."
        )
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
        The DMRG algorithm for optimizing the shift-invert spectrum of
        a finite size system.

        Args:
            mpo:
            bond_dim:
            offset:
            block_size:
            mps:
            exact_solver_dim:
        """
        super().__init__(
            mpo, bond_dim=bond_dim, block_size=block_size,
            mps=mps, exact_solver_dim=exact_solver_dim
        )
        self._env2 = Environment(mpo=mpo.square(), mps=self.mps)
        self._offset = offset
        self._restored_mps = None

    @property
    def restored_mps(self) -> MatrixProductState:
        return self._restored_mps

    def _restore_mps(self):
        def gen_array(site: int) -> np.ndarray:
            tn = self._env.mpo[site] & conj_mps[site]
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
            return tn.contract().fuse(fuse_map).data
        conj_mps = self.mps.conj(mangle_outer=True)
        self._restored_mps = MatrixProductState(
            [gen_array(site) for site in range(self.n_sites)], shape='lpr'
        )

    def one_site_solver(self, site: int, tol: float = 1e-8,
                        **kwargs) -> Tuple[float, np.ndarray]:
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

    def sweep(self, direction: Direction = Direction.RIGHTWARD,
              tol: float = 1e-8, **kwargs) -> float:
        iter_sites = range(self.n_sites - 1) \
            if direction == Direction.RIGHTWARD \
            else range(self.n_sites - 1, 0, -1)
        energy = None
        for site in iter_sites:
            energy, psi = self.one_site_solver(site, tol, **kwargs)
            logger.info(f"Sweeping to site [{site + 1}/{self.n_sites}], "
                        f"E0 = {1 / energy + self._offset}")
            self._env.update_mps(site, data=psi.reshape(self.mps[site].shape))
            self.perturb_wave_function(site)
            self._env.split_tensor(site, direction=direction)
            self._env.update(site, direction=direction)
            self._env2.update_mps(site, data=self.mps[site].data)
            if direction == Direction.LEFTWARD:
                self._env2.update_mps(site - 1, data=self.mps[site - 1].data)
            elif direction == Direction.RIGHTWARD:
                self._env2.update_mps(site + 1, data=self.mps[site + 1].data)
            self._env2.update(site, direction=direction)
        return energy

    def run(self, tol: float = 1e-7, max_sweep: int = 100,
            metric: Metric = Metric.ENERGY, **kwargs) -> List[float]:
        energies = super().run(
            tol, max_sweep, metric, **kwargs
        )
        self._restore_mps()
        return np.reciprocal(energies) + self._offset

    @property
    def measurements(self) -> MatrixProductStateMeasurements:
        if len(self._energies) == 1:
            raise RuntimeError("FiniteDMRG is probably not executed yet.")
        return MatrixProductStateMeasurements(self.restored_mps)

    @property
    def variance(self) -> float:
        tn = self.restored_mps & \
              self._env.mpo.square() & \
              self.restored_mps.conj(mangle_inner=True, mangle_outer=True)
        return tn.contract() - self._energies[-1] ** 2 - self._offset ** 2
