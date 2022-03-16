import time
import numpy as np
from datetime import timedelta
from itertools import cycle
from tnpy import logger
from tnpy.linalg import eigh, eigshmv
from tnpy.matrix_product_state import MatrixProductState, Environment, Direction
from tnpy.operators import MatrixProductOperator
from typing import Tuple, List


class FiniteDMRG:

    def __init__(self, mpo: MatrixProductOperator,
                 bond_dim: int,
                 block_size: int = 1,
                 mps: MatrixProductState = None):
        """

        Args:
            mpo:
            bond_dim:
            block_size: Block size of each local update.
            mps:

        Examples:

        """
        self._n_sites = mpo.nsites
        self._bond_dim = bond_dim
        self._phys_dim = mpo.phys_dim(0)
        self._block_size = block_size
        mps = MatrixProductState.random(
            n=self.n_sites, bond_dim=self.bond_dim, phys_dim=self.phys_dim
        ) if mps is None else mps
        self._env = Environment(mpo=mpo, mps=mps)

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

    def one_site_solver(self, site: int, tol: float = 1e-8, **kwargs) -> Tuple[float, np.ndarray]:
        v0 = self.mps[site].data.reshape(-1, 1)
        if v0.size < 200:
            return eigh(self._env.one_site_full_matrix(site))
        return eigshmv(
            self._env.one_site_matvec(site),
            v0=v0, tol=tol, **kwargs
        )

    def two_site_solver(self, site: int, tol: float = 1e-8, **kwargs):
        return NotImplemented

    def modified_density_matrix(self, site: int, alpha: float = 0):
        return NotImplemented

    def sweep(self, direction: Direction = Direction.rightward, tol: float = 1e-8, **kwargs) -> float:
        """
        Perform a single sweep on the given direction.

        Args:
            direction:
            tol:
            **kwargs:

        Returns:

        """
        iter_sites = range(self.n_sites - 1) if direction == Direction.rightward \
            else range(self.n_sites - 1, 0, -1)
        energy = None
        for site in iter_sites:
            energy, psi = self.one_site_solver(site, tol, **kwargs)
            logger.info(f"Sweeping to site [{site + 1}/{self.n_sites}], E0/N = {energy / self.n_sites}")
            self._env.update_mps(site, data=psi.reshape(self.mps[site].shape))
            self._env.split_tensor(site, direction=direction)
            self._env.update(site, direction=direction)
        return energy

    def _converged(self, energy_gradient: float, tol: float, n_sweep: int, max_sweep: int) -> bool:
        if abs(energy_gradient) < tol:
            logger.info(f"Reaching set tolerance {tol}, stop sweeping.")
            return True
        elif n_sweep == max_sweep:
            # @TODO: dump mps to file and raise error
            logger.warning(f"Maximum number of sweeps {max_sweep} is reached, "
                           f"yet dE/N = {energy_gradient:e} is greater than tol = {tol}.")
        elif abs(energy_gradient) > tol and energy_gradient < 0:
            logger.warning(f"Fail on lowering energy in this sweep, "
                           f"got dE/N = {energy_gradient:e}, skip and proceed.")
        return False

    def run(self, tol: float = 1e-7, max_sweep: int = 100, **kwargs) -> List[float]:
        """

        Args:
            tol:
            max_sweep:
            **kwargs:

        Returns:

        """
        clock, energies = [time.process_time()], [np.nan]
        logger.info(f"Set tolerance = {tol}, up to maximally {max_sweep} sweeps.")
        for n_sweep, direction in zip(range(1, max_sweep + 1), cycle([Direction.rightward, Direction.leftward])):
            logger.info(f"<<==== In sweep epoch [{n_sweep}/{max_sweep}] ====>>")
            energy = self.sweep(direction, tol=tol, **kwargs)
            energy_gradient = (energies[-1] - energy) / self.n_sites
            clock.append(time.process_time())
            energies.append(energy)
            logger.info(f"This sweep takes {timedelta(seconds=np.diff(clock[-2:])[0])} "
                        f"and lowers dE/N by {energy_gradient:e}.")
            if self._converged(energy_gradient, tol, n_sweep, max_sweep):
                break
        elapsed_time = np.mean(np.sort(np.diff(clock))[:3])
        logger.info(f"Summary - {n_sweep} sweeps, "
                    f"best of {min(3, n_sweep)} - {timedelta(seconds=elapsed_time)} per sweep.")
        return energies


class ShiftInvertDMRG(FiniteDMRG):

    def __init__(self, mpo: MatrixProductOperator,
                 bond_dim: int,
                 offset: float,
                 block_size: int = 1,
                 mps: MatrixProductState = None):
        """

        Args:
            mpo:
            bond_dim:
            offset:
            block_size:
            mps:
        """
        super(ShiftInvertDMRG, self).__init__(mpo, bond_dim=bond_dim, block_size=block_size, mps=mps)
        self._env2 = Environment(mpo=mpo.square(), mps=self.mps)
        self._offset = offset

    def one_site_solver(self, site: int, tol: float = 1e-8, **kwargs) -> Tuple[float, np.ndarray]:
        v0 = self.mps[site].data.reshape(-1, 1)
        if v0.size < 200:
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
                        f"E0/N = {(1 / energy + self._offset) / self.n_sites}")
            self._env.update_mps(site, data=psi.reshape(self.mps[site].shape))
            self._env.split_tensor(site, direction=direction)
            self._env.update(site, direction=direction)
            self._env2.update_mps(site, data=psi.reshape(self.mps[site].shape))
            self._env2.split_tensor(site, direction=direction)
            self._env2.update(site, direction=direction)
        return energy

    def run(self, tol: float = 1e-7, max_sweep: int = 100, **kwargs) -> List[float]:
        energies = super(ShiftInvertDMRG, self).run(tol, max_sweep, **kwargs)
        return [(1 / energy + self._offset) for energy in energies]
