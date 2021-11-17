import logging
from tensornetwork import Node
from scipy.integrate import solve_ivp
from tnpy.finite_algorithm_base import FiniteAlgorithmBase
from tnpy.linalg import qr
from enum import Enum


class Evolve(Enum):
    FORWARD = 1
    BACKWARD = -1


class FiniteTDVP(FiniteAlgorithmBase):

    def __init__(self, mpo, chi, init_method):
        logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.root.setLevel(level=logging.INFO)
        super(FiniteTDVP, self).__init__(mpo, chi, init_method)
        self.center_matrices = {}

    def _unit_solver(self, proceed, t_span, site):

        def forward(t, y):
            M = Node(y.reshape(self.mps_shape(site)))
            W = self.mpo[site]
            if site == 0:
                Renv = self.right_envs[site]
                Rnorm = self.right_norms[site]
                Renv[0] ^ M[2]
                Renv[1] ^ W[0]
                M[1] ^ W[1]
                Renv[2] ^ Rnorm[0]
                result = M @ W @ Renv @ Rnorm
            elif site == self.N-1:
                Lenv = self.left_envs[site]
                Lnorm = self.left_norms[site]
                Lenv[0] ^ M[0]
                Lenv[1] ^ W[0]
                M[1] ^ W[1]
                Lenv[2] ^ Lnorm[0]
                result = Lenv @ Lnorm @ M @ W
            else:
                Lenv = self.left_envs[site]
                Lnorm = self.left_norms[site]
                Renv = self.right_envs[site]
                Rnorm = self.right_norms[site]
                Lenv[0] ^ M[0]
                Lenv[1] ^ W[0]
                M[1] ^ W[2]
                Renv[0] ^ M[2]
                Renv[1] ^ W[1]
                Lenv[2] ^ Lnorm[0]
                Renv[2] ^ Rnorm[0]
                result = Lenv @ Lnorm @ M @ W @ Renv @ Rnorm
            return -1j * result.tensor.reshape(y.shape)

        def backward(t, y):
            C = Node(y.reshape(self.center_matrices[site].tensor.shape))
            Lenv = self.left_envs[site+1]
            Lnorm = self.left_norms[site+1]
            Renv = self.right_envs[site]
            Rnorm = self.right_norms[site]
            Lenv[0] ^ C[0]
            Renv[0] ^ C[1]
            Lenv[1] ^ Renv[1]
            Lenv[2] ^ Lnorm[0]
            Renv[2] ^ Rnorm[0]
            result = Lenv @ Lnorm @ C @ Renv @ Rnorm
            return 1j * result.tensor.reshape(y.shape)

        if proceed == Evolve.FORWARD:
            y0 = self._mps.nodes[site].tensor.reshape(-1).astype(complex)
            result = solve_ivp(forward, t_span, y0)
        elif proceed == Evolve.BACKWARD:
            y0 = self.center_matrices[site].tensor.reshape(-1).astype(complex)
            result = solve_ivp(backward, t_span, y0)
        return result.y[:, -1]

    def sweep(self, iterator, t_span):
        direction = 1 if iterator[0] < iterator[-1] else -1
        for site in iterator:
            theta = self._unit_solver(Evolve.FORWARD, t_span, site)
            if direction == 1:
                theta = theta.reshape(self.d * self.mps_shape(site)[0], -1)
            elif direction == -1:
                theta = theta.reshape(-1, self.d * self.mps_shape(site)[2])
            q, r = qr(theta, chi=self.mps_shape(site)[1+direction])
            if direction == 1:
                self._mps.nodes[site] = Node(q.reshape(self.mps_shape(site)))
                self.center_matrices[site] = Node(r)
                self._update_left_env(site+1)
                self._update_left_norm(site+1)
                if site < self.N-1:
                    C = Node(self._unit_solver(Evolve.BACKWARD, t_span, site).reshape(r.shape))
                    Mp = self._mps.nodes[site+1]
                    C[1] ^ Mp[0]
                    self._mps.nodes[site+1] = C @ Mp
            elif direction == -1:
                self._mps.nodes[site] = Node(r.reshape(self.mps_shape(site)))
                self.center_matrices[site-1] = Node(q)
                self._update_right_env(site-1)
                self._update_right_norm(site-1)
                if site > 0:
                    C = Node(self._unit_solver(Evolve.BACKWARD, t_span, site-1).reshape(q.shape))
                    Mp = self._mps.nodes[site-1]
                    Mp[2] ^ C[0]
            # @TODO: measure something here to check the status of mps
            print(site)
            print(self._mps.check_orthonormality('l', self.N-1))
            print(self._mps.check_orthonormality('r', 0))
            print(self._mps.check_canonical())
            # logging.info("Sweeping to site [{}/{}], norm = ".format(site+1, self.N))
        return

    def evolve(self, t_span):
        pass
        #
