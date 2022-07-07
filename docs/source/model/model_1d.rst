Model 1D
==========
The base class for generating an 1-dimensional model
where its Hamiltonian can be expressed in
Matrix Product Operator (MPO).

.. autosummary::
    :toctree: _autosummary

    tnpy.model.Model1D

Customize the model
===================
Apart from the built-in models,
it's possible customize the model by implementing :function:`tnpy.model.Model1D._elem`,
which accepts `site` (int) as input, and is expected to return a :class:`numpy.ndarray`.

.. code-block::

    import numpy as np
    from tnpy.operators import SpinOperators
    from tnpy.model import Model1D

    class XXZ(Model1D):
        def __init__(self, n: int, delta: float):
            self.delta = delta
            super().__init__(n)

        def _elem(self, site: int) -> np.ndarray:
            Sp, Sm, Sz, I2, O2 = SpinOperators()
            return np.array(
                [[I2, -0.5 * Sp, -0.5 * Sm, -self.delta * Sz, O2],
                [O2, O2, O2, O2, Sm],
                [O2, O2, O2, O2, Sp],
                [O2, O2, O2, O2, Sz],
                [O2, O2, O2, O2, I2]]
            )

One can then access the MPO operator by calling :attr:`tnpy.model.Model1D.mpo`,
which returns a :class:`tnpy.operators.MatrixProductOperator` object.

.. code-block::

    model = XXZ(n=10, delta=0.5)
    mpo = model.mpo
