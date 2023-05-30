import pyproximal as pyp

from .forward_operator_pylops import *
from src.forward_operator.forward_operator import forward_operator


class Inverse_problem_FISTA:
    def __init__(self, cfa, binning, noise_level, output_size, spectral_stencil, max_iter, eps):
        self.output_size = output_size
        self.max_iter = max_iter
        self.eps = eps

        self.forward_op = forward_operator(cfa, binning, noise_level, output_size, spectral_stencil)


    def __call__(self, y):
        data_fidelity_term = pyp.L2(Op=Forward_operator_pylops(self.forward_op), b=y.reshape(-1, order='F'))
        regularizer = pyp.TV(self.output_size)

        x0 = self.forward_op.adjoint(y).reshape(-1, order='F')

        self.output = pyp.optimization.primal.ProximalGradient(data_fidelity_term, regularizer, x0=x0, epsg=self.eps, niter=self.max_iter, acceleration='fista', show=True)
        self.output = self.output.reshape(self.output_size, order='F')

        np.clip(self.output, 0, 1, self.output)

        return self.output