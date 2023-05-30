import pylops as pyl
import pyproximal as pyp
from scipy.sparse.linalg import norm

from .forward_operator_pylops import *
from src.forward_operator.forward_operator import forward_operator


class Inverse_problem_ADMM:
    def __init__(self, cfa, binning, noise_level, output_size, spectral_stencil, max_iter, eps):
        self.output_size = output_size
        self.max_iter = max_iter
        self.eps = eps

        self.forward_op = forward_operator(cfa, binning, noise_level, output_size, spectral_stencil)


    def __call__(self, y):
        data_fidelity_term = pyp.L2(Op=Forward_operator_pylops(self.forward_op), b=y.reshape(-1, order='F'))
        g = pyp.L21(np.prod(self.forward_op.output_size[:-1]), sigma=self.eps)
        grad = pyl.Gradient(dims=self.output_size, edge=True)

        x0 = self.forward_op.adjoint(y).reshape(-1, order='F')

        # tau = 1e-1
        # mu = tau / norm(self.forward_op.get_matrix())**2
        mu = 1e-1
        tau = mu / norm(self.forward_op.get_matrix())**2

        self.output = pyp.optimization.primal.LinearizedADMM(data_fidelity_term, g, grad, x0=np.zeros_like(x0), tau=tau, mu=mu, niter=self.max_iter, show=True)
        self.output = self.output[0].reshape(self.output_size, order='F')

        np.clip(self.output, 0, 1, self.output)

        return self.output


class Inverse_problem_ADMM_l2:
    def __init__(self, cfa, binning, noise_level, output_size, spectral_stencil, max_iter, eps):
        self.output_size = output_size
        self.max_iter = max_iter
        self.eps = eps

        self.forward_op = forward_operator(cfa, binning, noise_level, output_size, spectral_stencil)


    def __call__(self, y):
        op = Forward_operator_pylops(self.forward_op)
        g = pyp.L21(np.prod(self.forward_op.output_size[:-1]), sigma=self.eps)
        grad = pyl.Gradient(dims=self.output_size, edge=True)

        x0 = self.forward_op.adjoint(y).reshape(-1, order='F')

        tau = 1 / norm(self.forward_op.get_matrix())**2

        self.output = pyp.optimization.primal.ADMML2(g, op, y.reshape(-1, order='F'), grad, x0=x0, tau=tau, niter=self.max_iter, show=True)
        self.output = self.output.reshape(self.output_size, order='F')

        np.clip(self.output, 0, 1, self.output)

        return self.output