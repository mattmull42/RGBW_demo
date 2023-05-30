import numpy as np
import scipy.sparse as sp


def precompute_input(x, A=None, b=None):
    if A is not None:
        res = A(x)

    else:
        res = np.copy(x)

    if b is not None:
        res -= b

    return res


class L2:
    def __init__(self, A=None, b=None, sigma=1):
        self.A = A
        self.b = b
        self.sigma = sigma


    def __call__(self, x):
        tmp = precompute_input(x, self.A, self.b).reshape(-1)

        return self.sigma * np.dot(tmp, tmp) / 2


    def grad(self, x):
        if self.A is not None:
            if self.b is not None:
                return self.sigma * self.A.adjoint(self.A(x) - self.b)

            else:
                return self.sigma * self.A.adjoint(self.A(x))

        else:
            if self.b is not None:
                return self.sigma * (x - self.b)

            else:
                return self.sigma * x


    def prox(self, x, tau):
        if not hasattr(self, 'tau') or self.tau != tau:
            self.tau = tau
            self.tau_sigma = tau * self.sigma

            if self.A is not None:
                self.matrix_op = self.A.get_matrix_operator()
                self.solver = sp.linalg.factorized(sp.eye(self.matrix_op.shape[1], format='csc') + self.tau_sigma * self.matrix_op.T @ self.matrix_op)

        if self.A is not None:
            y = x.reshape(-1, order='F')

            if self.b is not None:
                y += tau * self.matrix_op.T @ self.b.reshape(-1, order='F')
            return self.solver(y).reshape(self.A.input_size, order='F')

        if self.b is not None:
            return (x + self.tau_sigma * self.b) / (1 + self.tau_sigma)

        return x / (1 + self.tau_sigma)


class L1:
    def __init__(self, b=None, sigma=1):
        self.b = b
        self.sigma = sigma


    def __call__(self, x):
        tmp = precompute_input(x, None, self.b).reshape(-1)

        return self.sigma * np.linalg.norm(tmp, ord=1)


    def prox(self, x, tau):
        if self.b is None:
            return np.maximum(np.abs(x) - tau * self.sigma, 0.) * np.sign(x)

        else:
            return np.maximum(np.abs(x - self.b) - tau * self.sigma, 0.) * np.sign(x) + self.b