import pylops as pyl
import numpy as np


class Forward_operator_pylops(pyl.LinearOperator):
    def __init__(self, forward_operator):
        self.forward_operator = forward_operator
        self.shape = (np.prod(forward_operator.output_size), np.prod(forward_operator.input_size))
        self.dtype = np.dtype(None)
        self.explicit = False


    def _matvec(self, x):
        return self.forward_operator(x.reshape(self.forward_operator.input_size, order='F')).reshape(-1, order='F')


    def _rmatvec(self, x):
        return self.forward_operator.adjoint(x.reshape(self.forward_operator.output_size, order='F')).reshape(-1, order='F')