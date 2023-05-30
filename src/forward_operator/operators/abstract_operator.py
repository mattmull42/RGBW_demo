from abc import ABC, abstractmethod
import scipy.sparse.linalg as sp_lin


class abstract_operator(ABC):
    def __init__(self, input_shape, output_shape, name):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.name = name


    @abstractmethod
    def direct(self, x):
        pass


    @abstractmethod
    def adjoint(self, y):
        pass


    @property
    def matrix(self):
        pass


    @property
    def norm(self):
        return sp_lin.norm(self.matrix, 2)


    def __str__(self):
        return f'{self.name} operator of {type(self)} from {self.input_shape} to {self.output_shape}'