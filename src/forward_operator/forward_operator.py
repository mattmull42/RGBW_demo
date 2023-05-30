from .operators.abstract_operator import abstract_operator


class forward_operator(abstract_operator):
    def __init__(self, operator_list, name=None):
        self.operator_list = operator_list
        self.name = 'forward' if name is None else name

        super().__init__(operator_list[0].input_shape, operator_list[-1].output_shape, self.name)


    def direct(self, x):
        res = x

        for operator in self.operator_list:
            res = operator.direct(res)

        return res


    def adjoint(self, y):
        res = y

        for operator in reversed(self.operator_list):
            res = operator.adjoint(res)

        return res


    @property
    def matrix(self):
        mat = self.operator_list[0].matrix

        for operator in self.operator_list[1:]:
            mat = operator.matrix @ mat

        return mat


    def __str__(self):
        res = f'{self.name} operator of {type(self)} from {self.input_shape} to {self.output_shape} with the operators:'

        for operator in self.operator_list:
            res += '\n   ' + operator.__str__()

        return res