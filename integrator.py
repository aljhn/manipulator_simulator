import abc
import numpy as np


class AbstractIntegrator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, n):
        self.n = n

    @abc.abstractmethod
    def step(self, manipulator, q, q_dot, u, h):
        pass


class ExplicitEuler(AbstractIntegrator):

    def __init__(self, n):
        self.n = n

    def step(self, manipulator, q, q_dot, u, h):
        qq = manipulator.dynamics(q, q_dot, u)
        q += qq[:self.n] * h
        q_dot += qq[self.n:] * h


class ExplicitRK4(AbstractIntegrator):

    def __init__(self, n):
        self.n = n

        self.butcher_A = np.array([
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0],
            [0, 0, 1, 0]
        ])
        self.butcher_b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        self.butcher_c = np.array([0, 1 / 2, 1 / 2, 1])
        self.k = np.zeros((2 * n, len(self.butcher_b)))

    def step(self, manipulator, q, q_dot, u, h):
        for i in range(len(self.butcher_b)):
            qq = np.dot(self.k, self.butcher_A[i, :]) * h
            self.k[:, i] = manipulator.dynamics(q + qq[:self.n], q_dot + qq[self.n:], u)
        
        qq = np.dot(self.k, self.butcher_b) * h
        q += qq[:self.n]
        q_dot += qq[self.n:]
