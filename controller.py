import abc
import numpy as np


class AbstractController(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get(self, q, q_dot):
        pass


class ZeroController(AbstractController):

    def __init__(self):
        pass
    
    def get(self, q, q_dot):
        return np.zeros(q.shape[0])


class PDGravityController(AbstractController):

    def __init__(self):
        pass
    
    def get(self, q):
        pass


class FeedbackLinearizationController(AbstractController):

    def __init__(self):
        pass
    
    def get(self, q):
        pass
