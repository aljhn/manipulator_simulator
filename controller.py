import abc
import numpy as np


class AbstractController(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get(self, q, q_dot, q_r, q_r_dot, q_r_ddot):
        pass


class ZeroController(AbstractController):

    def __init__(self):
        pass
    
    def get(self, q, q_dot, q_r, q_r_dot, q_r_ddot):
        return np.zeros(q.shape[0])


class PDController(AbstractController):

    def __init__(self, k_p, k_d):
        self.k_p = k_p
        self.k_d = k_d
    
    def get(self, q, q_dot, q_r, q_r_dot, q_r_ddot):
        return - self.k_p * (q - q_r) - self.k_d * (q_dot - q_r_dot)


class PDGravityController(AbstractController):

    def __init__(self, k_p, k_d, manipulator):
        self.k_p = k_p
        self.k_d = k_d
        self.manipulator = manipulator
    
    def get(self, q, q_dot, q_r, q_r_dot, q_r_ddot):
        u_ff = -self.manipulator.f(*q, *q_dot)[:, 0] + np.dot(self.manipulator.M(*q, *q_dot), q_r_ddot) - 0.5 * np.dot(self.manipulator.M_dot(*q, *q_dot), q_dot - q_r_dot)

        u_fb = - self.k_p * (q - q_r) - self.k_d * (q_dot - q_r_dot)

        return u_ff + u_fb


class FeedbackLinearizationController(AbstractController):

    def __init__(self, k_p, k_d, manipulator):
        self.k_p = k_p
        self.k_d = k_d
        self.manipulator = manipulator
    
    def get(self, q, q_dot, q_r, q_r_dot, q_r_ddot):
        v = q_r_ddot - self.k_p * (q - q_r) - self.k_d * (q_dot - q_r_dot)

        u = np.dot(self.manipulator.M(*q, *q_dot), v) - self.manipulator.f(*q, *q_dot)[:, 0]
        
        return u
