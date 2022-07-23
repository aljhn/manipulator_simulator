import abc
import numpy as np


class AbstractTrajectory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, cycle_time):
        self.cycle_time = cycle_time

    @abc.abstractmethod
    def get_end_position(self, t):
        pass

    @abc.abstractmethod
    def get_end_orientation(self, t):
        pass
    
    @abc.abstractmethod
    def get_end_transformation(self, t):
        pass
    
    @abc.abstractmethod
    def joint_angles(self, manipulator, t):
        pass


class EllipseTrajectory2D(AbstractTrajectory):

    def __init__(self, cycle_time, radius_x, radius_y):
        self.cycle_time = cycle_time
        self.radius_x = radius_x
        self.radius_y = radius_y
    
    def get_end_position(self, t):
        x = self.radius_x * np.cos(2 * np.pi / self.cycle_time * t)
        y = self.radius_y * np.sin(2 * np.pi / self.cycle_time * t)
        z = np.zeros_like(x)
        return np.array((x, y, z))
    
    def get_end_orientation(self, t):
        if np.floor(t / 2 / np.pi) % 2 == 0:
            theta = t
        else:
            theta = np.arctan2(self.radius_y * np.cos(t), -self.radius_x * np.sin(t))
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    
    def get_end_transformation(self, t):
        p_r = self.get_end_position(t)
        R_z = self.get_end_orientation(t)

        H_r = np.concatenate((R_z, p_r.reshape(-1, 1)), axis=1)
        H_r = np.concatenate((H_r, np.array((0, 0, 0, 1)).reshape(1, -1)), axis=0)
        return H_r
    
    def joint_angles(self, manipulator, t):
        return t
