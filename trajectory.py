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
    def get_joints(self, manipulator, t):
        pass


class EllipseTrajectory2D(AbstractTrajectory):

    def __init__(self, cycle_time, radius_x, radius_y):
        self.cycle_time = cycle_time
        self.radius_x = radius_x
        self.radius_y = radius_y

        self.q_r = None
        self.q_r_dot = None
    
    def get_end_position(self, t):
        x = self.radius_x * np.cos(2 * np.pi / self.cycle_time * t)
        y = self.radius_y * np.sin(2 * np.pi / self.cycle_time * t)
        z = np.zeros_like(x)
        return np.array((x, y, z))
    
    def get_end_orientation(self, t):
        if np.floor(t / self.cycle_time) % 2 == 0:
            theta = 2 * np.pi / self.cycle_time * t
        else:
            theta = np.arctan2(self.radius_y * np.cos(2 * np.pi / self.cycle_time * t), -self.radius_x * np.sin(2 * np.pi / self.cycle_time * t))
        
        return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    
    def get_end_transformation(self, t):
        p_r = self.get_end_position(t)
        R_z = self.get_end_orientation(t)

        H_r = np.concatenate((R_z, p_r.reshape(-1, 1)), axis=1)
        H_r = np.concatenate((H_r, np.array((0, 0, 0, 1)).reshape(1, -1)), axis=0)
        return H_r
    
    def get_joints(self, manipulator, t, delta_time):
        H_r = self.get_end_transformation(t)

        q_r_prev = self.q_r
        if q_r_prev is None:
            q_r_prev = np.zeros(len(manipulator.joints))
        self.q_r = manipulator.inverse_kinematics(H_r, q_r_prev.copy())

        if self.q_r_dot is None:
            self.q_r_dot = np.zeros(len(manipulator.joints))
        q_r_dot_prev = self.q_r_dot.copy()
        self.q_r_dot = (self.q_r - q_r_prev) / delta_time
        q_r_ddot = (self.q_r_dot - q_r_dot_prev) / delta_time

        #for i in range(len(manipulator.joints)):
            #self.q_r[i] -= manipulator.joints[i].theta
        
        #return self.q_r, self.q_r_dot, q_r_ddot
        return self.q_r, self.q_r_dot, q_r_ddot
