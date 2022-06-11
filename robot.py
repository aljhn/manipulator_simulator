import numpy as np


def rotationX(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ])

def rotationY(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rotationZ(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


class Joint:

    def __init__(self, theta, d, a, alpha):
        self.theta = theta
        self.d = d
        self.a = a
        self.alpha = alpha

        self.q = 0
    
    def forward_transformation(self):
        H_z = np.zeros((4, 4))
        H_z[0:3, 0:3] = rotationZ(self.theta + self.q)
        H_z[2, 3] = self.d
        H_z[3, 3] = 1
        
        H_x = np.zeros((4, 4))
        H_x[0:3, 0:3] = rotationX(self.alpha)
        H_x[0, 3] = self.a
        H_x[3, 3] = 1

        H = np.dot(H_z, H_x)

        return H
    
    def set(self, q):
        self.q = q


class Manipulator:

    def __init__(self, joints):
        self.joints = joints

    def forward_transformations(self):
        H = np.diag(np.ones(4))
        transformations = [H]

        for joint in self.joints:
            H_i = joint.forward_transformation()
            H = np.dot(H, H_i)
            transformations.append(H)

        return transformations
    
    def get_q(self):
        q = [joint.q for joint in self.joints]
        return np.array(q, dtype=np.float64)
    
    def set_q(self, q):
        for i in range(len(self.joints)):
            self.joints[i].set(q[i])
    
    def jacobians(self):
        J_omega = np.zeros((3, len(self.joints)))
        J_v = np.zeros((3, len(self.joints)))

        transformations = self.forward_transformations()

        z_0 = np.array([0, 0, 1])

        o_n = transformations[-1][0:3, 3]

        for i in range(len(self.joints)):
            z_i = np.dot(transformations[i][0:3, 0:3], z_0)
            o_i = transformations[i][0:3, 3]

            J_omega[:, i] = z_i
            J_v[:, i] = np.cross(z_i, o_n - o_i)
            
        return J_omega, J_v
