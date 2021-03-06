import numpy as np
import sympy as sp
import sympy.physics.mechanics as mech


def rotationX(angle):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(angle), -sp.sin(angle)],
        [0, sp.sin(angle), sp.cos(angle)],
    ])

def rotationY(angle):
    return sp.Matrix([
        [sp.cos(angle), 0, sp.sin(angle)],
        [0, 1, 0],
        [-sp.sin(angle), 0, sp.cos(angle)]
    ])

def rotationZ(angle):
    return sp.Matrix([
        [sp.cos(angle), -sp.sin(angle), 0],
        [sp.sin(angle), sp.cos(angle), 0],
        [0, 0, 1]
    ])


class Joint:

    joint_index = 0

    def __init__(self, theta, d, a, alpha, m, J):
        self.theta = theta
        self.d = d
        self.a = a
        self.alpha = alpha
        
        self.m = m
        self.J = J

        self.q = mech.dynamicsymbols("q_" + str(Joint.joint_index), real=True)
        self.q_dot = mech.dynamicsymbols("q_" + str(Joint.joint_index), 1, real=True)
        self.u = mech.dynamicsymbols("u_" + str(Joint.joint_index), real=True)
        Joint.joint_index += 1

        self.H = self.transformation()
        self.forward_transformation = sp.lambdify(self.q, self.H, "numpy")
    
    def transformation(self):
        H_z = sp.zeros(4, 4)
        H_z[0:3, 0:3] = rotationZ(self.theta + self.q)
        H_z[2, 3] = self.d
        H_z[3, 3] = 1
        
        H_x = sp.zeros(4, 4)
        H_x[0:3, 0:3] = rotationX(self.alpha)
        H_x[0, 3] = self.a
        H_x[3, 3] = 1

        H = H_z * H_x
        return H


class Manipulator:

    def __init__(self, *joints):
        self.joints = joints

    def forward_transformations(self, q):
        H = np.diag(np.ones(4))
        transformations = [H]

        for i in range(len(self.joints)):
            H_i = self.joints[i].forward_transformation(q[i])
            H = np.dot(H, H_i)
            transformations.append(H)

        return transformations
    
    def jacobians(self, q):
        J_omega = np.zeros((3, len(self.joints)))
        J_v = np.zeros((3, len(self.joints)))

        transformations = self.forward_transformations(q)

        z_0 = np.array([0, 0, 1])

        o_n = transformations[-1][0:3, 3]

        for i in range(len(self.joints)):
            z_i = np.dot(transformations[i][0:3, 0:3], z_0)
            o_i = transformations[i][0:3, 3]

            J_omega[:, i] = z_i
            J_v[:, i] = np.cross(z_i, o_n - o_i)
            
        return J_omega, J_v


    # Numerical inverse kinematics using Newton iterations
    # Works for both position and orientation
    def inverse_kinematics(self, H_r, prev_q=None):
        if prev_q is None:
            q = np.zeros(len(self.joints))
        else:
            q = prev_q

        k = 0.1
        iterations = 10
        
        R_r = H_r[0:3, 0:3]
        x_r, y_r, z_r = H_r[0:3, 3]

        for i in range(iterations):
            J_omega, J_v = self.jacobians(q)
            J = np.concatenate((J_omega, J_v))
            J_inv = np.linalg.pinv(J)

            transformations = self.forward_transformations(q)
            H = transformations[-1]
            R = H[0:3, 0:3]
            x, y, z = H[0:3, 3]

            # Poisson equation for rotation matrices solved for angular velocity omega
            # The time derivative is approximated as the difference from the reference to the current rotation
            S = np.dot((R_r - R), R.T)
            omega_e = np.array([-S[1, 2], S[0, 2], -S[0, 1]])

            # Also approximate the time derivative as the difference
            v_e = np.array([x_r - x, y_r - y, z_r - z])
            
            dq = np.dot(J_inv, np.concatenate((omega_e, v_e)))
            q += k * dq
        
        return q
    
    def compute_dynamics(self, g=9.81):
        K = 0
        P = 0

        H = sp.eye(4)
        for i in range(len(self.joints)):
            H *= self.joints[i].H

            p = H[0:3, 3]
            v = sp.diff(p)

            K += (0.5 * self.joints[i].m * v.T * v)[0]

            R = H[0:3, 0:3]
            S = sp.diff(R) * R.T
            omega = sp.Matrix([-S[1, 2], S[0, 2], -S[0, 1]])
            #K += (0.5 * omega.T * R * sp.Matrix([[0, 0, 0], [0, self.joints[i].J, 0], [0, 0, self.joints[i].J]]) * R.T * omega)[0]
            K += (0.5 * self.joints[i].J * omega.T * omega)[0]
            
            P += -self.joints[i].m * g * p[1]

        K = sp.factor(K)
        P = sp.factor(P)

        L = K - P

        q = [joint.q for joint in self.joints]
        q_dot = [joint.q_dot for joint in self.joints]

        LM = mech.LagrangesMethod(L, q)
        LM.form_lagranges_equations()

        # These simplififcations are also really slow and can take multiple minutes at startup
        # They do however make the overall program run faster while simulating
        #self.M = sp.lambdify([*q, *q_dot], sp.simplify(LM.mass_matrix), "numpy")
        #self.f = sp.lambdify([*q, *q_dot], sp.simplify(LM.forcing), "numpy")
        self.M = sp.lambdify([*q, *q_dot], LM.mass_matrix, "numpy")
        self.f = sp.lambdify([*q, *q_dot], LM.forcing, "numpy")

        try:
            #self.M_dot = sp.lambdify([*q, *q_dot], sp.simplify(sp.diff(LM.mass_matrix)), "numpy")
            self.M_dot = sp.lambdify([*q, *q_dot], sp.diff(LM.mass_matrix), "numpy")
        except ValueError:
            self.M_dot = sp.lambdify([*q, *q_dot], sp.zeros(*sp.shape(LM.mass_matrix)), "numpy")

            
    def dynamics(self, q, q_dot, u):
        M = self.M(*q, *q_dot)
        f = self.f(*q, *q_dot)[:, 0] + u
        q_ddot = np.linalg.solve(M, f)
        #print(M)
        #print(f)
        #print(q_ddot)
        #print()
        return np.concatenate((q_dot, q_ddot))
