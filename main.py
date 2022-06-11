import pygame
import sys
import copy
import numpy as np
import sympy as sp

from robot import *


pygame.init()

width = 800
height = 600
screen = pygame.display.set_mode((width, height))

white = (200, 200, 200)

joint1 = Joint(-np.pi / 2, 0, 200, 0)
joint2 = Joint(np.pi / 4, 0, 200, 0)
joint3 = Joint(np.pi / 4, 0, 150, 0)
joints = [joint1, joint2, joint3]
manipulator = Manipulator(joints)
manipulator_clone = copy.deepcopy(manipulator)


"""def inverse_kinematics_position(manipulator, x_r, y_r, z_r):
    q = manipulator.get_q()

    k = 0.01

    for i in range(10):
        J_omega, J_v = manipulator.jacobians()
        J_v_inv = np.linalg.pinv(J_v)

        transformations = manipulator.forward_transformations()
        H = transformations[-1]
        x, y, z = H[0:3, 3]

        e = np.array([x_r - x, y_r - y, z_r - z])
        dq = np.dot(J_v_inv, e)
        q += k * dq
        manipulator.set_q(q)
    
    return q"""

# Numerical inverse kinematics using Newton iterations
# Works for both position and orientation
def inverse_kinematics(manipulator, H_r):
    q = manipulator.get_q()

    k = 0.1
    iterations = 1

    for i in range(iterations):
        J_omega, J_v = manipulator.jacobians()
        J = np.concatenate((J_omega, J_v))
        J_inv = np.linalg.pinv(J)

        transformations = manipulator.forward_transformations()
        H = transformations[-1]

        R = H[0:3, 0:3]
        R_r = H_r[0:3, 0:3]

        # Poisson equation for rotation matrices solved for angular velocity omega
        # The time derivative is approximated as the difference from the reference to the current rotation
        S = np.dot((R_r - R), R.T)
        omega_e = np.array([-S[1, 2], S[0, 2], -S[0, 1]])

        # Also approximate the time derivative as the difference
        x, y, z = H[0:3, 3]
        x_r, y_r, z_r = H_r[0:3, 3]
        v_e = np.array([x_r - x, y_r - y, z_r - z])
        
        dq = np.dot(J_inv, np.concatenate((omega_e, v_e)))
        q += k * dq
        manipulator.set_q(q)
    
    return q



def main():
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 28)
    
    t = 0

    run = True
    while run:
        delta_time = clock.tick() / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Reference trajectory to follow
        x_r = 300 * np.cos(t)
        y_r = 200 * np.sin(t)
        z_r = 0
        p_r = np.array((x_r, y_r, z_r))

        # Reference orientation of end-effector
        theta = t
        #theta = np.arctan2(200 * np.cos(t), -300 * np.sin(t))
        R_z = rotationZ(theta)

        H_r = np.concatenate((R_z, p_r.reshape(-1, 1)), axis=1)
        H_r = np.concatenate((H_r, np.array((0, 0, 0, 1)).reshape(1, -1)), axis=0)
        q_r = inverse_kinematics(manipulator_clone, H_r)
        
        manipulator.set_q(q_r)

        t += delta_time
        
        
        screen.fill((0, 0, 0))

        x_prev = 0
        y_prev = 0
        transformations = manipulator.forward_transformations()
        for H in transformations[1:]:
            x, y = H[0:2, 3]
            pygame.draw.line(screen, white, (width // 2 + x_prev, height // 2 + y_prev), (width // 2 + x, height // 2 + y), width=5)
            pygame.draw.circle(screen, white, (width // 2 + x_prev, height // 2 + y_prev), 10)
            x_prev, y_prev = x, y
        pygame.draw.circle(screen, white, (width // 2 + x_prev, height // 2 + y_prev), 10)


        pygame.draw.ellipse(screen, (100, 0, 0), (width // 2 - 300, height // 2 - 200, 600, 400), width=1)
        pygame.draw.circle(screen, (100, 0, 0), (width // 2 + x_r, height // 2 + y_r), 10)

        fps = clock.get_fps()
        fps_text = font.render("FPS: {:.0f}".format(fps), True, white)
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()


    sys.exit()

if __name__ == "__main__":
    main()

