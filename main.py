import pygame
import sys
import copy
import numpy as np
import sympy as sp

from robot import Joint, Manipulator
from controller import ZeroController


pygame.init()

width = 800
height = 600
screen = pygame.display.set_mode((width, height))
screen_scale = 100

white = (200, 200, 200)
grey = (100, 100, 100)

joint1 = Joint(-sp.pi / 4, 0, 2, 0, 1, 10)
joint2 = Joint(sp.pi / 4, 0, 2, 0, 1, 10)
joint3 = Joint(sp.pi / 4, 0, 1.5, 0, 1, 10)
manipulator = Manipulator(joint1, joint2)

manipulator.compute_dynamics()

controller = ZeroController()


def main():
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 28)
    
    time = 0

    butcher_A = np.array([
        [0, 0, 0, 0],
        [1 / 2, 0, 0, 0],
        [0, 1 / 2, 0, 0],
        [0, 0, 1, 0]
    ])
    butcher_b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    butcher_c = np.array([0, 1 / 2, 1 / 2, 1])
    k = np.zeros((2 * len(manipulator.joints), len(butcher_b)))

    q = np.zeros(len(manipulator.joints))
    q_dot = np.zeros(len(manipulator.joints))

    q_r = np.zeros(len(manipulator.joints))
    
    q_r_prev = np.zeros(len(manipulator.joints))
    q_r_dot_prev = np.zeros(len(manipulator.joints))

    fps = 0 # 0 = uncapped

    run = True
    while run:
        delta_time = clock.tick(fps) / 1000
        if delta_time == 0:
            delta_time = 0.001
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False


        # Reference trajectory to follow
        x_r = 3 * np.cos(time)
        y_r = 2 * np.sin(time)
        z_r = 0
        p_r = np.array((x_r, y_r, z_r))

        # Reference orientation of end-effector
        if np.floor(time / 2 / np.pi) % 2 == 0:
            theta = time
        else:
            theta = np.arctan2(200 * np.cos(time), -300 * np.sin(time))
        R_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        H_r = np.concatenate((R_z, p_r.reshape(-1, 1)), axis=1)
        H_r = np.concatenate((H_r, np.array((0, 0, 0, 1)).reshape(1, -1)), axis=0)
        
        q_r = manipulator.inverse_kinematics(H_r, q_r)

        q_r_dot = (q_r - q_r_prev) / delta_time
        q_r_ddot = (q_r_dot - q_r_dot_prev) / delta_time
        
        q_r_prev = q_r
        q_r_dot_prev = q_r_dot


        # Control systems
        #u = np.zeros(len(manipulator.joints))
        
        # PD+ controller
        #k_p = 1000
        #k_d = 100
        #u_ff = manipulator.f(*q, *q_dot)[:, 0] + np.dot(manipulator.M(*q, *q_dot), q_r_ddot) - 0.5 * np.dot(manipulator.M_dot(*q, *q_dot), q_dot - q_r_dot)
        #u_fb = - k_p * (q - q_r) - k_d * (q_dot - q_r_dot)
        #u = u_ff + u_fb

        # Feedback linearization + full state feedback
        #k_p = 1
        #k_d = 1
        #v = q_r_ddot - k_p * (q - q_r - k_d * (q_dot - q_r_dot))
        #u = np.dot(manipulator.M(*q, *q_dot), v) + manipulator.f(*q, *q_dot)[:, 0]

        u = controller.get(q, q_dot)


        # Euler integration
        #dynamics = manipulator.dynamics(q, q_dot, u)
        #q += dynamics[:len(manipulator.joints)] * delta_time
        #q_dot += dynamics[len(manipulator.joints):] * delta_time

        # Explicit Runge-Kutta
        for i in range(len(butcher_b)):
            qq = delta_time * np.dot(k, butcher_A[i, :])
            k[:, i] = manipulator.dynamics(q + qq[:len(manipulator.joints)], q_dot + qq[len(manipulator.joints):], u)
        
        qq = delta_time * np.dot(k, butcher_b)
        q += qq[:len(manipulator.joints)]
        q_dot += qq[len(manipulator.joints):]

        time += delta_time
        
        
        screen.fill((0, 0, 0))

        x_prev = 0
        y_prev = 0
        transformations = manipulator.forward_transformations(q_r)
        for H in transformations[1:]:
            x, y = H[0:2, 3] * screen_scale
            pygame.draw.line(screen, grey, (width // 2 + x_prev, height // 2 + y_prev), (width // 2 + x, height // 2 + y), width=5)
            pygame.draw.circle(screen, grey, (width // 2 + x_prev, height // 2 + y_prev), 10)
            x_prev, y_prev = x, y
        pygame.draw.circle(screen, grey, (width // 2 + x_prev, height // 2 + y_prev), 10)

        x_prev = 0
        y_prev = 0
        transformations = manipulator.forward_transformations(q)
        for H in transformations[1:]:
            x, y = H[0:2, 3] * screen_scale
            pygame.draw.line(screen, white, (width // 2 + x_prev, height // 2 + y_prev), (width // 2 + x, height // 2 + y), width=5)
            pygame.draw.circle(screen, white, (width // 2 + x_prev, height // 2 + y_prev), 10)
            x_prev, y_prev = x, y
        pygame.draw.circle(screen, white, (width // 2 + x_prev, height // 2 + y_prev), 10)

        pygame.draw.ellipse(screen, (100, 0, 0), (width // 2 - 3 * screen_scale, height // 2 - 2 * screen_scale, 2 * 3 * screen_scale, 2 * 2 * screen_scale), width=1)
        pygame.draw.circle(screen, (100, 0, 0), (width // 2 + x_r * screen_scale, height // 2 + y_r * screen_scale), 10)

        fps = clock.get_fps()
        fps_text = font.render("FPS: {:.0f}".format(fps), True, white)
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()

    sys.exit()

if __name__ == "__main__":
    main()

