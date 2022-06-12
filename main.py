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

joint1 = Joint(-sp.pi / 2, 0, 150, 0)
joint2 = Joint(sp.pi / 4, 0, 150, 0)
joint3 = Joint(sp.pi / 4, 0, 150, 0)
joints = [joint1, joint2, joint3]
manipulator = Manipulator(joints)


def main():
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 28)
    
    time = 0

    q_r = np.zeros(3)

    run = True
    while run:
        delta_time = clock.tick() / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Reference trajectory to follow
        x_r = 300 * np.cos(time)
        y_r = 200 * np.sin(time)
        z_r = 0
        p_r = np.array((x_r, y_r, z_r))

        # Reference orientation of end-effector
        theta = time
        #theta = np.arctan2(200 * np.cos(time), -300 * np.sin(time))
        R_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        H_r = np.concatenate((R_z, p_r.reshape(-1, 1)), axis=1)
        H_r = np.concatenate((H_r, np.array((0, 0, 0, 1)).reshape(1, -1)), axis=0)
        
        q_r = manipulator.inverse_kinematics(H_r, q_r)

        time += delta_time
        
        
        screen.fill((0, 0, 0))

        x_prev = 0
        y_prev = 0
        transformations = manipulator.forward_transformations(q_r)
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

