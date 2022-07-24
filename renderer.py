import pygame
import numpy as np


class Renderer:

    color_white = (200, 200, 200)
    color_grey = (100, 100, 100)
    color_red = (150, 0, 0)

    def __init__(self, width, height, scale, fps=0):
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps

        pygame.init()
        
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
    
    def render_manipulator(self, manipulator, q, color):
        x_prev = 0
        y_prev = 0
        transformations = manipulator.forward_transformations(q)
        for H in transformations[1:]:
            x, y = H[0:2, 3] * self.scale
            pygame.draw.line(self.screen, color, (self.width // 2 + x_prev, self.height // 2 + y_prev), (self.width // 2 + x, self.height // 2 + y), width=5)
            pygame.draw.circle(self.screen, color, (self.width // 2 + x_prev, self.height // 2 + y_prev), 10)
            x_prev, y_prev = x, y
        pygame.draw.circle(self.screen, color, (self.width // 2 + x_prev, self.height // 2 + y_prev), 10)
    
    def render_trajectory(self, time, trajectory, trajectory_points):
        cycle_times = np.linspace(time, time + trajectory.cycle_time, trajectory_points)
        x_r, y_r, z_r = trajectory.get_end_position(cycle_times)
        
        for i in range(trajectory_points - 1):
            pygame.draw.line(self.screen, Renderer.color_red, (self.width // 2 + x_r[i] * self.scale, self.height // 2 + y_r[i] * self.scale), (self.width // 2 + x_r[i + 1] * self.scale, self.height // 2 + y_r[i + 1] * self.scale), width=2)
        pygame.draw.line(self.screen, Renderer.color_red, (self.width // 2 + x_r[-1] * self.scale, self.height // 2 + y_r[-1] * self.scale), (self.width // 2 + x_r[0] * self.scale, self.height // 2 + y_r[0] * self.scale), width=2)

        pygame.draw.circle(self.screen, Renderer.color_red, (self.width // 2 + x_r[0] * self.scale, self.height // 2 + y_r[0] * self.scale), 10)
    
    def render(self, manipulator, q, q_r=None, trajectory=None, trajectory_points=100):
        time = pygame.time.get_ticks() / 1000
        delta_time = self.clock.tick(self.fps) / 1000
        
        run = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        self.screen.fill((0, 0, 0))

        if trajectory is not None:
            self.render_trajectory(time, trajectory, trajectory_points)

        if q_r is not None:
            self.render_manipulator(manipulator, q_r, Renderer.color_grey)

        self.render_manipulator(manipulator, q, Renderer.color_white)

        fps = self.clock.get_fps()
        fps_text = self.font.render(f"FPS: {fps:.0f}", True, Renderer.color_white)
        self.screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        
        return time, delta_time, run
