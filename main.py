import sys
import numpy as np
import sympy as sp

from robot import Joint, Manipulator
from controller import PDController, FeedbackLinearizationController
from renderer import Renderer
from trajectory import EllipseTrajectory2D
from integrator import ExplicitEuler, ExplicitRK4


width = 800
height = 600
scale = 100
renderer = Renderer(width, height, scale)


joint1 = Joint(-sp.pi / 4, 0, 2, 0, 1, 10)
joint2 = Joint(sp.pi / 4, 0, 2, 0, 1, 10)
joint3 = Joint(sp.pi / 4, 0, 1.5, 0, 1, 10)
manipulator = Manipulator(joint1, joint2, joint3)

manipulator.compute_dynamics()


trajectory = EllipseTrajectory2D(cycle_time=5, radius_x=3, radius_y=2)


controller = PDController(k_p=10000, k_d=1000)
#controller = FeedbackLinearizationController(k_p=1000, k_d=100, manipulator=manipulator)


#integrator = ExplicitEuler(len(manipulator.joints))
integrator = ExplicitRK4(len(manipulator.joints))


def main():
    q = np.zeros(len(manipulator.joints))
    q_dot = np.zeros(len(manipulator.joints))

    q_r, q_r_dot, q_r_ddot = trajectory.get_joints(manipulator, 0, 0.01)

    renderer.init()

    run = True
    while run:
        time, delta_time, run = renderer.render(manipulator, q, q_r=q_r, trajectory=trajectory)
        if delta_time == 0:
            continue

        q_r, q_r_dot, q_r_ddot = trajectory.get_joints(manipulator, time, delta_time)

        u = controller.get(q, q_dot, q_r, q_r_dot, q_r_ddot)

        integrator.step(manipulator, q, q_dot, u, delta_time)

    sys.exit()

if __name__ == "__main__":
    main()

