import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from integrator import euler, modified_euler, leapfrog, scipy_integrator


def get_args():
    parser = argparse.ArgumentParser(description='N-Body Simulation')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-particles', default=3, type=int, help='number of particles')
    parser.add_argument('--softening', type=float, default=0.1, help='softening')
    parser.add_argument('--num-samples', default=500, type=int, help='number of samples')
    parser.add_argument('--t-begin', type=float, default=0.0, help='begining time')
    parser.add_argument('--t-end', type=float, default=20.0, help='end time')
    parser.add_argument('--integrator',
                        type=str,
                        default='leapfrog',
                        choices=['euler', 'modified_euler', 'leapfrog', 'scipy'],
                        help='which integrator to use')
    parser.add_argument('--num-tail', default=50, type=int, help='number of particles in tail in animation')
    parser.add_argument('--interval', default=30, type=int, help='delay between frames in milliseconds')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)

    # used for debug
    # mass = np.array([[1.1], [0.907], [1.0]])
    mass = np.random.uniform(0.8, 1.2, (args.num_particles, 1))
    print(mass, end='\n\n')

    init_pos = np.random.randn(args.num_particles, 3)
    init_vel = np.random.randn(args.num_particles, 3) * 0.05
    # used for debug
    # init_pos = np.array([
    #     [-0.5, 0, 0],
    #     [0.5, 0, 0],
    #     [0, 1., 0]
    # ])
    # init_vel = np.array([
    #     [0.01, 0.01, 0],
    #     [-0.05, 0, -0.1],
    #     [0, -0.01, 0]
    # ])
    print(init_pos[:2, :])
    print(init_vel[:2, :], end='\n\n')

    if args.integrator == 'euler':
        pos_sol, vel_sol = euler(args, mass, init_pos, init_vel)
    elif args.integrator == 'modified_euler':
        pos_sol, vel_sol = modified_euler(args, mass, init_pos, init_vel)
    elif args.integrator == 'leapfrog':
        pos_sol, vel_sol = leapfrog(args, mass, init_pos, init_vel)
    else:
        pos_sol, vel_sol = scipy_integrator(args, mass, init_pos, init_vel)

    print(pos_sol[0, :2, :])
    print(vel_sol[0, :2, :])
    print(pos_sol.shape, vel_sol.shape)
    print(pos_sol[-2:, :2, :], end='\n\n')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    ax.set_zlabel('z-coordinate')
    ax.set_title(f'Visualization of orbits of stars in a n-bodysystem\n(with {args.integrator} integrator)')

    color_choices = ['darkblue', 'tab:red', 'green',
                     'tab:cyan', 'tab:brown', 'tab:olive',
                     'tab:orange', 'tab:pink']

    if args.num_particles > 8:
        tra = [ax.plot([], [], [], color='darkblue')[0] for i in range(args.num_particles)]
        p = [ax.plot([], [], [], color='darkblue', marker="o")[0] for i in range(args.num_particles)]
    else:
        tra = [ax.plot([], [], [], color=color_choices[i])[0] for i in range(args.num_particles)]
        p = [ax.plot([], [], [], color=color_choices[i], marker="o")[0] for i in range(args.num_particles)]

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    def init():
        for i in range(args.num_particles):
            tra[i].set_data([], [])
            tra[i].set_3d_properties([], 'z')
            p[i].set_data([], [])
            p[i].set_3d_properties([], 'z')

        return tra + p

    def animate(i):
        for j in range(args.num_particles):
            p[j].set_data([pos_sol[i - 1, j, 0]], [pos_sol[i - 1, j, 1]])
            p[j].set_3d_properties(pos_sol[i - 1, j, 2], 'z')

        # draw the tails
        if i <= args.num_tail:
            for j in range(args.num_particles):
                tra[j].set_data(pos_sol[:i, j, 0], pos_sol[:i, j, 1])
                tra[j].set_3d_properties(pos_sol[:i, j, 2], 'z')
        else:
            for j in range(args.num_particles):
                tra[j].set_data(pos_sol[i - args.num_tail:i, j, 0], pos_sol[i - args.num_tail:i, j, 1])
                tra[j].set_3d_properties(pos_sol[i - args.num_tail:i, j, 2], 'z')

        return tra + p

    anim = FuncAnimation(fig,
                         animate,
                         init_func=init,
                         frames=list(range(1, args.num_samples + 1)),
                         blit=True,
                         interval=args.interval)
    plt.show()
