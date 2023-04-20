import scipy as sci
import numpy as np

# define universal gravitation constant
G = 6.67408e-11  # Nm2 / kg2

# reference quantities
m_nd = 1.989e+30  # kg, mass of the sun
r_nd = 5.326e+12  # m, distance between stars in Alpha Centauri
v_nd = 30000  # m / s, relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s, orbital period of Alpha Centauri

# net constants
K1 = G * t_nd * m_nd / (r_nd**2 * v_nd)
K2 = v_nd * t_nd / r_nd


def get_acc(args, pos, mass):
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    inv_r3 = (dx**2 + dy**2 + dz**2 + args.softening**2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

    ax = K1 * (dx * inv_r3) @ mass
    ay = K1 * (dy * inv_r3) @ mass
    az = K1 * (dz * inv_r3) @ mass

    a = np.hstack((ax, ay, az))

    return a


def euler(args, mass, init_pos, init_vel):
    time_span = np.linspace(0, args.t_end, args.num_samples)
    pos_sol = np.zeros((args.num_samples, args.num_particles, 3))
    vel_sol = np.zeros((args.num_samples, args.num_particles, 3))

    pos_sol[0] = init_pos
    vel_sol[0] = init_vel

    for i in range(1, args.num_samples):
        dt = time_span[i] - time_span[i - 1]
        acc = get_acc(args, pos_sol[i - 1], mass)
        vel_sol[i] = vel_sol[i - 1] + acc * dt
        pos_sol[i] = pos_sol[i - 1] + K2 * vel_sol[i] * dt

    return pos_sol, vel_sol


def modified_euler(args, mass, init_pos, init_vel):
    time_span = np.linspace(0, args.t_end, args.num_samples)
    pos_sol = np.zeros((args.num_samples, args.num_particles, 3))
    vel_sol = np.zeros((args.num_samples, args.num_particles, 3))

    pos_sol[0] = init_pos
    vel_sol[0] = init_vel

    for i in range(1, args.num_samples):
        dt = time_span[i] - time_span[i - 1]
        acc = get_acc(args, pos_sol[i - 1], mass)

        pred_vel = vel_sol[i - 1] + acc * dt
        pos_sol[i] = pos_sol[i - 1] + K2 * dt * 0.5 * (vel_sol[i - 1] + pred_vel)
        vel_sol[i] = vel_sol[i - 1] + 0.5 * dt * (acc + get_acc(args, pos_sol[i], mass))

    return pos_sol, vel_sol


def leapfrog(args, mass, init_pos, init_vel):
    time_span = np.linspace(0, args.t_end, args.num_samples)
    pos_sol = np.zeros((args.num_samples, args.num_particles, 3))
    vel_sol = np.zeros((args.num_samples, args.num_particles, 3))

    pos_sol[0] = init_pos
    vel_sol[0] = init_vel

    acc = get_acc(args, init_pos, mass)

    for i in range(1, args.num_samples):
        dt = time_span[i] - time_span[i - 1]
        vel = vel_sol[i - 1] + acc * dt / 2
        pos_sol[i] = pos_sol[i - 1] + K2 * vel * dt
        acc = get_acc(args, pos_sol[i], mass)
        vel_sol[i] = vel + acc * dt / 2

    return pos_sol, vel_sol


def scipy_integrator(args, mass, init_pos, init_vel):
    def NBodyEquations(w, t, mass, softening):
        drbydt = K2 * w[3 * args.num_particles:]

        pos = w[:3 * args.num_particles].reshape(args.num_particles, 3)

        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]

        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
        inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

        ax = K1 * (dx * inv_r3) @ mass
        ay = K1 * (dy * inv_r3) @ mass
        az = K1 * (dz * inv_r3) @ mass

        dvbydt = np.hstack((ax, ay, az)).flatten()

        return np.concatenate((drbydt, dvbydt))

    init_params = np.array([init_pos, init_vel]).flatten()
    time_span = np.linspace(0, args.t_end, args.num_samples)

    n_body_sol = sci.integrate.odeint(NBodyEquations, init_params, time_span, args=(mass, args.softening))
    pos_sol = n_body_sol[:, :3 * args.num_particles].reshape(-1, args.num_particles, 3)
    vel_sol = n_body_sol[:, 3 * args.num_particles:].reshape(-1, args.num_particles, 3)
    return pos_sol, vel_sol
