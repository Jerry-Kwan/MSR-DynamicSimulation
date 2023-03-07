# Import scipy
import scipy as sci
import numpy as np

# Import matplotlib and associated modules for 3D and animations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define universal gravitation constant
G = 6.67408e-11  # Nm2 / kg2

# Reference quantities
m_nd = 1.989e+30  # kg, mass of the sun
r_nd = 5.326e+12  # m, distance between stars in Alpha Centauri
v_nd = 30000      # m / s, relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s, orbital period of Alpha Centauri

# Net constants
K1 = G * t_nd * m_nd / (r_nd**2 * v_nd)
K2 = v_nd * t_nd / r_nd

# Define the number of particles
N_PARTICLES = 10

# Define softening
# softening = 0.1
softening = 0.1

# Define masses
mass = np.random.uniform(0.8, 1.2, (N_PARTICLES, 1))
# mass = np.array([[1.1], [0.907], [1.0]])
print(mass)
print()

# Define initial position and velocity
init_pos = np.random.randn(N_PARTICLES, 3)
init_vel = np.random.randn(N_PARTICLES, 3) * 0.05
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
print(init_vel[:2, :])
print()


def NBodyEquations(w, t, G, mass, softening):
    drbydt = K2 * w[3 * N_PARTICLES:]

    pos = w[:3 * N_PARTICLES].reshape(N_PARTICLES, 3)

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


# Package initial parameters
N_SAMPLES = 500
T_END = 20
init_params = np.array([init_pos, init_vel]).flatten()
time_span = np.linspace(0, T_END, N_SAMPLES)

n_body_sol = sci.integrate.odeint(NBodyEquations, init_params, time_span, args=(G, mass, softening))

pos_sol = n_body_sol[:, :3 * N_PARTICLES].reshape(-1, N_PARTICLES, 3)
vel_sol = n_body_sol[:, 3 * N_PARTICLES:].reshape(-1, N_PARTICLES, 3)
print(pos_sol[0, :2, :])
print(vel_sol[0, :2, :])

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("x-coordinate")
ax.set_ylabel("y-coordinate")
ax.set_zlabel("z-coordinate")
ax.set_title("Visualization of orbits of stars in a n-bodysystem\n")

tra = [ax.plot([], [], [], color='darkblue')[0] for i in range(N_PARTICLES)]
p = [ax.plot([], [], [], color='darkblue', marker="o")[0] for i in range(N_PARTICLES)]

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])


def init():
    for i in range(N_PARTICLES):
        tra[i].set_data([], [])
        tra[i].set_3d_properties([], 'z')
        p[i].set_data([], [])
        p[i].set_3d_properties([], 'z')

    return tra + p


def animate(i):
    for j in range(N_PARTICLES):
        tra[j].set_data(pos_sol[:i, j, 0], pos_sol[:i, j, 1])
        tra[j].set_3d_properties(pos_sol[:i, j, 2], 'z')

        p[j].set_data([pos_sol[i - 1, j, 0]], [pos_sol[i - 1, j, 1]])
        p[j].set_3d_properties(pos_sol[i - 1, j, 2], 'z')

    return tra + p


anim = FuncAnimation(fig, animate, init_func=init, frames=list(range(1, N_SAMPLES + 1)), blit=True, interval=30)

plt.show()
