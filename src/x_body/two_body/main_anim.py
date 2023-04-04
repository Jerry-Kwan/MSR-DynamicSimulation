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

# Define masses
m1 = 1.1    # Alpha Centauri A
m2 = 0.907  # Alpha Centauri B

# Define initial position vectors
r1 = [-0.5, 0, 0]  # m
r2 = [0.5, 0, 0]   # m

# Convert pos vectors to arrays
r1 = np.array(r1, dtype="float64")
r2 = np.array(r2, dtype="float64")

# Find Centre of Mass
r_com = (m1 * r1 + m2 * r2) / (m1 + m2)

# Define initial velocities
v1 = [0.01, 0.01, 0]  # m/s
v2 = [-0.05, 0, -0.1]  # m/s

# Convert velocity vectors to arrays
v1 = np.array(v1, dtype="float64")
v2 = np.array(v2, dtype="float64")

# Find velocity of COM
v_com = (m1 * v1 + m2 * v2) / (m1 + m2)


# A function defining the equations of motion
def TwoBodyEquations(w, t, G, m1, m2):
    r1 = w[:3]
    r2 = w[3:6]
    v1 = w[6:9]
    v2 = w[9:12]

    r = sci.linalg.norm(r2 - r1)  # Calculate magnitude or norm of vector

    dv1bydt = K1 * m2 * (r2 - r1) / r**3
    dv2bydt = K1 * m1 * (r1 - r2) / r**3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2

    r_derivs = np.concatenate((dr1bydt, dr2bydt))
    derivs = np.concatenate((r_derivs, dv1bydt, dv2bydt))
    return derivs


# Package initial parameters
N = 500
init_params = np.array([r1, r2, v1, v2])  # create array of initial params
init_params = init_params.flatten()       # flatten array to make it 1D
time_span = np.linspace(0, 8, N)          # 8 orbital periods and 500 points

# Run the ODE solver
two_body_sol = sci.integrate.odeint(TwoBodyEquations, init_params, time_span, args=(G, m1, m2))

r1_sol = two_body_sol[:, :3]
r2_sol = two_body_sol[:, 3:6]
print(r1_sol[:10])
print()
print(r2_sol[:10])

# Create figure
fig = plt.figure()

# Create 3D axes
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("x-coordinate")
ax.set_ylabel("y-coordinate")
ax.set_zlabel("z-coordinate")
ax.set_title("Visualization of orbits of stars in a two-bodysystem\n")
# ax.legend()

# Initialize two lines variable
tra1, = ax.plot([], [], [], color='darkblue')
tra2, = ax.plot([], [], [], color='tab:red')
p1, = ax.plot([], [], [], color='darkblue', marker="o")
p2, = ax.plot([], [], [], color='tab:red', marker="o")

# Limit
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])


# data which the line will contain (x, y, z)
def init():
    tra1.set_data([], [])
    tra1.set_3d_properties([], 'z')
    tra2.set_data([], [])
    tra2.set_3d_properties([], 'z')

    p1.set_data([], [])
    p1.set_3d_properties([], 'z')
    p2.set_data([], [])
    p2.set_3d_properties([], 'z')
    return tra1, tra2, p1, p2


def animate(i):
    tra1.set_data(r1_sol[:i, 0], r1_sol[:i, 1])
    tra1.set_3d_properties(r1_sol[:i, 2], 'z')

    tra2.set_data(r2_sol[:i, 0], r2_sol[:i, 1])
    tra2.set_3d_properties(r2_sol[:i, 2], 'z')

    p1.set_data(r1_sol[i - 1, 0], r1_sol[i - 1, 1])
    p1.set_3d_properties(r1_sol[i - 1, 2], 'z')
    p2.set_data(r2_sol[i - 1, 0], r2_sol[i - 1, 1])
    p2.set_3d_properties(r2_sol[i - 1, 2], 'z')

    return tra1, tra2, p1, p2


anim = FuncAnimation(fig, animate, init_func=init, frames=list(range(1, N + 1)), blit=True, interval=20)

plt.show()
