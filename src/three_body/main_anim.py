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
m3 = 1.0    # Third Star

# Define initial position vectors
r1 = [-0.5, 0, 0]  # m
r2 = [0.5, 0, 0]   # m
r3 = [0, 1., 0]    # m

# Convert pos vectors to arrays
r1 = np.array(r1, dtype="float64")
r2 = np.array(r2, dtype="float64")
r3 = np.array(r3, dtype="float64")

# Find Centre of Mass
r_com = (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)

# Define initial velocities
v1 = [0.01, 0.01, 0]   # m/s
v2 = [-0.05, 0, -0.1]  # m/s
v3 = [0, -0.01, 0]     # m/s

# Convert velocity vectors to arrays
v1 = np.array(v1, dtype="float64")
v2 = np.array(v2, dtype="float64")
v3 = np.array(v3, dtype="float64")

# Find velocity of COM
v_com = (m1 * v1 + m2 * v2 + m3 * v3) / (m1 + m2 + m3)


# A function defining the equations of motion
def ThreeBodyEquations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]

    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

    dv1bydt = K1 * m2 * (r2 - r1) / r12**3 + K1 * m3 * (r3 - r1) / r13**3
    dv2bydt = K1 * m1 * (r1 - r2) / r12**3 + K1 * m3 * (r3 - r2) / r23**3
    dv3bydt = K1 * m1 * (r1 - r3) / r13**3 + K1 * m2 * (r2 - r3) / r23**3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3

    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r_derivs = np.concatenate((r12_derivs, dr3bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v_derivs = np.concatenate((v12_derivs, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))

    return derivs


# Package initial parameters
N = 500
init_params = np.array([r1, r2, r3, v1, v2, v3])  # Initial parameters
init_params = init_params.flatten()                # Flatten to make 1D array
time_span = np.linspace(0, 20, N)                  # 20 orbital periods and 500 points

# Run the ODE solver
three_body_sol = sci.integrate.odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2, m3))

r1_sol = three_body_sol[:, :3]
r2_sol = three_body_sol[:, 3:6]
r3_sol = three_body_sol[:, 6:9]
print(r1_sol[:10])
print()
print(r2_sol[:10])
print()
print(r3_sol[:10])

# Create figure
fig = plt.figure()

# Create 3D axes
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("x-coordinate")
ax.set_ylabel("y-coordinate")
ax.set_zlabel("z-coordinate")
ax.set_title("Visualization of orbits of stars in a three-bodysystem\n")
# ax.legend()

# Initialize three lines variable
tra1, = ax.plot([], [], [], color='darkblue')
tra2, = ax.plot([], [], [], color='tab:red')
tra3, = ax.plot([], [], [], color='green')
p1, = ax.plot([], [], [], color='darkblue', marker="o")
p2, = ax.plot([], [], [], color='tab:red', marker="o")
p3, = ax.plot([], [], [], color='green', marker="o")

# Limit
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])


# data which the line will contain (x, y, z)
def init():
    tra1.set_data([], [])
    tra1.set_3d_properties([], 'z')
    tra2.set_data([], [])
    tra2.set_3d_properties([], 'z')
    tra3.set_data([], [])
    tra3.set_3d_properties([], 'z')

    p1.set_data([], [])
    p1.set_3d_properties([], 'z')
    p2.set_data([], [])
    p2.set_3d_properties([], 'z')
    p3.set_data([], [])
    p3.set_3d_properties([], 'z')

    return tra1, tra2, tra3, p1, p2, p3


def animate(i):
    tra1.set_data(r1_sol[:i, 0], r1_sol[:i, 1])
    tra1.set_3d_properties(r1_sol[:i, 2], 'z')

    tra2.set_data(r2_sol[:i, 0], r2_sol[:i, 1])
    tra2.set_3d_properties(r2_sol[:i, 2], 'z')

    tra3.set_data(r3_sol[:i, 0], r3_sol[:i, 1])
    tra3.set_3d_properties(r3_sol[:i, 2], 'z')

    p1.set_data([r1_sol[i - 1, 0]], [r1_sol[i - 1, 1]])
    p1.set_3d_properties(r1_sol[i - 1, 2], 'z')

    p2.set_data([r2_sol[i - 1, 0]], [r2_sol[i - 1, 1]])
    p2.set_3d_properties(r2_sol[i - 1, 2], 'z')

    p3.set_data([r3_sol[i - 1, 0]], [r3_sol[i - 1, 1]])
    p3.set_3d_properties(r3_sol[i - 1, 2], 'z')

    return tra1, tra2, tra3, p1, p2, p3


anim = FuncAnimation(fig, animate, init_func=init, frames=list(range(1, N + 1)), blit=True, interval=40)

plt.show()
