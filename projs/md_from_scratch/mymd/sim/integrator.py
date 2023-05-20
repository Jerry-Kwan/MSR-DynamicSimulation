from .sim_constants import TIME_FACTOR


class Integrator(object):
    """Base class for Integrator.

    Used for future extension.
    """

    def __init__(self):
        pass


class VelocityVerletIntegrator(Integrator):
    """An Integrator used in simulation using Velocity Verlet Algorithm."""

    def __init__(self, step_size):
        """Create a VelocityVerletIntegrator object.

        Parameters
        ----------
        step_size : double
            The step size with which to integrate in simulation (in femtosecond, fs).
        """
        super().__init__()
        self.dt_fs = step_size
        self.dt = step_size / TIME_FACTOR

    def step(self, steps, system, pos, vel, f):
        """Advance a simulation through time by taking a series of time steps.

        Parameters
        ----------
        steps : int
            The number of time steps to take.
        """
        _, forces = f.compute_potentials_and_forces(pos)

        for _ in range(steps):
            self._vv_1(pos, vel, forces, system.masses)
            _, forces = f.compute_potentials_and_forces(pos)
            self._vv_2(vel, forces, system.masses)

    def _vv_1(self, pos, vel, forces, masses):
        acc = forces / masses
        pos += vel * self.dt + 0.5 * acc * self.dt * self.dt
        vel += 0.5 * self.dt * acc

    def _vv_2(self, vel, forces, masses):
        acc = forces / masses
        vel += 0.5 * self.dt * acc
