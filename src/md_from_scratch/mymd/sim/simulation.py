import numpy as np
import torch

from .sim_constants import BOLTZMANN


class Simulation(object):
    """This class controls the whole simulation procedure.

    Imitating openmm.app.simulation.Simulation

    Including:
        1. TOBEDONE
    """

    def __init__(self, mol, system, integrator, device, dtype):
        self.mol = mol
        self.system = system
        self.integrator = integrator
        self.device = device
        self.dtype = dtype

        self._build_simulation()

    def _build_simulation(self):
        self.system.set_device_and_dtype(self.device, self.dtype)

        n = self.system.num_atoms
        self.pos = torch.zeros(n, 3).type(self.dtype).to(self.device)
        self.vel = torch.zeros(n, 3).type(self.dtype).to(self.device)
        self.forces = torch.zeros(n, 3).type(self.dtype).to(self.device)

    def set_positions(self, pos):
        assert pos.shape == (self.system.num_atoms, 3, 1), f'Shape of pos is not {(self.system.num_atoms, 3, 1)}'

        pos = np.squeeze(pos, 2)
        self.pos[:] = torch.tensor(pos, dtype=self.dtype, device=self.device)

    def set_velocities_to_temperature(self, T):
        """
        Set the velocities of all particles in the System to random values chosen from a
        Maxwell-Boltzmann distribution at a given temperature.
        """
        std_normal_dist = torch.randn((self.system.num_atoms, 3)).type(self.dtype)
        mb_dist = torch.sqrt(T * BOLTZMANN / self.system.masses) * std_normal_dist

        self.vel[:] = mb_dist.type(self.dtype).to(self.device)
