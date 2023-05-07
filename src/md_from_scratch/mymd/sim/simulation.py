import numpy as np
import torch

from .sim_constants import BOLTZMANN


class Simulation(object):
    """This class controls the whole simulation procedure.

    Imitating openmm.app.simulation.Simulation

    Including:
        1. TOBEDONE
    """

    BONDED_TERMS = ['bonds', 'angles', 'dihedrals', 'impropers']
    NONBONDED_TERMS = ['lj', 'electrostatics']
    TERMS = BONDED_TERMS + NONBONDED_TERMS

    def __init__(self, mol, system, integrator, device, dtype, use_external=False, sim_terms=TERMS):
        self.mol = mol
        self.system = system
        self.integrator = integrator
        self.device = device
        self.dtype = dtype
        self.use_external = use_external

        assert set(sim_terms) <= set(self.TERMS), 'Some of terms are not implemented.'
        self.sim_terms = sim_terms

        self._build_simulation()

    def _build_simulation(self):
        self.system.set_device_and_dtype(self.device, self.dtype)

        n = self.system.num_atoms
        self.pos = torch.zeros(n, 3).type(self.dtype).to(self.device)
        self.vel = torch.zeros(n, 3).type(self.dtype).to(self.device)

        # forces is a tensor with shape (n, 3) in self.dtype and self.device
        # potentials is a dict storing the value of potentials in Python scalars
        # see Forces.compute_potentials_and_forces for more details about these two variables
        self.forces = None
        self.potentials = None

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
