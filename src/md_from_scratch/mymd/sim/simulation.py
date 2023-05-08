import numpy as np
import torch

from .sim_constants import BOLTZMANN
from .forces import Forces


class Simulation(object):
    """This class controls the whole simulation procedure.

    Imitating openmm.app.simulation.Simulation

    Including:
        1. TOBEDONE
    """

    BONDED_TERMS = ['bonds', 'angles', 'dihedrals', 'impropers']
    NONBONDED_TERMS = ['lj', 'electrostatics']
    TERMS = BONDED_TERMS + NONBONDED_TERMS

    def __init__(self,
                 mol,
                 system,
                 integrator,
                 device,
                 dtype,
                 exclusions=['bonds', 'angles'],
                 use_external=False,
                 sim_terms=TERMS):
        """Create a simulation object.

        Parameters
        ----------
        exclusion: list=['bonds', 'angles']
            A list containing the exclusive force terms. If the force type of an atom pair
            is in the exclusion list, then this pair is not computed for nunbonded forces.
        sim_terms: list=TERMS
            A list containing the force terms computed in simulation.
        """
        self.mol = mol
        self.system = system
        self.integrator = integrator
        self.device = device
        self.dtype = dtype
        self.use_external = use_external

        assert set(sim_terms) <= set(self.TERMS), 'Some of terms are not implemented.'
        self.sim_terms = sim_terms

        assert set(exclusions) <= set(['bonds', 'angles']), (f'Exclusions should be the subset '
                                                             f'of {set(["bonds", "angles"])}')
        self.exclusions = exclusions

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
        self.potentials_sum = None

        self._f = Forces(self.system,
                         self.device,
                         self.dtype,
                         terms=self.sim_terms,
                         exclusions=self.exclusions,
                         use_external=self.use_external)

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

    def update_potentials_and_forces(self):
        self.potentials, self.forces = self._f.compute_potentials_and_forces(self.pos)
        self.potentials_sum = np.sum([v for _, v in self.potentials.items()])
