import numpy as np
import torch


class Forces(object):
    """This class conducts the computation of potentials and forces."""

    BONDED_TERMS = ['bonds', 'angles', 'dihedrals', 'impropers']
    NONBONDED_TERMS = ['lj', 'electrostatics']
    TERMS = BONDED_TERMS + NONBONDED_TERMS

    def __init__(self, system, device, dtype, terms=TERMS, exclusions=['bonds', 'angles'], use_external=False):
        """Create a forces object, whose attributes are used in computation.

        Parameters
        ----------
        terms: list=TERMS
            A list of terms being computed in simulation.
        exclusions: list=['bonds', 'angles']
            A list of exclusive terms used for the computation of nonbonded forces.
        use_external: bool=False
            Whether use external forces or not.
        """
        self.system = system
        self.device = device
        self.dtype = dtype

        assert set(terms) <= set(self.TERMS), 'Some of terms are not implemented.'
        self.terms = terms

        # make nonbonded-force distance indices if required
        self.require_nonbonded = any(f in self.NONBONDED_TERMS for f in terms)
        self.ava_idx = None
        if self.require_nonbonded:
            self.ava_idx = self._make_nonbonded_dist_indices(system.num_atoms, system.get_exclusions(exclusions),
                                                             device)

        self.use_external = use_external

    def _make_nonbonded_dist_indices(self, num_atoms, exclude_pairs, device):
        """Make distance computation indices for nonbonded forces.

        Return: shape (num_pairs_needed_computation_for_nonbonded_forces, 2)
        """
        full_mat = np.full((num_atoms, num_atoms), True, dtype=bool)
        if len(exclude_pairs):
            exclude_pairs = np.array(exclude_pairs)
            full_mat[exclude_pairs[:, 0], exclude_pairs[:, 1]] = False
            full_mat[exclude_pairs[:, 1], exclude_pairs[:, 0]] = False

        full_mat = np.triu(full_mat, +1)
        allvsall_indeces = np.vstack(np.where(full_mat)).T
        ava_idx = torch.tensor(allvsall_indeces).to(device)

        return ava_idx

    def compute_potentials_and_forces(self, pos):
        """Compute potentials and forces."""
        pass  # TOBEDONE
