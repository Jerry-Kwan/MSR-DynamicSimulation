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
        self.num_atoms = system.num_atoms
        self.box = system.box
        self.cutoff = system.cutoff

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

        Return: shape (num_pairs_needing_computation_for_nonbonded_forces, 2)
        """
        full_mat = np.full((num_atoms, num_atoms), True, dtype=bool)
        if len(exclude_pairs):
            exclude_pairs = np.array(exclude_pairs)
            full_mat[exclude_pairs[:, 0], exclude_pairs[:, 1]] = False
            full_mat[exclude_pairs[:, 1], exclude_pairs[:, 0]] = False

        full_mat = np.triu(full_mat, +1)
        allvsall_indices = np.vstack(np.where(full_mat)).T
        ava_idx = torch.tensor(allvsall_indices).to(device)

        return ava_idx

    def compute_potentials_and_forces(self, pos):
        """Compute potentials and forces.

        Returns
        -------
        potentials_dict
            A dict storing the value of potentials in Python scalars.
        forces
            A torch.tensor storing the value of forces with shape (num_atoms, 3) in self.dtype and self.device.
        """
        if torch.any(torch.isnan(pos)):
            raise RuntimeError('Found NaN coordinates.')

        potentials = {t: torch.zeros(1, device=self.device).type(self.dtype) for t in self.terms}
        forces = torch.zeros(self.num_atoms, 3).type(self.dtype).to(self.device)

        self._compute_bonded(pos, potentials, forces)
        self._compute_nonbonded(pos, potentials, forces)
        self._compute_external(pos, potentials, forces)

        return {k: v.cpu().item() for k, v in potentials.items()}, forces

    def _compute_bonded(self, pos, potentials, forces):
        if 'bonds' in self.terms and self.system.bonds is not None:
            bonds, bond_params = self.system.bonds, self.system.bond_params
            bond_dist, bond_unit_vec, _ = self.compute_distances(pos, bonds, self.box)

            if self.cutoff is not None:
                arrays = (bond_dist, bond_unit_vec, bonds, bond_params)
                bond_dist, bond_unit_vec, bonds, bond_params = self._filter_by_cutoff(bond_dist, arrays)

            pass  # TOBEDONE

        if 'angles' in self.terms and self.system.angles is not None:
            pass  # TOBEDONE

        if 'dihedrals' in self.terms and self.system.dihedrals is not None:
            pass  # TOBEDONE

        if 'impropers' in self.terms and self.system.impropers is not None:
            pass  # TOBEDONE

    def _compute_nonbonded(self, pos, potentials, forces):
        if self.require_nonbonded and len(self.ava_idx):
            pass  # TOBEDONE

    def _compute_external(self, pos, potentials, forces):
        if self.use_external:
            pass  # TOBEDONE

    @staticmethod
    def compute_distances(pos, idx, box):
        """Compute distances.

        How box is used? Find n such that `diff - n * box` or `diff + n * box` is the nearest value to 0, then
        diff is set to this value. The formula for this operation is `diff - box * round(diff / box)`, regardless
        of whether or not diff is positive.

        For example, if the original diff is 51 and box is 100, then new diff is -49.

        Parameters
        ----------
        pos: torch.Tensor
            The position of atoms in Tensor with shape (num_atoms, 3).
        idx: torch.Tensor
            The indices of atom pairs needing computation of distance in Tensor with shape (num_idx, 2).
        box: torch.Tensor / None
            The box size in Tensor with torch.Size([3]), could be None if not used.

        Returns
        -------
        dist: torch.Tensor
            A tensor storing the distance with torch.Size([num_idx])
        """
        dir_vec = pos[idx[:, 0]] - pos[idx[:, 1]]
        if box is not None:
            dir_vec = dir_vec - box.unsqueeze(0) * torch.round(dir_vec / box.unsqueeze(0))

        dist = torch.norm(dir_vec, dim=1)
        dir_unit_vec = dir_vec / dist.unsqueeze(1)

        return dist, dir_unit_vec, dir_vec

    def _filter_by_cutoff(self, dist, arrays):
        """Filter elements in arrays by the comparison between cutoff and dist."""
        under_cutoff = dist <= self.cutoff  # a tensor in bool
        new_arrays = []

        for arr in arrays:
            # arr[under_cutoff] just include the elements that is True in under_cutoff
            new_arrays.append(arr[under_cutoff])

        return new_arrays

    def _eval_bonds(self, bond_dist, bond_params):
        k, d0 = bond_params[:, 0], bond_params[:, 1]  # ndim of k and d0 is 1
        x = bond_dist - d0
        pot = k * (x**2)
        f = ...  # TOBEDONE
