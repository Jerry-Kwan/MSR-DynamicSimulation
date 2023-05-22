from math import pi
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
        potentials['external'] = torch.zeros(1, device=self.device).type(self.dtype)
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

            pot_bonds, f0, f1 = self._eval_bonds(bond_dist, bond_params, bond_unit_vec)

            potentials['bonds'] += pot_bonds.sum()
            forces.index_add_(0, bonds[:, 0], f0)
            forces.index_add_(0, bonds[:, 1], f1)

        if 'angles' in self.terms and self.system.angles is not None:
            _, _, r21 = self.compute_distances(pos, self.system.angles[:, [0, 1]], self.box)
            _, _, r23 = self.compute_distances(pos, self.system.angles[:, [2, 1]], self.box)

            pot_angles, f_angles = self._eval_angles(r21, r23, self.system.angle_params)

            potentials['angles'] += pot_angles.sum()

            forces.index_add_(0, self.system.angles[:, 0], f_angles[0])
            forces.index_add_(0, self.system.angles[:, 1], f_angles[1])
            forces.index_add_(0, self.system.angles[:, 2], f_angles[2])

        if 'dihedrals' in self.terms and self.system.dihedrals is not None:
            _, _, r12 = self.compute_distances(pos, self.system.dihedrals[:, [0, 1]], self.box)
            _, _, r23 = self.compute_distances(pos, self.system.dihedrals[:, [1, 2]], self.box)
            _, _, r34 = self.compute_distances(pos, self.system.dihedrals[:, [2, 3]], self.box)

            pot_dihedrals, f_dihedrals = self._eval_torsion(r12, r23, r34, self.system.dihedral_params)

            potentials['dihedrals'] += pot_dihedrals.sum()

            forces.index_add_(0, self.system.dihedrals[:, 0], f_dihedrals[0])
            forces.index_add_(0, self.system.dihedrals[:, 1], f_dihedrals[1])
            forces.index_add_(0, self.system.dihedrals[:, 2], f_dihedrals[2])
            forces.index_add_(0, self.system.dihedrals[:, 3], f_dihedrals[3])

        if 'impropers' in self.terms and self.system.impropers is not None:
            _, _, r12 = self.compute_distances(pos, self.system.impropers[:, [0, 1]], self.box)
            _, _, r23 = self.compute_distances(pos, self.system.impropers[:, [1, 2]], self.box)
            _, _, r34 = self.compute_distances(pos, self.system.impropers[:, [2, 3]], self.box)

            pot_impropers, f_impropers = self._eval_torsion(r12, r23, r34, self.system.improper_params)

            potentials['impropers'] += pot_impropers.sum()

            forces.index_add_(0, self.system.impropers[:, 0], f_impropers[0])
            forces.index_add_(0, self.system.impropers[:, 1], f_impropers[1])
            forces.index_add_(0, self.system.impropers[:, 2], f_impropers[2])
            forces.index_add_(0, self.system.impropers[:, 3], f_impropers[3])

    def _compute_nonbonded(self, pos, potentials, forces):
        if self.require_nonbonded and len(self.ava_idx):
            nb_dist, nb_unit_vec, _ = self.compute_distances(pos, self.ava_idx, self.box)
            ava_idx = self.ava_idx
            if self.cutoff is not None:
                nb_dist, nb_unit_vec, ava_idx = self._filter_by_cutoff(nb_dist, (nb_dist, nb_unit_vec, ava_idx))

            if 'electrostatics' in self.terms:
                pot_elec, f_elec = self._eval_electrostatics(nb_dist, ava_idx)
                potentials['electrostatics'] += pot_elec.sum()

                force_vec = nb_unit_vec * f_elec[:, None]
                forces.index_add_(0, ava_idx[:, 0], -force_vec)
                forces.index_add_(0, ava_idx[:, 1], force_vec)

            if 'lj' in self.terms:
                pot_lj, f_lj = self._eval_lj(nb_dist, ava_idx)
                potentials['lj'] += pot_lj.sum()

                force_vec = nb_unit_vec * f_lj[:, None]
                forces.index_add_(0, ava_idx[:, 0], -force_vec)
                forces.index_add_(0, ava_idx[:, 1], force_vec)

    def _compute_external(self, pos, potentials, forces):
        """Compute external potentials and forces."""
        if self.use_external:
            assert self.system.external is not None, 'External is None.'
            pos = pos.type(torch.float32)
            pot, f = self.system.external(self.system.z, pos)
            potentials['external'] += pot.detach()[0]
            forces += f.detach()

    @staticmethod
    def compute_distances(pos, idx, box):
        """Compute distances.

        How box is used? Find n such that `diff - n * box` or `diff + n * box` is the nearest value to 0 (or with
        min abs), then diff is set to this value. The formula for this operation is `diff - box * round(diff / box)`,
        regardless of whether or not diff is positive.

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

    def _eval_bonds(self, bond_dist, bond_params, bond_unit_vec):
        """Evaluate bonds.

        **WARNING**: k has been divided by 2 in parmed.

        Returns
        -------
        -f_vec is for atom0, f_vec is for atom1
        """
        k, d_0 = bond_params[:, 0], bond_params[:, 1]  # ndim of k and d_0 is 1
        x = bond_dist - d_0

        pot = k * x * x

        f = 2 * k * x
        f_vec = bond_unit_vec * f[:, None]

        return pot, -f_vec, f_vec

    def _eval_angles(self, r21, r23, angle_params):
        """Evaluate angles.

        **WARNING**: k has been divided by 2 in parmed.
        """
        k, theta_0 = angle_params[:, 0], angle_params[:, 1]

        inv_norm_r21 = 1 / torch.norm(r21, dim=1)
        inv_norm_r23 = 1 / torch.norm(r23, dim=1)
        dot_prod = torch.sum(r21 * r23, dim=1)

        cos_theta = dot_prod * inv_norm_r21 * inv_norm_r23
        cos_theta = torch.clamp(cos_theta, -1, 1)
        theta = torch.acos(cos_theta)

        delta_theta = theta - theta_0
        pot = k * delta_theta * delta_theta  # maybe **2 can not be tracked by torch

        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
        coef = torch.zeros_like(sin_theta)
        non_zero = sin_theta != 0
        coef[non_zero] = -2.0 * k[non_zero] * delta_theta[non_zero] / sin_theta[non_zero]

        force_0 = cos_theta[:, None] * r21 * inv_norm_r21[:, None] - r23 * inv_norm_r23[:, None]
        force_0 = coef[:, None] * force_0 * inv_norm_r21[:, None]

        force_2 = cos_theta[:, None] * r23 * inv_norm_r23[:, None] - r21 * inv_norm_r21[:, None]
        force_2 = coef[:, None] * force_2 * inv_norm_r23[:, None]

        force_1 = -(force_0 + force_2)

        return pot, (force_0, force_1, force_2)

    def _eval_torsion(self, r12, r23, r34, torsion_params):
        """Evaluate torsion angles (including dihedrals and impropers)."""
        crossA = torch.cross(r12, r23, dim=1)
        crossB = torch.cross(r23, r34, dim=1)
        crossC = torch.cross(r23, crossA, dim=1)
        normA = torch.norm(crossA, dim=1)
        normB = torch.norm(crossB, dim=1)
        normC = torch.norm(crossC, dim=1)
        norm_crossB = crossB / normB.unsqueeze(1)

        cos_phi = torch.sum(crossA * norm_crossB, dim=1) / normA
        sin_phi = torch.sum(crossC * norm_crossB, dim=1) / normC
        phi = -torch.atan2(sin_phi, cos_phi)

        num_torsions = len(torsion_params[0]['idx'])
        pot = torch.zeros(num_torsions, dtype=self.dtype, layout=r12.layout, device=self.device)
        coeff = torch.zeros(num_torsions, dtype=self.dtype, layout=r12.layout, device=self.device)

        for i in range(0, len(torsion_params)):
            idx = torsion_params[i]['idx']
            k_0 = torsion_params[i]['params'][:, 0]
            phi_0 = torsion_params[i]['params'][:, 1]
            per = torsion_params[i]['params'][:, 2]

            if torch.all(per > 0):  # AMBER torsions
                angle_diff = per * phi[idx] - phi_0
                pot.scatter_add_(0, idx, k_0 * (1 + torch.cos(angle_diff)))
                coeff.scatter_add_(0, idx, -per * k_0 * torch.sin(angle_diff))
            else:  # CHARMM torsions
                angle_diff = phi[idx] - phi_0
                angle_diff[angle_diff < -pi] = angle_diff[angle_diff < -pi] + 2 * pi
                angle_diff[angle_diff > pi] = angle_diff[angle_diff > pi] - 2 * pi
                pot.scatter_add_(0, idx, k_0 * angle_diff**2)
                coeff.scatter_add_(0, idx, 2 * k_0 * angle_diff)

        # Taken from OpenMM
        normDelta2 = torch.norm(r23, dim=1)
        norm2Delta2 = normDelta2**2

        force_factor_0 = (-coeff * normDelta2) / (normA**2)
        force_factor_1 = torch.sum(r12 * r23, dim=1) / norm2Delta2
        force_factor_2 = torch.sum(r34 * r23, dim=1) / norm2Delta2
        force_factor_3 = (coeff * normDelta2) / (normB**2)

        force_0_vec = force_factor_0.unsqueeze(1) * crossA
        force_3_vec = force_factor_3.unsqueeze(1) * crossB
        s = force_factor_1.unsqueeze(1) * force_0_vec - force_factor_2.unsqueeze(1) * force_3_vec

        force_0 = -force_0_vec
        force_1 = force_0_vec + s
        force_2 = force_3_vec - s
        force_3 = -force_3_vec

        return pot, (force_0, force_1, force_2, force_3)

    def _eval_electrostatics(self, dist, pair_idx):
        charges = self.system.charges
        pot = self.system.ELECTRO_FACTOR * charges[pair_idx[:, 0]] * charges[pair_idx[:, 1]] / dist

        return pot, -pot / dist

    def _eval_lj(self, dist, pair_idx):
        atom_type_idx = self.system.mapped_atom_types[pair_idx]
        a = self.system.A[atom_type_idx[:, 0], atom_type_idx[:, 1]]  # ndim is 1
        b = self.system.B[atom_type_idx[:, 0], atom_type_idx[:, 1]]

        inv_r_1 = 1 / dist
        inv_r_6 = inv_r_1**6
        inv_r_12 = inv_r_6 * inv_r_6

        pot = a * inv_r_12 - b * inv_r_6
        f = (-12 * a * inv_r_12 + 6 * b * inv_r_6) * inv_r_1

        return pot, f