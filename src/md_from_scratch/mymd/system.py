import numpy as np
import torch
import networkx as nx


class System(object):
    """
    This class represents a molecular system (use torch).
    """

    BONDED_TERMS = ['bonds', 'angles', 'dihedrals', 'impropers']
    NONBONDED_TERMS = ['lj', 'electrostatics']
    TERMS = BONDED_TERMS + NONBONDED_TERMS

    def __init__(self, mol, ff, terms=TERMS, cutoff=None, external=None):
        """Create a system.

        Parameters
        ----------
        external: =None
            An object that describes the external forces, such as Neural Network Potential.
        """
        assert set(terms) <= set(self.TERMS), 'Some of terms are not implemented.'

        # set to None, used in set_device_and_dtype
        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None
        self.A = None
        self.B = None

        self._build_sys_params(mol, ff, terms, cutoff, external)

    def _build_sys_params(self, mol, ff, terms, cutoff, external):
        self.external = external
        self.cutoff = cutoff
        self.num_atoms = mol.numAtoms

        # build unique atom types and corresponding indices
        self.uni_atom_types, indices = np.unique(mol.atomtype, return_inverse=True)
        self.mapped_atom_types = torch.tensor(indices)

        # charges and masses
        # it is better to gain charges and masses from ff, but not mol, because this is the meaning of forcefield
        self.charges = torch.tensor(mol.charge.astype(np.float64))
        self.masses = torch.tensor(mol.masses.astype(np.float64)).unsqueeze_(1)  # (num_atoms, 1)

        uni_bonds = None  # used in bonds and impropers

        if 'bonds' in terms and len(mol.bonds):
            uni_bonds = np.unique([sorted(bond) for bond in mol.bonds], axis=0)
            self.bonds = torch.tensor(uni_bonds.astype(np.int64))  # (num_uni_bonds, 2)
            uni_bonds_atom_types = self.uni_atom_types[indices[uni_bonds]]
            self.bond_params = torch.tensor([ff.get_bond_params(*atoms)
                                             for atoms in uni_bonds_atom_types])  # (num_uni_bonds, 2), 2 is k and req

        if 'angles' in terms and len(mol.angles):
            uni_angles = np.unique([ang if ang[0] < ang[2] else ang[::-1] for ang in mol.angles], axis=0)
            self.angles = torch.tensor(uni_angles.astype(np.int64))
            uni_angles_atom_types = self.uni_atom_types[indices[uni_angles]]
            self.angle_params = torch.tensor([ff.get_angle_params(*atoms) for atoms in uni_angles_atom_types])

        if 'dihedrals' in terms and len(mol.dihedrals):
            uni_dihedrals = np.unique([dih if dih[0] < dih[3] else dih[::-1] for dih in mol.dihedrals], axis=0)
            self.dihedrals = torch.tensor(uni_dihedrals.astype(np.int64))
            uni_dihedrals_atom_types = self.uni_atom_types[indices[uni_dihedrals]]
            self.dihedral_params = self._get_dihedrals_params(ff, uni_dihedrals_atom_types)

        if 'impropers' in terms and len(mol.impropers):
            assert uni_bonds is not None, 'Impropers are implemented only when bonds are not None.'

            uni_impropers = np.unique(mol.impropers, axis=0)
            self.impropers = torch.tensor(uni_impropers.astype(np.int64))
            self.improper_params = self._get_impropers_params(ff, self.uni_atom_types, indices, uni_impropers,
                                                              uni_bonds)

        if 'lj' in terms:
            self.A, self.B = self._get_lj_params(ff, self.uni_atom_types)

        if 'electrostatics' in terms:
            self.ELECTRO_FACTOR = self.get_electrostatics_factor()

        self.box = self._get_periodic_box(mol)
        self.mol_groups, self.non_grouped = self._build_atom_groups(mol.numAtoms, mol.bonds if len(mol.bonds) else None)
        self.num_groups = len(self.mol_groups)
        self.num_non_grouped = len(self.non_grouped)

    def _get_dihedrals_params(self, ff, uni_dihedrals_atom_types):
        """
        Return dihedrals params.

        new_dihedrals: max_terms elements, each element is a dict with 'idx' and 'params', both are tensor
        """
        from collections import defaultdict

        # dihedrals[i] represents the i-th dihedral params of all dihedrals
        dihedrals = defaultdict(lambda: {'idx': [], 'params': []})

        for i, at in enumerate(uni_dihedrals_atom_types):
            terms = ff.get_dihedral_params(*at)

            for j, term in enumerate(terms):
                dihedrals[j]['idx'].append(i)
                dihedrals[j]['params'].append(term)

        # the max number of params of one dihedral
        max_terms = max(dihedrals.keys()) + 1
        new_dihedrals = []

        for j in range(max_terms):
            dihedrals[j]['idx'] = torch.tensor(dihedrals[j]['idx'])
            dihedrals[j]['params'] = torch.tensor(dihedrals[j]['params'])
            new_dihedrals.append(dihedrals[j])

        return new_dihedrals

    def _get_impropers_params(self, ff, uni_atom_types, indices, uni_impropers, bonds):
        """
        Return impropers parameteres.

        [impropers] is list, impropers is a dict with 'idx' and 'params'
        """
        impropers = {'idx': [], 'params': []}
        graph = self.get_improper_graph(uni_impropers, bonds)

        for i, impr in enumerate(uni_impropers):
            atoms = uni_atom_types[indices[impr]]
            params = None

            try:
                params = ff.get_improper_params(*atoms)
            except:
                center = self.detect_improper_center(impr, graph)
                not_center = sorted(np.setdiff1d(impr, center))
                order = [not_center[0], not_center[1], center, not_center[2]]  # the third is the center
                atoms = uni_atom_types[indices[order]]
                params = ff.get_improper_params(*atoms)

            assert params is not None, f'Could not get improper params for {i}: {impr}'

            impropers['idx'].append(i)
            impropers['params'].append(params)

        impropers['idx'] = torch.tensor(impropers['idx'])
        impropers['params'] = torch.tensor(impropers['params'])

        return [impropers]

    @staticmethod
    def get_improper_graph(impropers, bonds):
        """Build a graph with nodes representing atoms in impropers and edges representing bonds."""
        g = nx.Graph()
        g.add_nodes_from(np.unique(impropers))
        g.add_edges_from([tuple(b) for b in bonds])

        return g

    @staticmethod
    def detect_improper_center(impr, graph):
        """Find the center atom in the improper."""
        for i in impr:
            if len(np.intersect1d(list(graph.neighbors(i)), impr)) == 3:
                return i

        raise RuntimeError(f'Could not find center atom for improper {impr}')

    def _get_lj_params(self, ff, uni_atom_types):
        """
        See self._compute_lj_AB for the shape of the return values.
        """
        pa = np.array([list(ff.get_lj_params(atom)) for atom in uni_atom_types], dtype=np.float64)
        sigma, epsilon = pa[:, 0].flatten(), pa[:, 1].flatten()
        A, B = self._compute_lj_AB(sigma, epsilon)

        return torch.tensor(A), torch.tensor(B)

    def _compute_lj_AB(self, sigma, epsilon):
        """
        Compute A and B in Lennard-Jones potentials with Lorentz-Berthelot combination rule.

        References:
            1. http://jerkwin.github.io/GMX/GMXman-4/
        """
        # sigma_table: (num_uni_atoms, num_uni_atoms), st_{ij} is (s_i + s_j) / 2
        # eps_table: (num_uni_atoms, num_uni_atoms), et_{ij} is sqrt(s_i * s_j)
        sigma_table = 0.5 * (sigma + sigma[:, None])
        eps_table = np.sqrt(epsilon * epsilon[:, None])

        sigma_table_6 = sigma_table**6
        sigma_table_12 = sigma_table_6 * sigma_table_6
        A = 4 * eps_table * sigma_table_12
        B = 4 * eps_table * sigma_table_6

        del sigma_table_12, sigma_table_6, eps_table, sigma_table
        return A, B

    @staticmethod
    def get_electrostatics_factor():
        r"""
        Return electrostatics factor: `k_e = \frac{1}{4\pi\epsilon_0}`.

        V_{electrostatics} = k_e \frac{q_i q_j}{r}

        Notice:
            1. Unit of q: electron charge unit
            2. Unit of r: Angstrom
            3. Unit of V_{electrostatics}: kcal/mol

        Since the units above, k_e is transformed to trans_k_e so that V_{electrostatics} is in kcal/mol
        when trans_k_e is multiplied by \frac{q_i q_j}{r} directly in their own units (shown in the
        notice above). The transformation is shown in the following code.

        References:
            1. https://arxiv.org/abs/2012.12106
        """
        from scipy import constants

        # constants.epsilon_0 is in F/m, so factor is in m/F, i.e. Jm/C^2
        factor = 1 / (4 * constants.pi * constants.epsilon_0)

        # constants.elementary_charge is about 1.6e-19, so that q can be multiplied directly
        factor *= constants.elementary_charge**2

        # constants.angstrom is 1e-10, so that 1/r can be multiplied directly
        factor /= constants.angstrom

        # convert J to kcal/mol
        factor *= constants.Avogadro / (constants.kilo * constants.calorie)

        return factor

    def _get_periodic_box(self, mol):
        """Get periodic box (in Angstroms)."""
        assert mol.box.shape == (3, 1), 'Shape of mol.box is not (3, 1).'

        if np.all(mol.box == 0):
            return None

        return torch.tensor(mol.box.flatten())

    def set_periodic_box_manual(self, box):
        """Set periodic box manually (in Angstroms).

        Parameters
        ----------
        box: numpy.ndarray
            A numpy array representing the box with shape (3, 1).
        """
        assert box.shape == (3, 1), 'Shape of box is not (3, 1).'

        if np.all(box == 0):
            self.box = None

        self.box = torch.tensor(box.flatten())

    def set_device_and_dtype(self, device, dtype):
        self._set_dtype(dtype)
        self._set_device(device)

    def _set_dtype(self, dtype):
        self.dtype = dtype

        self.charges = self.charges.type(dtype)
        self.masses = self.masses.type(dtype)

        if self.bonds is not None:
            self.bond_params = self.bond_params.type(dtype)

        if self.angles is not None:
            self.angle_params = self.angle_params.type(dtype)

        if self.dihedrals is not None:
            for j in range(len(self.dihedral_params)):
                p_term = self.dihedral_params[j]
                p_term['params'] = p_term['params'].type(dtype)

        if self.impropers is not None:
            p_term = self.improper_params[0]
            p_term['params'] = p_term['params'].type(dtype)

        if self.A is not None:
            self.A = self.A.type(dtype)
            self.B = self.B.type(dtype)

        if self.box is not None:
            self.box = self.box.type(dtype)

    def _set_device(self, device):
        self.device = device

        if self.mapped_atom_types is not None:
            self.mapped_atom_types = self.mapped_atom_types.to(device)

        self.charges = self.charges.to(device)
        self.masses = self.masses.to(device)

        if self.bonds is not None:
            self.bonds = self.bonds.to(device)
            self.bond_params = self.bond_params.to(device)

        if self.angles is not None:
            self.angles = self.angles.to(device)
            self.angle_params = self.angle_params.to(device)

        if self.dihedrals is not None:
            self.dihedrals = self.dihedrals.to(device)

            for j in range(len(self.dihedral_params)):
                p_term = self.dihedral_params[j]
                p_term['idx'] = p_term['idx'].to(device)
                p_term['params'] = p_term['params'].to(device)

        if self.impropers is not None:
            self.impropers = self.impropers.to(device)
            p_term = self.improper_params[0]
            p_term['idx'] = p_term['idx'].to(device)
            p_term['params'] = p_term['params'].to(device)

        if self.A is not None:
            self.A = self.A.to(device)
            self.B = self.B.to(device)

        if self.box is not None:
            self.box = self.box.to(device)

        self.non_grouped = self.non_grouped.to(device)
        if self.num_groups:
            for i in range(self.num_groups):
                self.mol_groups[i] = self.mol_groups[i].to(device)

    def get_exclusions(self, types=['bonds', 'angles']):
        """Get a list of exclusive atom pairs with type in types.

        Exclusive atom pairs are not computed for nonbonded forces, currently only supporting bonds and angles.
        """
        assert set(types) <= set(['bonds', 'angles']), f'{set(types)} is not the subset of {set(["bonds", "angles"])}'
        exclusions = []

        if self.bonds is not None and 'bonds' in types:
            exclusions += self.bonds.cpu().numpy().tolist()

        if self.angles is not None and 'angles' in types:
            np_angles = self.angles.cpu().numpy()
            exclusions += np_angles[:, [0, 2]].tolist()

        return exclusions

    def _build_atom_groups(self, num_atoms, bonds):
        if bonds is not None and len(bonds):
            g = nx.Graph()
            g.add_nodes_from(range(num_atoms))
            g.add_edges_from(bonds.astype(np.int64))

            mol_groups = list(nx.connected_components(g))
            non_grouped = torch.tensor([list(group)[0] for group in mol_groups if len(group) == 1])
            mol_groups = [torch.tensor(list(group)) for group in mol_groups if len(group) > 1]
        else:
            mol_groups = []
            non_grouped = torch.arange(num_atoms)

        return mol_groups, non_grouped
