import numpy as np
from math import radians


class ForceField(object):
    """
    Base class for forcefield.

    Used for future extension.
    """

    def __init__(self):
        pass


class PrmtopMolForceField(ForceField):
    """Forcefield built from PRMTOP file and moleculekit.molecule object."""

    def __init__(self, mol, prmtop, allow_unequal_duplicates=None):
        super().__init__()
        self.mol = mol
        self.params = prmtop
        if isinstance(prmtop, str):
            assert isinstance(allow_unequal_duplicates,
                              bool), 'Allow_unequal_duplicates should be bool when prmtop is a path.'
            self.params = self.read_prmtop(prmtop, allow_unequal_duplicates)

    @staticmethod
    def read_prmtop(prmtop_file, allow_unequal_duplicates):
        """Read PRMTOP file."""
        import parmed

        struct = parmed.amber.AmberParm(prmtop_file)
        if not allow_unequal_duplicates:
            return parmed.amber.AmberParameterSet.from_structure(struct)

        class MyAmberParameterSet(parmed.amber.AmberParameterSet):
            """Inherited from AmberParameterSet to modify from_structure method."""

            @classmethod
            def my_from_structure(cls, struct):
                return super(parmed.amber.AmberParameterSet, cls).from_structure(struct, allow_unequal_duplicates=True)

        return MyAmberParameterSet.my_from_structure(struct)

    def get_bond_params(self, atom1, atom2):
        """
        Return bond parameters.

        atom1 and atom2 should be str such as H1
        """
        p = self.params.bond_types[(atom1, atom2)]
        return p.k, p.req

    def get_angle_params(self, atom1, atom2, atom3):
        """
        Return angle parameters (convert degrees to radians).

        atom123 should be str such as H1
        """
        p = self.params.angle_types[(atom1, atom2, atom3)]
        return p.k, radians(p.theteq)

    def get_dihedral_params(self, atom1, atom2, atom3, atom4):
        """
        Return dihedral parameters (maybe multiple).
        """
        p = None
        variants = [(atom1, atom2, atom3, atom4), (atom4, atom3, atom2, atom1)]

        for var in variants:
            if var in self.params.dihedral_types:
                p = self.params.dihedral_types[var]
                break

        if p is None:
            raise RuntimeError(f'Could not find dihedral parameters for ({atom1}, {atom2}, {atom3}, {atom4}).')

        ret = []

        for pp in p:
            ret.append([pp.phi_k, radians(pp.phase), pp.per])

        return ret

    def get_improper_params(self, atom1, atom2, atom3, atom4):
        """
        Return improper parameters.

        atom3 is the center atom
        """
        from itertools import permutations

        types = np.array((atom1, atom2, atom3, atom4))
        perms = np.array([x for x in list(permutations((0, 1, 2, 3))) if x[2] == 2])

        for p in perms:
            if tuple(types[p]) in self.params.improper_types:
                pa = self.params.improper_types[tuple(types[p])]
                return pa.psi_k, radians(pa.psi_eq), 0
            elif tuple(types[p]) in self.params.improper_periodic_types:
                pa = self.params.improper_periodic_types[tuple(types[p])]
                return pa.phi_k, radians(pa.phase), pa.per

        raise RuntimeError(f'Could not find improper parameters for key {types}.')

    def get_lj_params(self, atom):
        p = self.params.atom_types[atom]
        return p.sigma, p.epsilon
