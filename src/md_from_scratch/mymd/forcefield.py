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
