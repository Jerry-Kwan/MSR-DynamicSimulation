from moleculekit.molecule import Molecule


def get_molecule(psf_file=None, prmtop_file=None, xtc_file=None, coor_file=None, pdb_file=None, xsc_file=None):
    """Construct Molecule Object from different files."""
    # psf or prmtop
    if psf_file is not None:
        mol = Molecule(psf_file)
    elif prmtop_file is not None:
        mol = Molecule(prmtop_file)
    else:
        raise RuntimeError('Either PSF file or PRMTOP file should not be None.')

    # xtc, coor or pdb, used to set the mol's coordinate
    if xtc_file is not None:
        mol.read(xtc_file)
    elif coor_file is not None:
        mol.read(coor_file)
    elif pdb_file is not None:
        mol.read(pdb_file)
    else:
        raise RuntimeError('No XTC, COOR or PDB given.')

    # xsc file, used to set the box, can be None
    if xsc_file is not None:
        mol.read(xsc_file)

    return mol
