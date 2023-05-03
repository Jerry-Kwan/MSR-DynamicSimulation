import parmed
import openmm.app as app
from moleculekit.molecule import Molecule


def gen_files_from_pdb(old_pdb_file,
                       psf_file,
                       prmtop_file,
                       new_pdb_file,
                       forcefield_file='amber14-all.xml',
                       nonbondedMethod=app.NoCutoff,
                       constraints=None,
                       rigidWater=False):
    """
    Generate PSF file, PRMTOP file and a new PDB file from the original PDB file.
    """
    # load pdb file
    pdb = app.PDBFile(old_pdb_file)
    topology = pdb.topology
    positions = pdb.positions

    # construct forcefield
    forcefield = app.ForceField(forcefield_file)

    # construct system
    system = forcefield.createSystem(topology,
                                     nonbondedMethod=nonbondedMethod,
                                     constraints=constraints,
                                     rigidWater=rigidWater)

    # use parmed to generate files
    parmed_structure = parmed.openmm.load_topology(topology, system, positions)
    parmed_structure.save(psf_file, overwrite=True)
    parmed_structure.save(prmtop_file, overwrite=True)
    parmed_structure.save(new_pdb_file, overwrite=True)


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
