import parmed
import openmm.app as app


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
