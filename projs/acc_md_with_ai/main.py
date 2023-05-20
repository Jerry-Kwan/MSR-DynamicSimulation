import os
import random
import numpy as np
import torch
from moleculekit.molecule import Molecule
import parmed
import nglview as ng
import MDAnalysis as md

import openmm.app as app
from openmm import Platform, LangevinMiddleIntegrator, XmlSerializer
from openmm.unit import picosecond, kelvin
from openmm import unit
import openmm

import mymd

seed = 142
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# path
data_path = 'data/aspirin/'
model_path = 'data/aspirin/et-md17/'
csv_path = 'data/aspirin/rslt_mymd_net/'
dcd_path = 'data/aspirin/rslt_mymd_net/'

# file
pdb_file = os.path.join(data_path, 'aspirin.pdb')
psf_file = os.path.join(data_path, 'aspirin.psf')
prmtop_file = os.path.join(data_path, 'aspirin.prmtop')
model_file = os.path.join(model_path, 'epoch=2139-val_loss=0.2543-test_loss=0.2317.ckpt')

# hyperparameters
cutoff = None
cutoff_lower = 1e-5  # needed
T = 300
dt_fs = 2
dcd_interval = 100
csv_interval = 50
steps = 50000
min_energy_max_iter = 100
box_size = 100
device = 'cpu'
precision = torch.float
use_centered = True

use_external = True
if use_external:
    from torchmdnet.models.model import load_model

    model = load_model(model_file, derivative=True, cutoff_lower=cutoff_lower)
    sim_terms = []
else:
    model = None
    sim_terms = ['bonds', 'angles', 'dihedrals', 'impropers', 'lj', 'electrostatics']

# build Molecule object
mol = mymd.get_molecule(prmtop_file=prmtop_file, pdb_file=pdb_file)

# build forcefield
try:
    ff = mymd.PrmtopMolForceField(mol, prmtop_file, allow_unequal_duplicates=False)
except:
    print('False causes error, use True.')
    ff = mymd.PrmtopMolForceField(mol, prmtop_file, allow_unequal_duplicates=True)

# build system
system = mymd.System(mol, ff, cutoff=cutoff, external=model)
system.set_periodic_box_manual(np.array([box_size, box_size, box_size]).reshape(3, 1))

# set integrator
integrator = mymd.VelocityVerletIntegrator(dt_fs)

# build simulation object
simulation = mymd.Simulation(
    mol,
    system,
    integrator,
    device,
    precision,
    use_centered=use_centered,
    use_external=use_external,
    sim_terms=sim_terms
)  # yapf: disable
simulation.set_positions(mol.coords)
simulation.set_velocities_to_temperature(T=T)
simulation.update_potentials_and_forces()
if use_external:
    simulation.set_external_model_eval()

# add reporter
csv_reporter = mymd.CSVReporter(csv_path, csv_interval)
simulation.add_reporter(csv_reporter)

print(simulation.potentials)
print(simulation.potentials_sum)

# minimize_energy
simulation.minimize_energy(min_energy_max_iter)
print(simulation.potentials)
print(simulation.potentials_sum)

# start simulation
simulation.step(steps, dcd_path, dcd_interval)
