Storing the results used in report.

`traj-nocutoff.dcd` is the result of OpenMM, should be used with `md_example/ala15_openmm/ala15,ipynb` to get the Ramachandran plots and distance between two atoms during simulation.

`traj.dcd` is the result of mymd, should be used with `main_ala15_mymd.ipynb` to get the Ramachandran plots and distance between two atoms during simulation. Hyperparameters:

```
seed = 108
cutoff = None
T = 300.15
dt_fs = 2
dcd_interval = 100
csv_interval = 50
# steps = 50000
steps = 50000
min_energy_max_iter = 100
box_size = 100
device = 'cuda:1'
precision = torch.float
use_centered = True
```

mymd and OpenMM use the same psf file `../ala15.psf`