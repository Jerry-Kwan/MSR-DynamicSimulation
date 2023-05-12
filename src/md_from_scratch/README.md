# MD from Scratch

An MD Simulation from scratch using Python.

## References

1. [OpenMM](https://openmm.org/), [The Theory Behind OpenMM](http://docs.openmm.org/latest/userguide/theory.html)
2. [TorchMD](https://github.com/torchmd/torchmd)
3. [An MD Tutorial](https://klyshko.github.io/teaching/2019-03-01-teaching)
4. [LAMMPS Units](https://docs.lammps.org/99/units.html)
5. [AMBER File Formats](https://ambermd.org/FileFormats.php)
6. ...

## TODO List

* Logically, charges and masses in System class should be (maybe) obtained from a ForceField object, not a Molecule object.

* Implement the computation of forces according to $\frac{\partial E_{pot}}{\partial \vec {r_i}}$ using PyTorch autograd.

* Implement external forces.

* Implement DCDReporter class for recording the trajectory in simulation (currently it is implemented directly in step() method in Simulation class, which is not good).
* Implement multiple reporters.

* Add langevin in Integrator and [run Equilibration before NVE Simulation](https://github.com/noegroup/torchmd-autodiff/blob/main/simulate.ipynb).