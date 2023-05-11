import os
from tqdm import tqdm
import numpy as np
import torch
from mdtraj.formats import DCDTrajectoryFile

from .sim_constants import BOLTZMANN, FS_2_NS
from .forces import Forces
from .reporter import Reporter


class Simulation(object):
    """This class controls the whole simulation procedure.

    Imitating openmm.app.simulation.Simulation
    """

    BONDED_TERMS = ['bonds', 'angles', 'dihedrals', 'impropers']
    NONBONDED_TERMS = ['lj', 'electrostatics']
    TERMS = BONDED_TERMS + NONBONDED_TERMS

    def __init__(self,
                 mol,
                 system,
                 integrator,
                 device,
                 dtype,
                 exclusions=['bonds', 'angles'],
                 use_external=False,
                 sim_terms=TERMS):
        """Create a simulation object.

        Parameters
        ----------
        exclusion: list=['bonds', 'angles']
            A list containing the exclusive force terms. If the force type of an atom pair
            is in the exclusion list, then this pair is not computed for nunbonded forces.
        sim_terms: list=TERMS
            A list containing the force terms computed in simulation.
        """
        self.mol = mol
        self.system = system
        self.integrator = integrator
        self.device = device
        self.dtype = dtype
        self.use_external = use_external

        self.reporter = []

        assert set(sim_terms) <= set(self.TERMS), 'Some of terms are not implemented.'
        self.sim_terms = sim_terms

        assert set(exclusions) <= set(['bonds', 'angles']), (f'Exclusions should be the subset '
                                                             f'of {set(["bonds", "angles"])}')
        self.exclusions = exclusions

        self._build_simulation()

    def _build_simulation(self):
        self.system.set_device_and_dtype(self.device, self.dtype)

        n = self.system.num_atoms
        self.pos = torch.zeros(n, 3).type(self.dtype).to(self.device)
        self.vel = torch.zeros(n, 3).type(self.dtype).to(self.device)

        # forces is a tensor with shape (n, 3) in self.dtype and self.device
        # potentials is a dict storing the value of potentials in Python scalars
        # see Forces.compute_potentials_and_forces for more details about these two variables
        self.forces = None
        self.potentials = None
        self.potentials_sum = None

        self._f = Forces(self.system,
                         self.device,
                         self.dtype,
                         terms=self.sim_terms,
                         exclusions=self.exclusions,
                         use_external=self.use_external)

    def set_positions(self, pos):
        assert pos.shape == (self.system.num_atoms, 3, 1), f'Shape of pos is not {(self.system.num_atoms, 3, 1)}'

        pos = np.squeeze(pos, 2)
        self.pos[:] = torch.tensor(pos, dtype=self.dtype, device=self.device)

    def set_velocities_to_temperature(self, T):
        """
        Set the velocities of all particles in the System to random values chosen from a
        Maxwell-Boltzmann distribution at a given temperature.
        """
        std_normal_dist = torch.randn((self.system.num_atoms, 3)).type(self.dtype)
        mb_dist = torch.sqrt(T * BOLTZMANN / self.system.masses) * std_normal_dist

        self.vel[:] = mb_dist.type(self.dtype).to(self.device)

    def update_potentials_and_forces(self):
        self.potentials, self.forces = self._f.compute_potentials_and_forces(self.pos)
        self.potentials_sum = np.sum([v for _, v in self.potentials.items()])

    def minimize_energy(self, max_iter, disp_scipy=False):
        """Search for a new set of particle positions that represent a local potential
        energy minimum (use L-BFGS-B in scipy). On exit, the positions, potentials and forces
        will have been updated (use L-BFGS-B in scipy).

        Currently not supporting tolerance parameter like minimizeEnergy() method in openmm. If
        you want to precisely halt the minimization once the root-mean-square value of all force
        components reaches the tolerance, you need to manually determine the max_iter.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations to perform. In scipy, depending on the
            method each iteration may use several function evaluations. Here,
            we use L-BFGS-B.
        disp_scipy : bool
            Whether display scipy info or self-made info.
        """
        from scipy.optimize import minimize

        def fun_eval(coords, info):
            coords = coords.reshape(-1, 3)
            coords = torch.tensor(coords).type(self.dtype).to(self.device)
            pot, f = self._f.compute_potentials_and_forces(coords)

            pot = np.sum([v for _, v in pot.items()])
            grad = -f.detach().cpu().numpy().astype(np.float64)
            norm_f_max = np.max(np.linalg.norm(grad, axis=1))
            rms = np.sqrt(np.mean(grad**2))

            # this is maybe max |proj g_i | in scipy L-BFGS-B minimize function
            max_abs_f_1d = np.max(np.abs(grad))

            if not disp_scipy:
                ss = '{0:^12d}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}'
                print(ss.format(info['num_fun_eval'], pot, norm_f_max, rms, max_abs_f_1d))

            info['num_fun_eval'] += 1
            return pot, grad.reshape(-1)

        if not disp_scipy:
            ss = '{0:12s}\t{1:9s}\t{2:9s}\t{3:9s}\t{4:12s}'
            print(ss.format('num_fun_eval', 'E_pot', 'norm_F_max', 'rms', 'max_abs_f_1d'))

        x_0 = self.pos.detach().cpu().numpy().astype(np.float64).reshape(-1)
        options = {'maxiter': max_iter, 'disp': disp_scipy}
        args = ({'num_fun_eval': 0}, )

        # jac=True means use the gradient returned by fun_eval()
        ret = minimize(fun_eval, x_0, method='L-BFGS-B', jac=True, options=options, args=args)

        self.pos = torch.tensor(
            ret.x.reshape(-1, 3),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.pos.requires_grad
        )  # yapf: disable

        self.update_potentials_and_forces()

    def add_reporter(self, reporter):
        assert isinstance(reporter, Reporter), 'Parameter "reporter" is not an instance of Reporter class.'
        self.reporter.append(reporter)

    def step(self, steps, dcd_path, dcd_interval, dcd_name='traj.dcd'):
        """Advance the simulation by integrating a specified number of time steps.

        Currently, reporter only support CSVReporter and only support one reporter.

        Notice, for example, if steps=100 and dcd_interval=10, then totally 11 frames are
        in the trajectory.

        TOBEDONE: multiple reporters.
        """
        reporter = self.reporter[0] if len(self.reporter) else None

        os.makedirs(dcd_path, exist_ok=True)
        dcd_name = os.path.join(dcd_path, dcd_name)
        if os.path.exists(dcd_name):
            print('Remove the original dcd file.')
            os.remove(dcd_name)

        traj = []
        self._wrap()
        traj.append(self.pos.detach().cpu().numpy().copy())
        iter = tqdm(range(1, steps + 1))

        for i in iter:
            self.integrator.step(1, self.system, self.pos, self.vel, self._f)
            # TOBEDONE: test wrap and add temperature in integrator for better alpha-beta
            self._wrap()
            self.update_potentials_and_forces()

            if i % dcd_interval == 0:
                traj.append(self.pos.detach().cpu().numpy().copy())

            if reporter is not None and i % reporter.report_interval == 0:
                info = dict()
                info['step'] = i
                info['ns'] = i * self.integrator.dt_fs * FS_2_NS
                info['e_pot'] = self.potentials_sum

                ret = self.get_e_kin_and_temperature()
                info['e_kin'] = ret[0]
                info['e_tot'] = ret[0] + self.potentials_sum
                info['T'] = ret[1]
                reporter.write_row(info)

        # save trajectory
        # https://www.mdtraj.org/1.9.8.dev0/api/generated/mdtraj.formats.DCDTrajectoryFile.html
        # the conventional units in the DCD file are angstroms and degrees
        # search "dcd file unit" in bing or google for details
        with DCDTrajectoryFile(dcd_name, 'w') as f:
            for i in range(len(traj)):
                f.write(traj[i])

    def get_e_kin_and_temperature(self):
        """Return Kinetic Energy and corresponding temperature."""
        e_kin = torch.sum(0.5 * torch.sum(self.vel * self.vel, dim=1, keepdim=True) * self.system.masses, dim=0)
        e_kin = e_kin.item()
        temperature = 2.0 / (3.0 * self.system.num_atoms * BOLTZMANN) * e_kin
        return e_kin, temperature

    def _wrap(self):
        if self.system.box is None:
            return

        if self.system.num_groups:
            # work out the center and offset of every group and move group to [0, box] range
            for i, group in enumerate(self.system.mol_groups):
                center = torch.sum(self.pos[group], dim=0) / len(group)

                # find the nearest box endpoint to center (use floor to make sure offset is smaller than center)
                offset = torch.floor(center / self.system.box) * self.system.box
                self.pos[group] -= offset.unsqueeze(0)

        if self.system.num_non_grouped:
            # move non_grouped atoms
            box = self.system.box.unsqueeze(0)
            offset = torch.floor(self.pos[self.system.non_grouped] / box) * box
            self.pos[self.system.non_grouped] -= offset
