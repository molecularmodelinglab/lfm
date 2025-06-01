import os

from datasets.pmf_dataset import PMFData
from pymbar import MBAR
import numpy as np
from tqdm import tqdm, trange
from common.bingham import BinghamDistribution
from openmm import app, unit
import openmm as mm
import torch
import torch.nn as nn
from openmmtorch import TorchForce
from copy import deepcopy
from torch_geometric.data import Data
import mdtraj as md
from common.restraint import kB, BinghamRestraint
from scipy.spatial.transform import Rotation
from scipy import stats

# tqdm = lambda x: x
# trange = lambda x: range(x)

class PMFWrapper(nn.Module):

    def __init__(self, model, pmf_data): #nb_params):
        super().__init__()
        # self.register_buffer("nb_params", nb_params)
        self.register_buffer(
            "batch",
            torch.zeros(len(pmf_data.nb_params), dtype=torch.long, device=pmf_data.nb_params.device),
        )
        self.pmf_data = pmf_data
        self.model = model

    def forward(self, positions):  # , pmf_lambda):
        """at pmf_lambda=0, the ligand is unbound. At pmf_lambda=1, the ligand is bound."""
        positions = positions.to(torch.float32).cuda() * 10.0  # nm -> A
        # unbound_trans = torch.tensor([0.0, 15.0, 15.0], device=positions.device)
        energy_bound = self.model(PMFData(*self.pmf_data), positions, self.batch)[0][0, 0]
        # energy_unbound = self.model(self.nb_params, positions + unbound_trans[None], self.batch)[0][0,0]

        return energy_bound  # *pmf_lambda + energy_unbound*(1-pmf_lambda)


class ABFE_Sim:
    """Runs the simulation for ABFE"""

    def __init__(
        self,
        model: nn.Module,
        system: mm.System,
        batch: Data,
        out_folder: str,
        device: torch.device,
    ):
        self.solv_system = system
        self.bound_system = deepcopy(system)
        self.topology = self.make_topology(batch)
        self.init_pos = batch.pos.cpu().numpy() / 10  # Angstrom -> nm
        self.model = model
        self.device = device

        self.kT = (kB * 300 * unit.kelvin).in_units_of(unit.kilojoules_per_mole)

        pmf_data = PMFData.from_pyg_data(batch.to(device))
        wrapper = PMFWrapper(self.model, pmf_data)
        self.wrapper_model = torch.jit.script(wrapper).to(device)

        self.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)

        self.pdb_fname = os.path.join(out_folder, "init.pdb")
        # save pdb file
        app.PDBFile.writeFile(
            self.topology, self.init_pos * unit.nanometer, open(self.pdb_fname, "w")
        )

        self._remove_solvent()
        self._add_pmf_force(batch)

    def _remove_solvent(self):
        """Removes OBC2 implicit solvent from the system"""
        to_remove = None

        for i, force in enumerate(self.bound_system.getForces()):
            if isinstance(force, mm.CustomGBForce):
                to_remove = i
                break

        if to_remove is not None:
            self.bound_system.removeForce(to_remove)

    def _add_pmf_force(self, batch: Data):
        """Adds the PMF force to the system with force group 1"""
        force = TorchForce(self.wrapper_model)
        force.setForceGroup(1)
        # force.addGlobalParameter("pmf_lambda", 1.0)
        self.bound_system.addForce(force)

    def make_topology(self, batch):
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("LIG", chain)
        for num in batch.elements:
            element = app.Element.getByAtomicNumber(num.item())
            topology.addAtom(f"{element.symbol}", element, residue)
        return topology

    def simulate_bound(
        self,
        total_time=20 * unit.picoseconds,
        dt=100 * unit.femtoseconds,
        report_interval=1.0 * unit.picoseconds,
    ):
        """Simulate the bound system. Returns the trajectory"""

        total_steps = int(total_time / dt)
        report_steps = int(report_interval / dt)

        report_fname = os.path.join(self.out_folder, "bound.dcd")

        tot_len = total_steps // report_steps

        append = False
        remaining_steps = total_steps

        if os.path.exists(report_fname):
            # os.remove(report_fname)
            try:
                traj = md.load(report_fname, top=self.pdb_fname)
                if len(traj) >= tot_len:
                    return traj.xyz[:tot_len]
                else:
                    remaining_steps = total_steps - (len(traj) * report_steps)
                    append = True
            except OSError:
                pass

        integrator = mm.MTSLangevinIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            dt,
            [(0, 25), (1, 1)],
        )
        platform = mm.Platform.getPlatformByName("CUDA")

        simulation = app.Simulation(
            self.topology, self.bound_system, integrator, platform
        )

        simulation.context.setPositions(self.init_pos if not append else traj.xyz[-1])
        simulation.reporters.append(app.DCDReporter(report_fname, report_steps, append=append))

        if not append:
            simulation.minimizeEnergy()

        for i in trange(remaining_steps):
            simulation.step(1)

        return md.load(report_fname, top=self.pdb_fname).xyz

    def simulate_solv(
        self,
        total_time=2.5 * unit.nanoseconds,
        dt=2 * unit.femtoseconds,
        report_interval=2.0 * unit.picoseconds,
    ):
        """Simulate the unbound system. Returns the trajectory"""

        total_steps = int(total_time / dt)
        report_steps = int(report_interval / dt)

        report_fname = os.path.join(self.out_folder, "solv.dcd")
        # try to return the trajectory if it already exists

        tot_len = total_steps // report_steps

        append = False
        remaining_steps = total_steps

        if os.path.exists(report_fname):
            # os.remove(report_fname)
            try:
                traj = md.load(report_fname, top=self.pdb_fname)
                if len(traj) >= tot_len:
                    return traj.xyz[:tot_len]
                else:
                    remaining_steps = total_steps - (len(traj) * report_steps)
                    append = True
            except OSError:
                pass

        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            dt,
        )
        platform = mm.Platform.getPlatformByName("CUDA")

        simulation = app.Simulation(
            self.topology, self.solv_system, integrator, platform
        )

        simulation.context.setPositions(self.init_pos if not append else traj.xyz[-1])
        simulation.reporters.append(app.DCDReporter(report_fname, report_steps, append=append))

        if not append:
            simulation.minimizeEnergy()

        for i in trange(remaining_steps):
            simulation.step(1)

        return md.load(report_fname, top=self.pdb_fname).xyz


class ABFE:
    """Analyzes the ABFE simulations"""

    def __init__(
        self,
        abfe_sim,
        bound_traj,
        solv_traj,
        bound_burnin=10,
        bound_subsample=2,
        solv_burnin=100,
        solv_subsample=2,
        n_solv_transformations=1,
    ):

        self.init_pos = abfe_sim.init_pos
        self.device = abfe_sim.device
        self.kT = abfe_sim.kT
        self.wrapper_model = abfe_sim.wrapper_model
        self.topology = abfe_sim.topology
        self.out_folder = abfe_sim.out_folder

        self.n_solv_transformations = n_solv_transformations

        if bound_traj is None:
            self.bound_traj = np.zeros((0, *solv_traj.shape[1:]))
        else:
            self.bound_traj = bound_traj[bound_burnin::bound_subsample]
        
        solv_traj = solv_traj[solv_burnin::solv_subsample]

        Z = np.array([-100, -100, -100, 0])
        M = np.identity(4)
        R_dist = BinghamDistribution(M, Z)

        mean = np.array([0, 0, 0])
        cov = np.identity(3)*0.001
        t_dist = stats.multivariate_normal(mean=mean, cov=cov)

        self.restraint = BinghamRestraint(
            self.init_pos * unit.nanometers, self.kT, t_dist, R_dist, None
        )

        # self.restraint = BinghamRestraint.from_frames(
        #     self.init_pos * unit.nanometers, self.kT, self.bound_traj, 0.0
        # )
        self.solv_traj = self.restraint.align_lig_frames(solv_traj)

    def bound_U(self, pos):
        with torch.no_grad():
            return self.wrapper_model(pos.to(self.device)).item()

    def bound_U_traj(self, traj):
        """Compute the bound energy of a trajectory"""
        return torch.tensor(
            [
                self.bound_U(torch.tensor(frame, device=self.device))
                for frame in traj
            ]
        )

    def solv_U(self, pos):
        solv_vec = torch.tensor([0.0, 1.5, 1.5]).to(self.device)
        with torch.no_grad():
            return self.wrapper_model(pos.to(self.device) + solv_vec).item()

    def solv_U_traj(self, traj):
        """Compute the unbound energy of a trajectory"""
        return torch.tensor(
            [
                self.solv_U(torch.tensor(frame, device=self.device))
                for frame in tqdm(traj)
            ]
        )

    def U0(self, pos):
        """Compute the solv + restraint energy of a frame"""
        return self.solv_U(
            torch.tensor(pos)
        ) * unit.kilojoules_per_mole + self.restraint.U(pos)

    def U1(self, pos):
        """Compute the bound energy of a frame"""
        return self.bound_U(torch.tensor(pos)) * unit.kilojoules_per_mole

    def get_combined_energies(self, frames):
        """Returns the (reduced) potential energies U0 and U1 for each
        frame in the trajectory as a numpy array"""
        U0s = np.array([self.U0(frame) / self.kT for frame in tqdm(frames)])
        U1s = np.array([self.U1(frame) / self.kT for frame in tqdm(frames)])
        return np.vstack([U0s, U1s])

    def get_all_u_kn(self, *trajectories):  # all the trajectories we want to analyze
        """Returns all that we need for MBAR to analyze the trajectories -- the
        combined energies u_kn array and the N_k array describing the lengths of
        each trajectory."""
        all_frames = np.concatenate(trajectories, axis=0)
        u_kn = self.get_combined_energies(all_frames)
        N_k = np.array([len(frames) for frames in trajectories])
        return u_kn, N_k
    
    def get_sampled_solv_traj(self):
        """Returns the solvent trajectory after sampling with restraint transformations"""

        assert self.n_solv_transformations == 1
        Ts = self.restraint.sample_transformation_batch(len(self.solv_traj))
        solv_traj = np.stack([T(frame) for T, frame in zip(Ts, self.solv_traj)])
        return solv_traj
    
    def run(self):
        """ Return predicted delta G for binding """

        solv_traj = self.get_sampled_solv_traj()
        # save to a dcd file
        file = app.DCDFile(
            open(os.path.join(self.out_folder, "solv_sampled.dcd"), "wb"),
            self.topology,
            2 * unit.picoseconds,
        )

        for frame in solv_traj:
            file.writeModel(frame)

        bound_traj = self.bound_traj

        u_kn, N_k = self.get_all_u_kn(solv_traj, bound_traj)

        mbar = MBAR(u_kn, N_k)
        full_estimate = mbar.compute_free_energy_differences()["Delta_f"][0, 1]
        full_estimate = full_estimate * self.kT

        self.u_kn = u_kn
        self.N_k = N_k
        self.full_estimate = full_estimate
        self.restraint_dF = self.restraint.get_standard_state_dF()

        dG = full_estimate + self.restraint.get_standard_state_dF()
        return dG.value_in_unit(unit.kilocalories_per_mole)