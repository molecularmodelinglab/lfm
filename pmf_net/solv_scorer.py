from copy import deepcopy
import os
import numpy as np
from tqdm import tqdm, trange
from common.bingham import BinghamDistribution
from common.pose_transform import Pose
from pmf_net.pose_pred import mol_to_pmf_input
import torch
from openmm import app, unit
import openmm as mm
import mdtraj as md
from scipy.spatial.transform import Rotation
from scipy import stats

from common.openmm_utils import make_system
from common.utils import (
    add_coords_to_mol,
    align_mols_noh,
    get_output_dir,
    rdkit_to_modeller,
    save_modeller_pdb,
)
from pmf_net.scorer import PMFScore


class PMFSolvScore:
    """A better approximation of binding affinity by subtracting off the mean solv PMF"
    energy from an OBC2 trajectory of the ligand"""

    def __init__(
        self, model, device, n_steps=1000000, n_burnin=100, report_interval=1000
    ):
        self.scorer = PMFScore(model, device)
        self.device = device
        self.dt = 4.0 * unit.femtoseconds
        self.report_interval = report_interval
        self.steps = n_steps
        self.n_burnin = n_burnin

        Z = np.array([-100, -100, -100, 0])
        M = np.identity(4)
        self.R_dist = BinghamDistribution(M, Z)

        mean = np.array([0, 0, 0])
        cov = np.identity(3) * 0.001
        self.t_dist = stats.multivariate_normal(mean=mean, cov=cov)

    def __call__(self, mol, system=None, top=None):

        # first minimize to get the correct pose
        pmf_score, mol, _ = self.scorer(mol, system=system, top=top)

        # now run an OBC2 simuation

        struct = rdkit_to_modeller(mol)
        system, modeller = make_system(struct, "obc2", [mol])
        for force in system.getForces():
            if (
                isinstance(force, mm.PeriodicTorsionForce)
                or isinstance(force, mm.NonbondedForce)
                or isinstance(force, mm.CustomGBForce)
            ):
                force.setForceGroup(1)

        platform = mm.Platform.getPlatformByName("CUDA")
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picoseconds, self.dt
        )
        sim = app.Simulation(struct.topology, system, integrator, platform)
        context = sim.context

        out_fname = f"{get_output_dir()}/solv.dcd"
        out_top_fname = f"{get_output_dir()}/solv.pdb"
        save_modeller_pdb(struct, out_top_fname)

        if os.path.exists(out_fname):
            os.remove(out_fname)

        dcd_reporter = app.DCDReporter(out_fname, self.report_interval)
        sim.reporters.append(dcd_reporter)

        sim.context.setPositions(struct.positions)
        sim.minimizeEnergy()
        for i in trange(self.steps):
            sim.step(1)

        data = mol_to_pmf_input(mol, system=system, top=struct.topology).to(self.device)
        traj = md.load(out_fname, top=out_top_fname)

        mol_aligned = deepcopy(mol)
        with torch.no_grad():
            solv_Us = []
            lr_Us = []
            obc2_Us = []
            for frame in tqdm(traj.xyz[self.n_burnin :]):

                # first move mol to exit point
                pos_torch = (
                    torch.tensor(frame, dtype=torch.float32).to(self.device) * 10.0
                )
                # center pos on exit point
                pos_torch -= pos_torch.mean(0)
                pos_torch += self.scorer.exit_point
                pose = Pose(pos_torch)

                solv_Us.append(self.scorer.pp.total_U(data, pose).item())

                context.setPositions(frame * unit.nanometers)
                # only get torsion, nonbonded, and GB forces
                state = context.getState(getEnergy=True, groups=1 << 1)
                obc2_Us.append(
                    state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                )

                # now align to docked pose (plus add noise)
                add_coords_to_mol(mol_aligned, frame * 10)
                mol_aligned = align_mols_noh(mol_aligned, mol)
                coords = mol_aligned.GetConformer().GetPositions()

                # add noise
                q = self.R_dist.random_samples(1)[0]
                R = Rotation.from_quat(q)
                t = self.t_dist.rvs()
                coords = R.apply(coords) + t

                pose = Pose(torch.tensor(coords, dtype=torch.float32).to(self.device))
                lr_Us.append(self.scorer.pp.total_U(data, pose).item())

            solv_Us = np.array(solv_Us)
            lr_Us = np.array(lr_Us)
            obc2_Us = np.array(obc2_Us)

            docked_pos = torch.tensor(
                mol.GetConformer().GetPositions(), dtype=torch.float32
            ).to(self.device)
            docked_pose = Pose(docked_pos)
            docked_U = self.scorer.pp.total_U(data, docked_pose).item()

            # maybe remove?
            # the idea here is to account for the possibility that
            # some of the solv positions sampled are clashing with the
            # receptor -- this would cause a major underestimation of dG
            # todo -- account for the fact that the bond/angle terms aren't being cancelled out exactly
            solv_Us = solv_Us.min() + obc2_Us - obc2_Us.min()

            dU = lr_Us - solv_Us
            # EXP for free energy
            dG = -np.log(np.exp(-dU).mean())
            # now enthalpy
            dH = docked_U - np.mean(solv_Us)

        return dG, mol, {"pmf_score": pmf_score, "dH": dH}
