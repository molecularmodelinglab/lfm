from copy import deepcopy
from traceback import print_exc
import numpy as np
import openmm as mm
from common.utils import (
    CONFIG,
    get_angle_params,
    get_bond_params,
    get_constraints,
    get_nb_matrices,
    get_nonbonded_params,
    get_torsion_params,
)
from rdkit import Chem
from openff.toolkit.topology import Molecule
from openff.units.openmm import to_openmm
from openmmforcefields.generators import EspalomaTemplateGenerator
from openmm import app, unit
from common.torsion import TorsionData
import torch
from torch_geometric.data import Data
from common.pose_transform import (
    MultiPose,
    Pose,
    add_multi_pose_to_mol,
    add_pose_to_mol,
)
from pmf_net.pose_pred import PosePred, get_system_and_top_from_mol, mol_to_pmf_input


def add_pos_restraint(system, top, pos, k):
    """Adds harmonic restraints between all the atoms
    and their initial positions to the system. Returns
    the new system. K is in kJ/mol/nm^2.
    """

    if k is None:
        return system

    ret = deepcopy(system)

    restraint = mm.CustomExternalForce("k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    ret.addForce(restraint)
    restraint.addGlobalParameter("k", k)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for atom in top.atoms():
        restraint.addParticle(atom.index, pos[atom.index].value_in_unit(unit.nanometer))

    return ret


class PMFScore:
    """Approximates binding affinity by subtracting solvation energy from PMF in the binding site"""

    def __init__(
        self, model, device, minmize_mm=False, minimize_pmf=True, mm_restraint_k=None
    ):
        self.model = model
        self.minmize_mm = minmize_mm
        self.mm_restraint_k = mm_restraint_k
        self.minimize_pmf = minimize_pmf
        self.pp = PosePred(self.model)
        self.pp.com_k = None
        self.device = device
        self.exit_point = torch.tensor(
            CONFIG.dataset.pocket.exit_point, dtype=torch.float32
        ).to(device)

    def score_single_pose(self, data, td, init_pose, optimize=True, sim=None):
        """Returns the total score, the total U, and the optimized pose"""

        if sim is not None:
            # first minimize the initial pose
            # print("Minimizing initial pose...")
            pos = init_pose.coord.cpu().numpy() * unit.angstrom
            sim.context.setPositions(pos)
            sim.minimizeEnergy(maxIterations=1000)
            # print("Stepping...")
            # sim.step(100)
            new_pos = (
                sim.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                .value_in_unit(unit.angstrom)
            )
            init_pose = Pose(torch.tensor(new_pos, dtype=torch.float32).to(self.device))

        if optimize:
            pred_pose = self.pp.optimize_pose(data, td, init_pose)
        else:
            pred_pose = init_pose

        # move the mean to the exit point for solvation
        solv_vec = self.exit_point - torch.tensor(CONFIG.dataset.pocket.center).to(
            self.device
        )
        solv_vec_unit = solv_vec / torch.norm(solv_vec)
        # add offset
        solv_vec += solv_vec_unit * CONFIG.scoring.solv_buffer

        solv_pose = Pose(pred_pose.coord + solv_vec)
        score = self.pp.pmf_U(data, pred_pose) - self.pp.pmf_U(data, solv_pose)
        total_U = self.pp.total_U(data, pred_pose)

        return score, total_U, pred_pose

    def __call__(self, mol, system=None, top=None, optimize=None):
        """Returns the score and mol with the optimized pose"""
        # todo: add all the poses in order of increasing score

        if optimize is None:
            optimize = self.minimize_pmf

        if isinstance(mol, list):
            if len(mol) > 0:
                return self(mol[0], system=system, top=top)
            else:
                raise ValueError("Empty list of molecules")

        data = None
        td = None

        if system is None:
            system, top = get_system_and_top_from_mol(mol)

        scores = []
        Us = []
        poses = []

        for conf_id in range(mol.GetNumConformers()):
            mol = Chem.AddHs(mol, addCoords=True)
            if data is None:
                data = mol_to_pmf_input(mol, system=system, top=top).to(self.device)
                td = TorsionData.from_pyg(data).to(self.device)

            try:

                init_coords = mol.GetConformer(conf_id).GetPositions()
                sim = None
                if self.minmize_mm:
                    # get a simulation
                    integrator = mm.LangevinMiddleIntegrator(
                        300 * unit.kelvin,
                        1 / unit.picosecond,
                        2 * unit.femtoseconds,
                    )
                    platform = mm.Platform.getPlatformByName("CUDA")
                    sim = mm.app.Simulation(
                        top,
                        add_pos_restraint(
                            system,
                            top,
                            init_coords * unit.angstrom,
                            self.mm_restraint_k,
                        ),
                        integrator,
                        platform,
                    )

                init_pose = Pose(
                    torch.asarray(init_coords, dtype=torch.float32).to(self.device)
                )
                score, U, optim_pose = self.score_single_pose(
                    data, td, init_pose, optimize=optimize, sim=sim
                )
                scores.append(score.item())
                Us.append(U.item())
                poses.append(optim_pose)
            except KeyboardInterrupt:
                raise
            except:
                print_exc()

        pose_index = int(np.argmin(Us))
        best_score = scores[pose_index]
        best_pose = poses[pose_index]

        indices_sorted = np.argsort(scores)
        sorted_poses = [poses[i] for i in indices_sorted]

        ret_mol = deepcopy(mol)
        # add_pose_to_mol(ret_mol, best_pose)
        add_multi_pose_to_mol(ret_mol, MultiPose.combine(sorted_poses))

        # None is the metadata
        return best_score, ret_mol, None
