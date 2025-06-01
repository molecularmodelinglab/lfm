from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
import os
import shutil
from typing import Dict

from common.alignment import find_rigid_alignment, rigid_align
from common.openmm_utils import make_system
import zarr
import numpy as np
from tqdm import tqdm
import parmed as pmd
from openmm import app
from openmm import unit
import openmm as mm
from rdkit import Chem
import torch
from torch import nn
from openmmtools import alchemy, states
from openmmtorch import TorchForce
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import EspalomaTemplateGenerator
from common.utils import (
    combine_modellers,
    get_CA_indices,
    get_nonbonded_params,
    get_output_dir,
    save_modeller_pdb,
)


def neutralize_mol(lig, lig_rd, cache_dir=None):
    """Neutralizes the system without adding waters"""

    ret = app.Modeller(lig.topology, lig.positions)

    forcefield = app.ForceField("amber/ff14SB.xml", "amber14/tip3p.xml")
    off_mols = [Molecule.from_rdkit(lig_rd)]

    generator = EspalomaTemplateGenerator(
        off_mols,
        cache=None if cache_dir is None else f"{cache_dir}/espaloma.json",
        forcefield="espaloma-0.3.2",
    )
    forcefield.registerTemplateGenerator(generator.generator)

    ret.addSolvent(forcefield)
    ret.delete([res for res in ret.topology.residues() if res.name == "HOH"])

    return ret


def rigid_transform_Kabsch_3D_torch(A, B):

    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, dim=-1, keepdim=True)
    centroid_B = torch.mean(B, dim=-1, keepdim=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.transpose(-2, -1)

    # find rotation
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)

    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

    s = R.det().sign()
    R = torch.cat((R[..., :-1], (R[..., -1] * s.unsqueeze(-1)).unsqueeze(-1)), -1)

    t = -R @ centroid_A + centroid_B
    return R, t


# these functions are inspired by openmm's efficient implementation
# https://github.com/openmm/openmm/blob/91966dc8eb976fadc206e8a345470a737c09fd8b/wrappers/python/openmm/app/internal/compiled.pyx#L44
def periodic_delta(pos1, pos2, box_vectors):
    """
    Compute the periodic delta between two points given the box vectors.
    """
    delta = pos2 - pos1
    inv_box_size = 1.0 / torch.diag(
        box_vectors
    )  # Inverse of the box size along each axis

    # Apply periodic boundary conditions in reverse order (z -> y -> x)
    scale3 = torch.round(delta[2] * inv_box_size[2])
    delta -= scale3 * box_vectors[2]

    scale2 = torch.round(delta[1] * inv_box_size[1])
    delta -= scale2 * box_vectors[1]

    scale1 = torch.round(delta[0] * inv_box_size[0])
    delta -= scale1 * box_vectors[0]

    return delta


def periodic_delta_many(pos1, pos2, box_vectors):
    """
    Compute the periodic distance between two sets of points given the box vectors.

    Args:
        pos1 (torch.Tensor): Tensor of shape (N, 3) representing the first set of positions.
        pos2 (torch.Tensor): Tensor of shape (N, 3) representing the second set of positions.
        box_vectors (torch.Tensor): Tensor of shape (3,3) representing the box vectors.

    """
    delta = pos2 - pos1  # Shape: (N, 3)
    inv_box_size = 1.0 / torch.diag(box_vectors)  # Shape: (3,)

    # Apply periodic boundary conditions in reverse order (z -> y -> x)
    scale3 = torch.round(delta[:, 2] * inv_box_size[2]).unsqueeze(-1)  # Shape: (N, 1)
    delta -= scale3 * box_vectors[2]  # Shape: (N, 3)

    scale2 = torch.round(delta[:, 1] * inv_box_size[1]).unsqueeze(-1)  # Shape: (N, 1)
    delta -= scale2 * box_vectors[1]  # Shape: (N, 3)

    scale1 = torch.round(delta[:, 0] * inv_box_size[0]).unsqueeze(-1)  # Shape: (N, 1)
    delta -= scale1 * box_vectors[0]  # Shape: (N, 3)

    return delta


def periodic_delta_many_numpy(pos1, pos2, box_vectors):
    """Same as above, but used numpy instead of torch"""

    delta = pos2 - pos1  # Shape: (N, 3)
    inv_box_size = 1.0 / np.diag(box_vectors)  # Shape: (3,)

    # Apply periodic boundary conditions in reverse order (z -> y -> x)
    scale3 = np.round(delta[:, 2] * inv_box_size[2]).reshape(-1, 1)  # Shape: (N, 1)
    delta -= scale3 * box_vectors[2]  # Shape: (N, 3)

    scale2 = np.round(delta[:, 1] * inv_box_size[1]).reshape(-1, 1)  # Shape: (N, 1)
    delta -= scale2 * box_vectors[1]  # Shape: (N, 3)

    scale1 = np.round(delta[:, 0] * inv_box_size[0]).reshape(-1, 1)  # Shape: (N, 1)
    delta -= scale1 * box_vectors[0]  # Shape: (N, 3)

    return delta


def best_periodic_translation(pos1, pos2, box_vectors):
    """
    Compute the a vector i*bv[0] + j*bv[1] + k*bv[2] that minimizes the distance between pos1 and pos2.
    """

    delta = pos2 - pos1
    inv_box_size = 1.0 / torch.diag(
        box_vectors
    )  # Inverse of the box size along each axis

    # Apply periodic boundary conditions in reverse order (z -> y -> x)
    scale3 = torch.round(delta[2] * inv_box_size[2])
    delta -= scale3 * box_vectors[2]

    scale2 = torch.round(delta[1] * inv_box_size[1])
    delta -= scale2 * box_vectors[1]

    scale1 = torch.round(delta[0] * inv_box_size[0])
    delta -= scale1 * box_vectors[0]

    ret = scale1 * box_vectors[0] + scale2 * box_vectors[1] + scale3 * box_vectors[2]
    return ret


def best_periodic_translation_numpy(pos1, pos2, box_vectors):
    """Same as above, but used numpy instead of torch"""

    delta = pos2 - pos1
    inv_box_size = 1.0 / np.diag(box_vectors)  # Inverse of the box size along each axis

    # Apply periodic boundary conditions in reverse order (z -> y -> x)
    scale3 = np.round(delta[2] * inv_box_size[2])
    delta -= scale3 * box_vectors[2]

    scale2 = np.round(delta[1] * inv_box_size[1])
    delta -= scale2 * box_vectors[1]

    scale1 = np.round(delta[0] * inv_box_size[0])
    delta -= scale1 * box_vectors[0]

    ret = scale1 * box_vectors[0] + scale2 * box_vectors[1] + scale3 * box_vectors[2]
    return ret


class AlignmentForce(nn.Module):
    """Force that aligns the pocket alpha carbons to a reference frame"""

    def __init__(self, topology, rec_indices, poc_indices, ref_poc, k):
        super().__init__()

        chain_ids = np.array([atom.residue.chain.index for atom in topology.atoms()])
        chain_ids = chain_ids[rec_indices]
        poc_chain_ids = chain_ids[poc_indices]

        chain_ids = torch.tensor(chain_ids)
        poc_chain_ids = torch.tensor(poc_chain_ids)

        rec_indices = torch.asarray(rec_indices)
        poc_indices = torch.asarray(poc_indices)
        ref_poc = torch.asarray(ref_poc, dtype=torch.float32)
        k = torch.tensor(
            k.value_in_unit(unit.kilojoule_per_mole / (unit.nanometer**2)),
            dtype=torch.float32,
        )

        self.register_buffer("chain_ids", chain_ids)
        self.register_buffer("poc_chain_ids", poc_chain_ids)
        self.register_buffer("rec_indices", rec_indices)
        self.register_buffer("ref_poc", ref_poc)
        self.register_buffer("poc_indices", poc_indices)
        self.register_buffer("k", k)

        for chain_id in torch.unique(self.poc_chain_ids):
            poc_chain_mask = self.poc_chain_ids == chain_id
            chain_mask = self.chain_ids == chain_id
            chain_indices = self.rec_indices[chain_mask]
            poc_chain_indices = self.poc_indices[poc_chain_mask]
            ref_poc_chain_com = self.ref_poc[poc_chain_mask].mean(dim=0)

            self.register_buffer(f"poc_chain_{chain_id}_mask", poc_chain_mask)
            self.register_buffer(f"chain_{chain_id}_mask", chain_mask)
            self.register_buffer(f"chain_{chain_id}_indices", chain_indices)
            self.register_buffer(f"poc_chain_{chain_id}_indices", poc_chain_indices)
            self.register_buffer(f"ref_poc_chain_{chain_id}_com", ref_poc_chain_com)

        self.unique_chain_ids = torch.unique(self.poc_chain_ids).tolist()

    def forward(self, positions, box_vectors):
        rec_pos = positions[self.rec_indices]

        # unwrap the positions per-chain so it's closer to the reference

        chain_mask = self.chain_0_mask
        poc_chain_indices = self.poc_chain_0_indices
        ref_poc_chain_com = self.ref_poc_chain_0_com
        for chain_id in self.unique_chain_ids:
            chain_mask = getattr(self, f"chain_{chain_id}_mask")
            poc_chain_indices = getattr(self, f"poc_chain_{chain_id}_indices")
            ref_poc_chain_com = getattr(self, f"ref_poc_chain_{chain_id}_com")

            poc_chain_pos = positions[poc_chain_indices]
            poc_chain_com = poc_chain_pos.mean(dim=0)

            best_t = best_periodic_translation(
                poc_chain_com, ref_poc_chain_com, box_vectors
            )
            rec_pos = rec_pos + best_t * chain_mask[:, None].float()

        poc_pos = rec_pos[self.poc_indices]

        rec_origin = rec_pos.mean(dim=0)
        rec_centered = rec_pos - rec_origin
        r_sq = torch.sum(rec_centered**2, dim=1)

        # delta = periodic_delta_many(self.ref_poc, poc_pos, box_vectors)
        # assert torch.allclose((poc_pos - self.ref_poc), delta)

        F = torch.zeros_like(rec_pos)
        F[self.poc_indices] = -2 * self.k * (poc_pos - self.ref_poc)

        # now get net force and torque
        F_mean = F.mean(dim=0)
        torque = torch.linalg.cross(rec_centered, F)
        torque_mean = torque.mean(dim=0)

        # now find rigid-body forces
        F_rot = torch.linalg.cross(torque_mean[None], rec_centered) / r_sq[:, None]

        F_final = torch.zeros_like(positions)
        F_final[self.rec_indices] = F_mean + F_rot

        return torch.tensor(0.0, dtype=torch.float32), F_final


class LigandRestraintForce(nn.Module):
    """Harmonic force that keeps the ligand COM close to the original COM"""

    def __init__(self, lig_indices, lig_coords, k):
        super().__init__()
        lig_indices = torch.asarray(lig_indices)
        lig_coords = torch.asarray(lig_coords, dtype=torch.float32)
        ref_com = lig_coords.mean(axis=0)
        k = torch.tensor(k.value_in_unit(unit.kilojoule_per_mole / (unit.nanometer**2)))

        self.register_buffer("lig_indices", lig_indices)
        self.register_buffer("ref_com", ref_com)
        self.register_buffer("k", k)
        self.register_buffer("Nlig", torch.tensor(len(lig_indices)))

    def forward(self, positions, box_vectors):
        lig_pos = positions[self.lig_indices]
        lig_com = lig_pos.mean(dim=0)
        delta = periodic_delta(self.ref_com, lig_com, box_vectors)
        U = self.k * torch.sum(delta**2)
        F = -2 * self.k * delta / self.Nlig
        F_all = torch.zeros_like(positions)
        F_all[self.lig_indices] = F

        return U, F_all


def set_nonbonded_params_i(nb_force, i, nb_param):
    charge, sigma, eps = nb_param
    nb_force.setParticleParameters(
        i,
        charge * unit.elementary_charge,
        sigma * unit.nanometers,
        eps * unit.kilojoules_per_mole,
    )


TORCH_FORCE_GROUP = 3
MAX_FORCE_GROUP = 32

def com_restraint_force(lig_indices, poc_indices, ref_distance, k):
    """Returns a force that restrains the distance between the ligand center of mass
    and the pocket alpha carbon atoms to be close to its original distance.
    """

    force = mm.CustomCentroidBondForce(2, "0.5*k*(distance(g1, g2) - d0)^2")
    force.addGlobalParameter("k", k)
    force.addGlobalParameter("d0", ref_distance)

    force.addGroup(lig_indices, [1] * len(lig_indices))
    force.addGroup(poc_indices, [1] * len(poc_indices))
    force.addBond([0, 1])

    return force


class PMFSim:
    """Class for running and saving data for training PMF models"""

    systems: Dict[str, mm.System]
    mols: Dict[str, app.Modeller]

    def __init__(
        self,
        rec: app.Modeller,
        lig_noion: app.Modeller,
        lig_rd: Chem.Mol,
        rec_reference: app.Modeller,  # the reference pose of the receptor (no water)
        poc_indices: np.ndarray,  # indices of the pocket atoms
        extra_mols = None,
        use_systems: bool = True,
        neutralize: bool = False,  # explicitly neutralize the ligand-only system ?
    ):

        cache_dir = f"{get_output_dir()}/cache"

        lig_noion.topology.setPeriodicBoxVectors(rec.topology.getPeriodicBoxVectors())
        lig_ion = (
            neutralize_mol(lig_noion, lig_rd, cache_dir) if neutralize else lig_noion
        )

        self.mols = {
            "rec": rec,
            "lig_noion": lig_noion,
            "lig_ion": lig_ion,
            "lr": combine_modellers(rec, lig_ion),
        }
        self.lig_rd = lig_rd

        # save_modeller_pdb(self.mols["lr"], "output/lr.pdb")

        shutil.rmtree(cache_dir, ignore_errors=True)

        mols = [lig_rd]
        if extra_mols is not None:
            if isinstance(extra_mols, list):
                mols.extend(extra_mols)
            else:
                mols.append(extra_mols)

        if use_systems:
            self.systems = {
                key: make_system(
                    mol,
                    mols=mols,
                    solvent="tip3p" if key != "lig_noion" else "none",
                    add_waters=False,
                    cache_dir=cache_dir,
                )[0]
                for key, mol in self.mols.items()
            }

            self.contexts = {
                key: mm.Context(
                    system,
                    mm.LangevinIntegrator(
                        300 * unit.kelvin,
                        1.0 / unit.picoseconds,
                        4.0 * unit.femtoseconds,
                    ),
                )
                for key, system in self.systems.items()
            }

        rec_indices = np.arange(len(rec.positions))
        lig_ion_indices = np.arange(
            len(rec.positions), len(rec.positions) + len(lig_ion.positions)
        )
        lig_noion_indices = np.arange(
            len(rec.positions), len(rec.positions) + len(lig_noion.positions)
        )
        poc_alpha_indices = np.array(
            [i for i in poc_indices if i in get_CA_indices(rec_reference.topology)]
        )
        self.rec_ref_indices = np.arange(len(rec_reference.positions))

        self.rec_indices = rec_indices
        self.lig_ion_indices = lig_ion_indices
        self.lig_noion_indices = lig_noion_indices
        self.poc_indices = poc_indices
        self.poc_alpha_indices = poc_alpha_indices

        init_poc_pos = rec.positions[poc_alpha_indices].value_in_unit(unit.nanometers)
        self.sim_origin = init_poc_pos.mean(axis=0)

        # ref_poc_* means the reference frame in the universal reference used by all the simulations
        # sim_poc_* means the frame in the simulation's reference frame

        self.ref_poc_pos = rec_reference.positions[poc_alpha_indices].value_in_unit(
            unit.nanometers
        )
        self.ref_origin = self.ref_poc_pos.mean(axis=0)
        # rotation that takes us from the simulation frame to the reference frame
        self.ref_R, _ = find_rigid_alignment(
            init_poc_pos - self.sim_origin, self.ref_poc_pos - self.ref_origin
        )
        # rotation that takes us from the reference frame to the simulation frame
        self.sim_R = np.linalg.inv(self.ref_R)

        self.sim_poc_pos = (
            rigid_align(
                self.ref_poc_pos - self.ref_origin, init_poc_pos - self.sim_origin
            )
            + self.sim_origin
        )

        self.set_init_lig_pos(lig_noion.positions.value_in_unit(unit.nanometers))

    def set_init_lig_pos(self, pos):
        self.init_lig_pos = pos

        self.ref_lig_pos = (
            self.ref_R.dot((self.init_lig_pos - self.sim_origin).T).T + self.ref_origin
        )

    def make_alchemical_system(self):
        """Returns alchemically-modifed LR system"""
        factory = alchemy.AbsoluteAlchemicalFactory()
        region = alchemy.AlchemicalRegion(alchemical_atoms=self.lig_ion_indices)
        return factory.create_alchemical_system(self.systems["lr"], region)

    def make_lr_simulation(
        self,
        system,
        dt: unit.Quantity = 4.0 * unit.femtoseconds,
        k=1e4 * unit.kilojoule_per_mole / (unit.nanometer**2),
        freeze_ligand=True,
        lig_restraint_k=None,
        lig_restraint_version=1,
    ):
        """Returns a simulation for the LR system"""

        system = deepcopy(system)

        if freeze_ligand:
            for i in self.lig_noion_indices:
                system.setParticleMass(i, 0.0)

        if lig_restraint_k is not None:
            # force = com_restraint_force(
            #     self.lig_noion_indices, self.init_lig_pos, lig_restraint_k
            # )
            og_pos = np.array(self.mols["lr"].positions.value_in_unit(unit.nanometers))
            if lig_restraint_version == 0:
                ref_dist = (
                    np.linalg.norm(
                        og_pos[self.lig_noion_indices].mean(axis=0)
                        - og_pos[self.poc_alpha_indices].mean(axis=0)
                    )
                    * unit.nanometers
                )
                force = com_restraint_force(
                    self.lig_noion_indices,
                    self.poc_alpha_indices,
                    ref_dist,
                    lig_restraint_k,
                )
            elif lig_restraint_version == 1:
                module = torch.jit.script(
                    LigandRestraintForce(
                        self.lig_noion_indices,
                        og_pos[self.lig_noion_indices],
                        lig_restraint_k,
                    )
                )
                force = TorchForce(module)
                force.setUsesPeriodicBoundaryConditions(True)
                force.setOutputsForces(True)
                force.setProperty("useCUDAGraphs", "true")

            else:
                raise ValueError(
                    f"Unknown lig_restraint_version {lig_restraint_version}"
                )

            system.addForce(force)

        module = torch.jit.trace(
            AlignmentForce(
                self.mols["lr"].topology,
                self.rec_ref_indices,
                self.poc_alpha_indices,
                self.sim_poc_pos,
                k,
            ),
            example_inputs=(
                torch.zeros((len(self.mols["lr"].positions), 3), dtype=torch.float32),
                torch.zeros((3, 3), dtype=torch.float32),
            ),
        )
        force = TorchForce(module)
        force.setUsesPeriodicBoundaryConditions(True)
        force.setOutputsForces(True)
        force.setProperty("useCUDAGraphs", "true")
        force.setForceGroup(TORCH_FORCE_GROUP)
        system.addForce(force)

        mol = self.mols["lr"]
        platform = mm.Platform.getPlatformByName("CUDA")

        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picoseconds, dt
        )
        sim = app.Simulation(mol.topology, system, integrator, platform)

        return sim

    def step_lr_simulation(self, sim, n_steps: int = 1):
        """Run n_steps of the LR simulation, ensuring that the ligand
        is always in the same location w/r/t the pocket coordinates"""
        sim.step(n_steps)

    def run_alchemical_lr_sim(
        self,
        coul_time: unit.Quantity = 10 * unit.picoseconds,
        sterics_time: unit.Quantity = 90 * unit.picoseconds,
        final_time: unit.Quantity = 0 * unit.picoseconds,
        dt: unit.Quantity = 4.0 * unit.femtoseconds,
        steps_per_update: int = 3,
        minimize: bool = False,
        initial_steps: int = 0,
        freeze_ligand=True,
        lig_restraint_k=None,
        lig_restraint_version=1,
        init_state=None,
        ignore_velocities=False,
        dcd_fname=None,
        reporter_interval=100,
    ) -> states.SamplerState:
        """Run a short simulation of the LR system, increasing
        the alchemical coupling over time so we remove all clashes.
        This returns the final frame."""

        coul_steps = int(coul_time / dt) // steps_per_update
        sterics_steps = int(sterics_time / dt) // steps_per_update
        final_steps = int(final_time / dt) // steps_per_update

        coul_lambdas = np.linspace(0, 1, coul_steps)
        sterics_lambdas = np.linspace(0, 1, sterics_steps)

        # some more steps at both lambdas = 1. Not sure why we need it...
        coul_lambdas = np.concatenate([coul_lambdas, np.ones(final_steps)])

        all_coul_lambdas = np.concatenate(
            [np.zeros_like(sterics_lambdas), coul_lambdas]
        )
        all_sterics_lambdas = np.concatenate(
            [sterics_lambdas, np.ones_like(coul_lambdas)]
        )

        alc_system = self.make_alchemical_system()
        sim = self.make_lr_simulation(
            alc_system,
            dt,
            freeze_ligand=freeze_ligand,
            lig_restraint_k=lig_restraint_k,
            lig_restraint_version=lig_restraint_version,
        )

        if dcd_fname is not None:
            pdb_fname = dcd_fname.replace(".dcd", ".pdb")
            save_modeller_pdb(self.mols["lr"], pdb_fname)
            dcd_reporter = app.DCDReporter(dcd_fname, reporter_interval)
            sim.reporters.append(dcd_reporter)

        if init_state is not None:
            init_state.apply_to_context(
                sim.context, ignore_velocities=ignore_velocities
            )
        else:
            sim.context.setPositions(self.mols["lr"].positions)
            sim.context.setPeriodicBoxVectors(
                *self.mols["rec"].topology.getPeriodicBoxVectors()
            )

        alc_state = alchemy.AlchemicalState.from_system(alc_system)
        alc_state.set_alchemical_parameters(0.0)
        alc_state.apply_to_context(sim.context)

        if minimize:
            print("Minimizing...")
            sim.minimizeEnergy()
            print("Running initial steps...")
            self.step_lr_simulation(sim, initial_steps)

        # for some reason computing the protocol work results in NaNs...
        # protocol_work = 0.0 * unit.kilojoules_per_mole
        # cur_U = sim.context.getState(getEnergy=True).getPotentialEnergy()

        with tqdm(total=len(all_coul_lambdas)) as pbar:
            for coul_lambda, sterics_lambda in zip(
                all_coul_lambdas, all_sterics_lambdas
            ):
                # energy_initial = sim.context.getState(getEnergy=True).getPotentialEnergy()

                alc_state.lambda_electrostatics = coul_lambda
                alc_state.lambda_sterics = sterics_lambda
                alc_state.apply_to_context(sim.context)

                # energy_final = sim.context.getState(getEnergy=True).getPotentialEnergy()
                # protocol_work += (energy_final - energy_initial)

                self.step_lr_simulation(sim, steps_per_update)
                pbar.update(1)
                pbar.set_description(f"λs: {sterics_lambda:.2f} λc: {coul_lambda:.2f}")

        return states.SamplerState.from_context(sim.context)  # , protocol_work

    def set_init_lig_pos_to_state(self, state):
        """Reset the ligand positions according to a state we get
        from the alchemical simulation"""

        new_lig_pos = state.positions.value_in_unit(unit.nanometers)[
            self.lig_ion_indices
        ]

        # find the periodic image of new_lig_pos closest to self.init_lig_pos
        bv = state.box_vectors.value_in_unit(unit.nanometers)
        new_center = new_lig_pos.mean(axis=0)
        old_center = self.init_lig_pos.mean(axis=0)

        cur_min = (np.inf, None)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    mul = np.tile(np.array([i, j, k]), 3).reshape(3, 3).T
                    shift = (bv * mul).sum(axis=0)
                    new_center_shifted = new_center + shift
                    dist = np.linalg.norm(new_center_shifted - old_center)

                    if dist < cur_min[0]:
                        cur_min = (dist, shift)

        new_lig_pos_shifted = new_lig_pos + cur_min[1]
        self.set_init_lig_pos(new_lig_pos_shifted)

    def run_pmf_sim(
        self,
        init_state: states.SamplerState,
        total_time: unit.Quantity,
        report_interval: unit.Quantity = 10.0 * unit.picoseconds,
        dt: unit.Quantity = 4.0 * unit.femtoseconds,
        dcd_fname: str = None,
        return_state=False,
        ignore_velocities=False,
        freeze_ligand=True,
        lig_restraint_k=None,
        lig_restraint_version=1,
        save_hessians=False,
    ):
        """Run a PMF simulation for the given total time, saving the
        positions, (intermolecular) forces, and energies to the given google storage
        path."""

        positions_all = []
        forces_all = []
        energies_all = []
        hessians_all = [] if save_hessians else None

        sim = self.make_lr_simulation(
            self.systems["lr"],
            dt,
            freeze_ligand=freeze_ligand,
            lig_restraint_k=lig_restraint_k,
            lig_restraint_version=lig_restraint_version,
        )
        if dcd_fname is not None:
            pdb_fname = dcd_fname.replace(".dcd", ".pdb")
            save_modeller_pdb(self.mols["lr"], pdb_fname)

            dcd_reporter = app.DCDReporter(dcd_fname, 500)
            sim.reporters.append(dcd_reporter)

        init_state.apply_to_context(sim.context, ignore_velocities=ignore_velocities)

        n_steps = int(total_time / dt)
        steps_per_report = int(report_interval / dt)
        n_reports = int(total_time / report_interval)

        with tqdm(total=n_reports) as pbar:
            for i in range(n_reports):
                sim.step(steps_per_report)
                state = sim.context.getState(
                    getEnergy=True, getPositions=True, getForces=True
                )
                U = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                sampler_state = states.SamplerState(
                    positions=state.getPositions(),
                    box_vectors=state.getPeriodicBoxVectors(),
                )

                pos = sampler_state.positions.value_in_unit(unit.nanometers)[self.lig_noion_indices]
                # make it as close as possible to the initial ligand position
                t = best_periodic_translation_numpy(
                    pos,
                    self.init_lig_pos,
                    sampler_state.box_vectors.value_in_unit(unit.nanometers),
                )
                pos += t
                # now rotate the ligand positions to the reference frame
                pos = self.ref_R.dot((pos - self.sim_origin).T).T + self.ref_origin

                forces = self.get_intermolecular_forces(sampler_state)

                positions_all.append(pos)
                forces_all.append(forces)
                energies_all.append(U)

                if save_hessians:
                    hessian = self.get_lig_hessian(sampler_state)
                    hessians_all.append(hessian)

                pbar.set_description(f"U: {U:.0f}")
                pbar.update(1)

        positions_all = np.array(positions_all, dtype=np.float32)
        forces_all = np.array(forces_all, dtype=np.float32)
        energies_all = np.array(energies_all, dtype=np.float32)
        if save_hessians:
            hessians_all = np.array(hessians_all, dtype=np.float32)

        if return_state:
            state = states.SamplerState.from_context(sim.context)
            return state

        return positions_all, energies_all, forces_all, hessians_all

    def lig_lig_forces(self, state: states.SamplerState) -> np.ndarray:
        """Returns the forces acting on the lig atoms in the lig system"""
        context = self.contexts["lig_noion"]
        context.setPositions(state.positions[self.lig_noion_indices])
        context.setPeriodicBoxVectors(*state.box_vectors)
        return context.getState(getForces=True).getForces(asNumpy=True)

    def lig_lr_forces(self, state: states.SamplerState) -> np.ndarray:
        """Returns the forces acting on the lig atoms in the LR system"""
        context = self.contexts["lr"]
        context.setPositions(state.positions)
        context.setPeriodicBoxVectors(*state.box_vectors)
        # exclude the torch force
        groups = ((2**MAX_FORCE_GROUP) - 1) ^ (1 << TORCH_FORCE_GROUP)

        state = context.getState(getForces=True, groups=groups)
        return state.getForces(asNumpy=True)[self.lig_noion_indices].value_in_unit(
            unit.kilojoules_per_mole / unit.nanometers
        )

    def get_intermolecular_forces(self, state: states.SamplerState) -> np.ndarray:
        """Returns the nonbonded forces acting on the ligand atoms, rotated to be
        in the reference frame. Values are in kJ/mol/nm."""

        lig_forces = self.lig_lig_forces(state).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometers
        )
        lr_forces = self.lig_lr_forces(state)
        # lr_forces = raw_forces[self.topography.lig_indices]

        # subtract the ligand-ligand forces from the ligand-receptor forces
        intermolecular_forces = lr_forces - lig_forces
        intermolecular_forces = self.ref_R.dot(intermolecular_forces.T).T

        return intermolecular_forces

    def get_lig_nb_grad(self, state: states.SamplerState, epsilon=5e-3):
        """Returns the gradient of the ligand's nonbonded energy w/r/t the
        nonbonded parameters (charge, sigma, epsilon)"""

        lig_nb_params = get_nonbonded_params(self.systems["lig_noion"])

        nb_force = None
        for i in range(self.systems["lr"].getNumForces()):
            force = self.systems["lr"].getForce(i)
            if isinstance(force, mm.NonbondedForce):
                nb_force = force
                break

        lig_nb_grad = []
        state.apply_to_context(self.contexts["lr"])
        U = self.contexts["lr"].getState(getEnergy=True).getPotentialEnergy()

        for i, nb_param in zip(self.lig_noion_indices, tqdm(lig_nb_params)):
            cur_grad = []
            for j in range(3):
                nb_param_tweaked = nb_param.copy()
                nb_param_tweaked[j] += epsilon
                set_nonbonded_params_i(nb_force, i, nb_param_tweaked)
                context = mm.Context(
                    self.systems["lr"],
                    mm.LangevinIntegrator(
                        300 * unit.kelvin,
                        1.0 / unit.picoseconds,
                        4.0 * unit.femtoseconds,
                    ),
                )
                state.apply_to_context(context)

                U_tweaked = context.getState(getEnergy=True).getPotentialEnergy()
                dU = (U_tweaked - U).value_in_unit(unit.kilojoules_per_mole)
                cur_grad.append(dU / epsilon)

                set_nonbonded_params_i(nb_force, i, nb_param)
            lig_nb_grad.append(cur_grad)

        return np.asarray(lig_nb_grad, dtype=np.float32)

    def get_lig_hessian(self, state, epsilon=1e-3):
        """Returns the Hessian of the ligand's nonbonded energy w/r/t the positions"""

        # todo: this needs to return the result _in the reference coordinate system!

        context = self.contexts["lr"]
        state.apply_to_context(context)
        cur_forces = self.get_intermolecular_forces(state)
        cur_pos = (
            context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometers)
        )

        hes_rows = []
        for i in self.lig_noion_indices:
            for j in range(3):
                all_dFs = []
                for eps in [-epsilon, epsilon]:

                    # tweak the position in the reference frame
                    vec_tweaked = np.zeros(3, dtype=np.float32)
                    vec_tweaked[j] = eps
                    vec = self.sim_R.dot(vec_tweaked.T).T

                    state_tweaked = deepcopy(state)
                    state_tweaked.positions[i] += vec * unit.nanometers
                    forces = self.get_intermolecular_forces(state_tweaked)
                    all_dFs.append((forces - cur_forces) / eps)
                hes_row = -np.array(all_dFs).mean(axis=0)
                hes_rows.append(hes_row.flatten())

        hes = np.array(hes_rows)
        return hes
