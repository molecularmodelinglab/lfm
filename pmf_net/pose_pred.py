from copy import deepcopy
from functools import partial
import shutil
import subprocess
import os
import sys
from traceback import print_exc
from omegaconf import OmegaConf
from torch.multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm, trange
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit.topology import Molecule
from openff.units.openmm import to_openmm
from openmmforcefields.generators import EspalomaTemplateGenerator
from openmm import app, unit
from datasets.pmf_dataset import PMFData, add_gb_params
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
import roma
from common.utils import (
    CONFIG,
    coul_E,
    get_angle_U,
    get_bond_U,
    get_nonbonded_U,
    get_torsion_U,
    lj_E,
    load_sdf,
    protonate_mol,
    save_sdf,
    get_output_dir,
    get_angle_params,
    get_bond_params,
    get_constraints,
    get_nb_matrices,
    get_nonbonded_params,
    get_torsion_params,
)
from common.torsion import TorsionData
from common.pose_transform import MultiPose, Pose, PoseTransform, add_multi_pose_to_mol

def get_system_and_top_from_mol(mol, partial_charges=None):
    """ Returns openmm system and topology from rdkit mol """

    ff_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)

    top = ff_mol.to_topology().to_openmm()
    pos = to_openmm(ff_mol.conformers[0])
    modeller = app.Modeller(top, pos)

    # assign partial charges if we have them
    if partial_charges is not None:
        for i, atom in enumerate(modeller.topology.atoms()):
            atom._partialCharge = partial_charges[i]
            
    generator = EspalomaTemplateGenerator(
        [ff_mol],
        forcefield="espaloma-0.3.2",
    )

    # generator = SMIRNOFFTemplateGenerator([ff_mol])

    forcefield = app.ForceField("amber/ff14SB.xml")
    forcefield.registerTemplateGenerator(generator.generator)

    system = forcefield.createSystem(modeller.topology)

    return system, top


def mol_to_pmf_input(mol, system=None, top=None):
    """Returns a pyg Data object for the given molecule using an
    Espaloma forcefield."""

    if system is None or top is None:
        system, top = get_system_and_top_from_mol(mol)

    pos = mol.GetConformer().GetPositions() * unit.angstrom

    nb_params = torch.tensor(get_nonbonded_params(system), dtype=torch.float32)
    elements = np.array([a.element.atomic_number for a in top.atoms()])
    bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()])
    bond_types = np.array([b.GetBondTypeAsDouble() for b in mol.GetBonds()])

    edge_index = bonds.T
    edge_attr = bond_types

    q_sq_mat, sig_mat, eps_mat = get_nb_matrices(nb_params, system)
    bond_index0, bond_index1, bond0, bond_k = get_bond_params(system)
    angle_index0, angle_index1, angle_index2, angle0, angle_k = get_angle_params(system)
    (
        tor_index0,
        tor_index1,
        tor_index2,
        tor_index3,
        tor_periodicity,
        tor_phase,
        tor_k,
    ) = get_torsion_params(system)

    const_index0, const_index1, const_r0 = get_constraints(system)

    formal_charges = np.array([a.GetFormalCharge() for a in mol.GetAtoms()])

    data = Data(
        pos=torch.tensor(pos.value_in_unit(unit.angstroms), dtype=torch.float32),
        nb_params=nb_params,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edata=torch.tensor(edge_attr, dtype=torch.float32),
        elements=torch.tensor(elements, dtype=torch.long),
        formal_charges=torch.tensor(formal_charges, dtype=torch.long),
        q_sq_mat=q_sq_mat,
        sig_mat=sig_mat,
        eps_mat=eps_mat,
        bond_index0=bond_index0,
        bond_index1=bond_index1,
        bond0=bond0,
        bond_k=bond_k,
        angle_index0=angle_index0,
        angle_index1=angle_index1,
        angle_index2=angle_index2,
        angle0=angle0,
        angle_k=angle_k,
        tor_index0=tor_index0,
        tor_index1=tor_index1,
        tor_index2=tor_index2,
        tor_index3=tor_index3,
        tor_periodicity=tor_periodicity,
        tor_phase=tor_phase,
        tor_k=tor_k,
        const_index0=const_index0,
        const_index1=const_index1,
        const_r0=const_r0,
    )
    if CONFIG.model.get("delta_gb", False):
        add_gb_params(data)
    return data

class PosePred:
    """Use the trained model to predict the pose of the input molecule"""

    def __init__(self, model, only_torsion=True, only_internal_U=False):
        self.model = model
        self.only_torsion = only_torsion
        self.only_internal_U = only_internal_U
        self.constraint_k = 1e6
        self.com_k = 1.0

    def pmf_U(self, data: Data, pose: Pose) -> torch.Tensor:
        """Model PMF energy for this pose"""

        batch = torch.zeros(
            pose.coord.shape[0], dtype=torch.long, device=pose.coord.device
        )
        pos = pose.coord.contiguous()
        x = PMFData.from_pyg_data(data)

        return self.model(x, pos, batch)[0][0, 0]

    def total_U(self, data: Data, pose: Pose) -> torch.Tensor:
        """Total energy (intramolecular + model PMF) for this pose"""

        batch = torch.zeros(
            pose.coord.shape[0], dtype=torch.long, device=pose.coord.device
        )
        pos = pose.coord.contiguous()
        U = get_nonbonded_U(pos, data.q_sq_mat, data.sig_mat, data.eps_mat)

        if not self.only_torsion:
            U += get_bond_U(
                pos, data.bond_index0, data.bond_index1, data.bond0, data.bond_k
            )
            U += get_angle_U(
                pos,
                data.angle_index0,
                data.angle_index1,
                data.angle_index2,
                data.angle0,
                data.angle_k,
            )
            const_k = torch.ones(data.const_index0.shape[0], device=pos.device) * self.constraint_k
            U += get_bond_U(pos, data.const_index0, data.const_index1, data.const_r0, const_k)

        U += get_torsion_U(
            pos,
            data.tor_index0,
            data.tor_index1,
            data.tor_index2,
            data.tor_index3,
            data.tor_periodicity,
            data.tor_phase,
            data.tor_k,
        )

        if not self.only_internal_U:
            x = PMFData.from_pyg_data(data)
            pmf_U = self.model(x, pos, batch)[0][0, 0]
            U += pmf_U

            if self.com_k is not None:
                cur_com = pos.mean(dim=0)
                poc_com = torch.tensor(CONFIG.dataset.pocket.center).to(pos.device)
                # print(cur_com)
                # print(poc_com - cur_com)

                com_U = self.com_k * ((cur_com - poc_com)**2).mean()
                # print(com_U, pmf_U)
                U += com_U

        return U

    def optimize_pose(
        self,
        data: Data,
        td: TorsionData,
        init_pose: Pose,
        verbose: bool = False,
        learn_rate: float = 1.0,
        tolerance_change: float = 5e-2,
        line_search_fn: str = "strong_wolfe",
        **kwargs,
    ) -> Pose:
        """Use L-BFGS to optimize the pose"""

        with torch.no_grad():
            device = data.nb_params.device
            transform = PoseTransform.identity(td).to(device)

            params = []

            if self.only_torsion:
                params.append(nn.Parameter(transform.rot))
                params.append(nn.Parameter(transform.trans))
                params.append(nn.Parameter(transform.tor_angles))
            else:
                params.append(nn.Parameter(deepcopy(init_pose.coord)))

            optimizer = torch.optim.LBFGS(
                params,
                learn_rate,
                line_search_fn=line_search_fn,
                tolerance_change=tolerance_change,
                **kwargs,
            )

            if verbose:
                print("Starting optimization")

            def closure():
                optimizer.zero_grad()
                params = optimizer.param_groups[0]["params"]

                if self.only_torsion:
                    transform = PoseTransform(params[0], params[1], params[2])
                    pose = transform.apply(init_pose, td)
                else:
                    pose = Pose(params[0])

                U = self.total_U(data, pose)

                if verbose:
                    print(f"U: {U.item()}")

                U.backward()
                return U

            optimizer.step(closure)

            params = optimizer.param_groups[0]["params"]
            if self.only_torsion:
                transform = PoseTransform(params[0], params[1], params[2])
                final_pose = transform.apply(init_pose, td)
            else:
                final_pose = Pose(params[0])

        return final_pose

    def init_pose(self, data: Data, td: TorsionData, t_std=1.0) -> Pose:

        device = data.nb_params.device

        cur_com = data.pos.mean(0)
        poc_com = torch.tensor(CONFIG.dataset.pocket.center).to(device)

        t_mean = poc_com - cur_com

        trans = torch.randn(3).to(device) * t_std + t_mean
        rot = roma.random_rotvec()
        tor_angles = 2 * torch.pi * torch.rand(len(td.rot_edges))

        transform = PoseTransform(rot, trans, tor_angles).to(device)

        return transform.apply(Pose(data.pos), td)

    def predict_pose_mcmc(
        self,
        data,
        td,
        t_prob=0.2,
        rot_prob=0.5, # tor prob = 1 - (t_prob + rot_prob)
        beta=1.0,
        n_steps=1,
        t_std=1.0,
        verbose=False,
    ):
        """Predict the pose using MCMC"""

        with torch.no_grad():

            device = data.nb_params.device

            cur_pose = self.init_pose(data, td)
            # in general, we want to first optimize the torsion angles
            self.only_internal_U = True
            cur_pose = self.optimize_pose(data, td, cur_pose)
            self.only_internal_U = False
            cur_pose = self.optimize_pose(data, td, cur_pose)
            cur_U = self.total_U(data, cur_pose)

            for i in trange(n_steps) if verbose else range(n_steps):
                r = np.random.rand()

                trans = torch.zeros(3).to(device)
                rot = torch.zeros(3).to(device)
                tor_angles = torch.zeros(len(td.rot_edges)).to(device)

                if r < t_prob:
                    if verbose:
                        print("Trans")
                    trans = torch.randn(3).to(device) * t_std
                elif r < t_prob + rot_prob:
                    if verbose:
                        print("Rot")
                    rot = roma.random_rotvec()
                    tor_angles = torch.zeros(len(td.rot_edges)).to(device)
                else:
                    if verbose:
                        print("Torsion")
                    if len(td.rot_edges) > 0:
                        # set a random angle to [0, 2pi]
                        index = np.random.randint(0, len(td.rot_edges))
                        tor_angles[index] = np.random.rand() * 2 * np.pi

                transform = PoseTransform(rot, trans, tor_angles).to(device)
                proposal_pose = transform.apply(cur_pose, td)

                # first optimize internal
                self.only_internal_U = True
                proposal_pose = self.optimize_pose(data, td, proposal_pose, verbose=False, tolerance_change=5e-4)
                # reject if current pose is too close to the proposal
                rmsd = torch.sqrt(torch.norm(proposal_pose.coord - cur_pose.coord))
                # print(f"RMSD: {rmsd}")
                if rmsd < 2.0:
                    if verbose:
                        print("Rejected due to closeness")
                    continue

                self.only_internal_U = False
                proposal_pose = self.optimize_pose(data, td, proposal_pose, verbose=False, tolerance_change=5e-4)

                proposal_U = self.total_U(data, proposal_pose)

                # MC
                metropolis = torch.exp(beta * (cur_U - proposal_U))
                if (torch.rand(1).item() < metropolis.item()) or proposal_U < cur_U:
                    cur_pose = proposal_pose
                    cur_U = proposal_U
                    if verbose:
                        print(f"Accepted (U={cur_U.item()})")
                else:
                    if verbose:
                        print("Rejected")

        return cur_pose

    def get_best_initial_poses(
        self, data: Data, td: TorsionData, sample: int, take: int, verbose=False
    ) -> List[Pose]:
        """Take the _take_ best poses from _sample_ random initial poses, according to
        the model PMF energy"""

        init_poses = []
        init_Us = []
        iterable = trange(sample) if verbose else range(sample)
        for i in iterable:
            init = self.init_pose(data, td)
            init_poses.append(init)
            init_Us.append(self.pmf_U(data, init))
        init_Us = torch.stack(init_Us)

        _, idx = torch.topk(init_Us, take, largest=False)
        return [init_poses[i] for i in idx]

    def predict_poses(
        self,
        data: Data,
        embeddings: Optional[torch.Tensor] = None,
        n_poses: int = 128,
        verbose: bool = False,
        **kwargs,
    ) -> List[Pose]:
        """ Predict n_poses poses for the input molecule. If embeddings
        is not None, randomly select an embedding as an initial pose """
        data = deepcopy(data)

        _range = trange if verbose else range

        with torch.no_grad():
            device = data.nb_params.device
            td = TorsionData.from_pyg(data).to(device)

            poses = []
            for i in _range(n_poses):
                if embeddings is not None:
                    idx = np.random.randint(0, embeddings.shape[0])
                    data.pos = embeddings[idx]

                pose = self.predict_pose_mcmc(data, td, **kwargs)
                poses.append(pose)

            # f = partial(self.predict_pose_mcmc, data, td, **kwargs)
            # with Pool() as p:
            #     poses = p.map(f, trange(n_poses))

            Us = [self.total_U(data, p) for p in poses]
            # resort by energy
            poses = [p for _, p in sorted(zip(Us, poses), key=lambda pair: pair[0])]

        return poses
    
    def __call__(self, mol, n_embed=16, n_poses=128, verbose=False):
        """ Given a (2D) rdkit molecule, predict the poses. Returns a new rdkit molecule with the poses """

        mol.RemoveAllConformers()
        mol = protonate_mol(mol)
        mol = Chem.AddHs(mol)

        ps = AllChem.ETKDGv3()
        AllChem.EmbedMultipleConfs(mol,n_embed,ps)
        AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000, numThreads=8)

        data = mol_to_pmf_input(mol)
        embeddings = torch.asarray(
            [mol.GetConformer(i).GetPositions() for i in range(mol.GetNumConformers())],
            dtype=torch.float32,
        )
        poses = self.predict_poses(data, embeddings=embeddings, n_poses=n_poses, verbose=verbose)
        poses = [ p for p in poses if not torch.isnan(p.coord).any() ]
        poses = MultiPose.combine(poses)

        add_multi_pose_to_mol(mol, poses)

        return mol
    
def predict_poses_parellel(model, mols, n_poses=128, n_embed=16):
    """ Calls subprocess.Popen on this file to predict poses in parallel """

    model_deriv = model.derivative
    try:
        model.derivative = False
        out_dir = get_output_dir()
        
        # save config to a file
        config_path = os.path.join(out_dir, "config.yaml")
        OmegaConf.save(CONFIG, config_path)

        # save model to a file
        model_path = os.path.join(out_dir, "model.pt")
        torch.jit.save(model, model_path)

        n_cpu = os.cpu_count()
        # divide the mols into n_cpu chunks
        chunk_size = (len(mols) + n_cpu - 1) // n_cpu
        chunks = [mols[i:i+chunk_size] for i in range(0, len(mols), chunk_size)]

        procs = []
        for i, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue

            input_smi = os.path.join(out_dir, f"input_{i}.smi")
            with open(input_smi, "w") as f:
                for mol in chunk:
                    smi = Chem.MolToSmiles(mol)
                    f.write(smi + "\n")

            output_folder = os.path.join(out_dir, f"{i}")
            env = f"CUDA_VISIBLE_DEVICES='' OUTPUT_DIR={output_folder}"

            cmd = f"{env} python -m pmf_net.pose_pred {config_path} {model_path} {input_smi} {output_folder} {n_poses} {n_embed}"
            print(cmd)
            proc = subprocess.Popen(cmd, shell=True)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(proc)

        for proc in procs:
            proc.wait()

        # now load the mols
        new_mols = []
        for i in range(len(chunks)):
            output_folder = os.path.join(out_dir, f"{i}")
            for j in range(len(chunks[i])):
                output_sdf = os.path.join(output_folder, f"{j}.sdf")
                if os.path.exists(output_sdf):
                    mol = load_sdf(output_sdf)
                else:
                    mol = None
                new_mols.append(mol)

        return new_mols
    finally:
        model.derivative = model_deriv

if __name__ == "__main__":
    """ Args: 
        1. Path to config file
        2. Path to model pt file
        3. Path to input smi file
        4. Path to output folder (where we store the sdf files)
        5. n_poses (optional)
        6. n_embed (optional)
    """
    
    from pmf_net.train import *

    config_path = sys.argv[1]
    model_path = sys.argv[2]
    input_smi = sys.argv[3]
    output_folder = sys.argv[4]
    n_poses = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    n_embed = int(sys.argv[6]) if len(sys.argv) > 6 else 16


    load_config(config_path, False)

    model = torch.jit.load(model_path, map_location='cpu')
    pp = PosePred(model)
    
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)

    with open(input_smi, "r") as f:
        for i, smi in enumerate(tqdm(f.readlines())):
            try:
                smi = smi.strip()
                mol = Chem.MolFromSmiles(smi)
                
                mol = pp(mol, n_embed=n_embed, n_poses=n_poses)
        
                output_sdf = os.path.join(output_folder, f"{i}.sdf")
                save_sdf(mol, output_sdf)
            except:
                print_exc()
                print(f"Error with {smi}")