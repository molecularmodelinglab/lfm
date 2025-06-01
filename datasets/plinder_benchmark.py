from collections import defaultdict
from traceback import print_exc
from tqdm import tqdm
from common.pose_transform import Pose, sym_rmsd
from common.torsion import TorsionData
from pmf_net.pose_pred import PosePred
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from rdkit import Chem
import os
from common.utils import (
    CONFIG,
    get_angle_params,
    get_bond_params,
    get_constraints,
    get_nb_matrices,
    get_torsion_params,
)
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import openmm as mm


class PlinderBenchmark(Dataset):
    """PLINDER docking (and soon free energy) benchmark"""

    def __init__(self, split, use_pK=False):
        """Split is either 'train', 'val', or 'test'."""

        self.folder = os.path.join(CONFIG.plinder_benchmark_dir, CONFIG.dataset.id)
        df_path = os.path.join(self.folder, f"{split}.csv")
        h5_path = os.path.join(self.folder, f"{split}.h5")

        self.df = pd.read_csv(df_path)
        if use_pK:
            self.df = self.df[~np.isnan(self.df["pK"])]
        self.df["pchembl_value"] = self.df.pK

        self.file = h5py.File(h5_path, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of bounds")

        id = self.df.iloc[index]["sys_id"]
        group = self.file[id]

        true_pos = group["true_pos"][:] * 10  # Convert from nm to Angstrom
        rand_pos = group["rand_pos"][:] * 10
        nb_params = group["nb_params"][:]
        bonds = group["bonds"][:]
        bond_types = group["bond_types"][:]
        elements = group["elements"][:]
        formal_charges = group["formal_charges"][:]

        edge_index = bonds.T
        edge_attr = bond_types

        pK = self.df.iloc[index]["pK"]

        nb_params = torch.tensor(nb_params, dtype=torch.float32)
        system = self.get_system(index)

        q_sq_mat, sig_mat, eps_mat = get_nb_matrices(nb_params, system)
        bond_index0, bond_index1, bond0, bond_k = get_bond_params(system)
        angle_index0, angle_index1, angle_index2, angle0, angle_k = get_angle_params(
            system
        )
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

        return Data(
            true_pos=torch.tensor(true_pos, dtype=torch.float32),
            pos=torch.tensor(rand_pos, dtype=torch.float32),
            nb_params=nb_params,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edata=torch.tensor(edge_attr, dtype=torch.float32),
            elements=torch.tensor(elements, dtype=torch.long),
            formal_charges=torch.tensor(formal_charges, dtype=torch.long),
            pK=torch.tensor(pK, dtype=torch.float32),
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

    def get_system(self, index):
        """Returns the openmm system for the given index."""
        systems_folder = os.path.join(self.folder, "systems")

        row = self.df.iloc[index]
        system_file = os.path.join(systems_folder, f"sys_{row.sys_id}_{row.lig_id}.xml")

        # load system into OpenMM
        with open(system_file, "r") as f:
            system = mm.XmlSerializer.deserialize(f.read())

        return system

    def get_rand_sdf(self, index):
        """Returns the (random) SDF file for the given index."""

        sdf_folder = os.path.join(self.folder, "structures")

        row = self.df.iloc[index]
        sdf_file = os.path.join(sdf_folder, f"rand_lig_{row.sys_id}_{row.lig_id}.sdf")

        return sdf_file

    def get_true_sdf(self, index):
        """Returns the true pose SDF file for the given index."""

        sdf_folder = os.path.join(self.folder, "structures")

        row = self.df.iloc[index]
        sdf_file = os.path.join(sdf_folder, f"lig_{row.sys_id}_{row.lig_id}.sdf")

        return sdf_file

    def get_gnina_score_sdf(self, index):
        folder = get_plinder_scoring_folder("gnina")
        return f"{folder}/{index}.sdf"


class PlinderDecoyBenchmark(Dataset):
    """PLINDER decoy benchmark generated from DUD-E. This loads in the
    pose generated by gnina"""

    def __init__(self, split):
        """Only val and test splits are available."""

        self.folder = os.path.join(
            CONFIG.plinder_benchmark_dir, CONFIG.dataset.id, f"{split}_decoys"
        )
        h5_path = os.path.join(self.folder, "all.h5")

        self.file = h5py.File(h5_path, "r")
        self.ids = list(self.file.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of bounds")

        id = self.ids[index]
        group = self.file[id]

        rand_pos = group["pos"][:] * 10
        nb_params = group["nb_params"][:]
        bonds = group["bonds"][:]
        bond_types = group["bond_types"][:]
        elements = group["elements"][:]

        edge_index = bonds.T
        edge_attr = bond_types

        nb_params = torch.tensor(nb_params, dtype=torch.float32)
        system = self.get_system(index)

        q_sq_mat, sig_mat, eps_mat = get_nb_matrices(nb_params, system)
        bond_index0, bond_index1, bond0, bond_k = get_bond_params(system)
        angle_index0, angle_index1, angle_index2, angle0, angle_k = get_angle_params(
            system
        )
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

        return Data(
            pos=torch.tensor(rand_pos, dtype=torch.float32),
            nb_params=nb_params,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edata=torch.tensor(edge_attr, dtype=torch.float32),
            elements=torch.tensor(elements, dtype=torch.long),
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

    def get_system(self, index):
        """Returns the openmm system for the given index."""
        systems_folder = os.path.join(self.folder, "systems")

        id = self.ids[index]
        system_file = os.path.join(systems_folder, f"{id}.xml")

        # load system into OpenMM
        with open(system_file, "r") as f:
            system = mm.XmlSerializer.deserialize(f.read())

        return system

    def get_rand_sdf(self, index):
        """Returns the (random) SDF file for the given index."""

        sdf_folder = os.path.join(self.folder, "structures")

        id = self.ids[index]
        sdf_file = os.path.join(sdf_folder, f"{id}.sdf")

        return sdf_file