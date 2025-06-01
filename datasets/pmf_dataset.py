from typing import NamedTuple, Optional
from common.db import get_db_gs_path
from common.gs import GS_FS, download_gs_file
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from common.utils import CONFIG, infer_bonding, pyg_to_topology
from torch_geometric.data import Data
from torch import Tensor
import numpy as np
import pandas as pd
from openmm import unit
import roma
from openmm.app.internal.customgbforces import GBSAGBn2Force

T = 300 * unit.kelvin
kT = (unit.BOLTZMANN_CONSTANT_kB * T *unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
beta = 1/kT

class PMFData(NamedTuple):
    formal_charges: Tensor
    elements: Tensor
    nb_params: Tensor
    edge_index: Tensor
    edata: Tensor
    forces: Optional[Tensor] = None
    mean_hessian: Optional[Tensor] = None
    force_cov: Optional[Tensor] = None

    # instead of storing the mean hessian and force cov
    # we store products of them with a random vector
    rand_vec: Optional[Tensor] = None
    mean_hessian_vp: Optional[Tensor] = None
    force_cov_vp: Optional[Tensor] = None

    true_pos: Optional[Tensor] = None
    pos: Optional[Tensor] = None
    batch: Optional[Tensor] = None
    pK: Optional[float] = None
    q_sq_mat: Optional[Tensor] = None
    sig_mat: Optional[Tensor] = None
    eps_mat: Optional[Tensor] = None
    bond_index0: Optional[Tensor] = None
    bond_index1: Optional[Tensor] = None
    bond0: Optional[Tensor] = None
    bond_k: Optional[Tensor] = None
    angle_index0: Optional[Tensor] = None
    angle_index1: Optional[Tensor] = None
    angle_index2: Optional[Tensor] = None
    angle0: Optional[Tensor] = None
    angle_k: Optional[Tensor] = None
    tor_index0: Optional[Tensor] = None
    tor_index1: Optional[Tensor] = None
    tor_index2: Optional[Tensor] = None
    tor_index3: Optional[Tensor] = None
    tor_periodicity: Optional[Tensor] = None
    tor_phase: Optional[Tensor] = None
    tor_k: Optional[Tensor] = None
    const_index0: Optional[Tensor] = None
    const_index1: Optional[Tensor] = None
    const_r0: Optional[Tensor] = None
    ptr: Optional[Tensor] = None

    atom_features: Optional[Tensor] = None
    solvent_model_tensor: Optional[Tensor] = None
    solvent_dielectric: Optional[Tensor] = None

    def from_pyg_data(data):
        # todo: change. this is horrible
        # this is to support the delta GB models which require
        # hackily modifying the PMFData object
        if not CONFIG.train.jit:
            return data
        return PMFData(**data.to_namedtuple()._asdict())

def add_gb_params(data):
    """ Adds GB params so that we can use this data
    with the Riniker lab models"""
    top = pyg_to_topology(data)
    q = data.nb_params[:,0].numpy()

    force = GBSAGBn2Force(cutoff=None, SA=None, soluteDielectric=1)
    gbn2_parameters = np.empty((top.getNumAtoms(), 7))
    gbn2_parameters[:, 0] = q  # Charges
    gbn2_parameters[:, 1:6] = force.getStandardParameters(top)
    radii = gbn2_parameters[:, 1]
    uniqueRadii = list(sorted(set(radii)))
    radiusToIndex = {r: i for i, r in enumerate(uniqueRadii)}
    gbn2_parameters[:, 6] = [
        radiusToIndex[r] for r in gbn2_parameters[:, 1]
    ]
    offset = 0.0195141
    gbn2_parameters[:, 1] = gbn2_parameters[:, 1] - offset
    gbn2_parameters[:, 2] = gbn2_parameters[:, 2] * gbn2_parameters[:, 1]

    data["atom_features"] = torch.tensor(gbn2_parameters, dtype=torch.float32)
    solvent_model = 0
    solvent_dielectric = 78.5

    data["solvent_model_tensor"] = torch.tensor(solvent_model, dtype=torch.int64).repeat(len(data.pos))
    data["solvent_dielectric"] = torch.tensor(solvent_dielectric, dtype=torch.float).repeat(
        len(data.pos)
    )

class PMFDataset(Dataset):
    """ PyTorch dataset for the PMF dataset for a target specified in CONFIG. """

    def __init__(self, dataset_id, split, use_metrics=True):
        """ Split is either 'train', 'val', or 'test'."""

        gs_folder = get_db_gs_path("pmf_datasets", dataset_id)
        df_path = download_gs_file(f"{gs_folder}/{split}.csv")
        
        # don't do this when computing metrics
        if use_metrics:
            metrics_path = os.path.join(gs_folder, "metrics.npz")
            self.metrics = dict(np.load(GS_FS.open(metrics_path, "rb")))

        self.df = pd.read_csv(df_path)
        if CONFIG.dataset.max_length is not None:
            self.df = self.df.iloc[:CONFIG.dataset.max_length]

        h5_path = download_gs_file(f"{gs_folder}/{split}.h5")
        try:
            self.file = h5py.File(h5_path, "r")
        except OSError:
            # try again, it got corrupted
            os.remove(h5_path)
            h5_path = download_gs_file(f"{gs_folder}/{split}.h5")
            self.file = h5py.File(h5_path, "r")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of bounds")

        id = self.df.iloc[index]["id"]
        group = self.file[id]

        positions = torch.tensor(group["positions"][:]*10, dtype=torch.float32) # Convert from nm to Angstrom
        elements = torch.tensor(group["elements"][:], dtype=torch.long)
        
        formal_charges = group["formal_charges"][:]
        nb_params = group["nb_params"][:]
        forces = group["forces"][:]/10 # Convert from kJ/mol/nm to kJ/mol/Angstrom

        # if we have an entire trajectory, sample a random frame
        if len(positions.shape) == 3:
            assert len(forces.shape) == 3
            index = np.random.randint(positions.shape[0])
            positions = positions[index]
            forces = forces[index]

        # if CONFIG.loss.nce_weight > 0.0:
        #     restraint_k = group["lig_restraint_k"][:] / 100 # Convert from kJ/mol/nm^2 to kJ/mol/Angstrom^2
        #     restraint_com = group["lig_restraint_com"][:]*10 # Convert from nm to Angstrom
            
        #     restraint_com = torch.tensor(restraint_com, dtype=torch.float32)
        #     restraint_k = torch.tensor(restraint_k, dtype=torch.float32)

        #     # sample a translation from the boltzmann distribution of this harmonic restraint
        #     sigma = torch.sqrt(kT / (2 * restraint_k))
        #     t = torch.randn_like(restraint_com) * sigma

        #     # now sample a random rotation
        #     R = roma.utils.ranom_rotmat()

        edge_index = group["bonds"][:].T
        edata = group["bond_types"][:]

        data =  Data(
            pos=positions,
            elements=elements,
            nb_params=torch.tensor(nb_params, dtype=torch.float32),
            formal_charges=torch.tensor(formal_charges, dtype=torch.long),
            forces=torch.tensor(forces, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edata=torch.tensor(edata, dtype=torch.float32),
            # mean_hessian=torch.tensor(mean_hessians, dtype=torch.float32),
            # force_cov=torch.tensor(force_cov, dtype=torch.float32),
        )

        if CONFIG.loss.get("hessian_weight", 0.0) > 0.0:
            # smh multiple names for this
            try:
                hess_raw = group["mean_hessians"][:]
            except KeyError:
                hess_raw = group["hessians"][:]

            mean_hessians = hess_raw/100 # Convert from kJ/mol/nm^2 to kJ/mol/Angstrom^2
            force_cov = group["force_cov"][:]/100

            # uniform vector from -1 to 1
            rand_vec = torch.rand(mean_hessians.shape[0]) * 2 - 1
            mean_hessian_vp = rand_vec @ mean_hessians
            force_cov_vp = rand_vec @ force_cov

            data["rand_vec"] = rand_vec
            data["mean_hessian_vp"] = torch.tensor(mean_hessian_vp, dtype=torch.float32),
            data["force_cov_vp"] = torch.tensor(force_cov_vp, dtype=torch.float32),

        if CONFIG.model.get("delta_gb", False):
            add_gb_params(data)

        return data