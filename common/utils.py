from copy import deepcopy
import gzip
from io import BytesIO, TextIOWrapper
import io
import subprocess

import requests
from omegaconf import DictConfig, OmegaConf
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from functools import wraps
import inspect
from typing import Dict, Union
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Geometry import Point3D

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    print("Warning: torch and torch_geometric not found. Some functions may not work.")

try:
    from openmm import unit, app
    import openmm as mm
except ImportError:
    print("Warning: openmm not found. Some functions may not work.")

try:
    from openff.toolkit.topology import Molecule
    from openff.units.openmm import to_openmm
except ImportError:
    print("Warning: openff not found. Some functions may not work.")

# from https://github.com/choderalab/openmmtools/issues/522
ONE_4PI_EPS0 = 138.93545764438198

CONFIG = DictConfig({})


def load_config(filename, include_cmd_line):
    """Loads configuration from default.yml, the config filename,
    and command line arguments, in that order. Returns nothing; it
    loads everything into the CONFIG global variable."""
    cfg = OmegaConf.load("configs/default.yaml")

    # platform-specific stuff for benchmarking
    platform_config = "configs/platform.yaml"
    if os.path.exists(platform_config):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(platform_config))

    if filename is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(filename))

    if include_cmd_line:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    # try to come up with reasonable defaults
    if "cache_dir" not in cfg:
        cfg.cache_dir = "cache"

    for key in list(CONFIG.keys()):
        del CONFIG[key]

    CONFIG.update(cfg)


load_config(None, True)


def get_output_dir() -> str:
    """Overridable output directory."""
    if "OUTPUT_DIR" in os.environ:
        ret = os.environ["OUTPUT_DIR"]
    else:
        ret = "output"
    os.makedirs(ret, exist_ok=True)
    return ret


def get_cache_dir():
    ret = "cache"
    os.makedirs(ret, exist_ok=True)
    return ret


def pyg_to_rdkit(data):
    """Convert a PyG data object to an RDKit molecule. Assumes the
    atomic numbers are in data.elements and bond order is in data.edge_attr
    (bond order is a float)"""
    mol = Chem.RWMol()
    for i, elem in enumerate(data.elements):
        a = Chem.Atom(elem.item())
        if "formal_charges" in data:
            a.SetFormalCharge(data.formal_charges[i].item())
        mol.AddAtom(a)
    for i, (j, k) in enumerate(data.edge_index.t().tolist()):
        bond_order = data.edata[i].item()
        bond_type = {
            1.0: Chem.BondType.SINGLE,
            1.5: Chem.BondType.AROMATIC,
            2.0: Chem.BondType.DOUBLE,
            3.0: Chem.BondType.TRIPLE,
        }[bond_order]
        mol.AddBond(j, k, bond_type)

    # add conformer
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(data.pos):
        conf.SetAtomPosition(i, Point3D(*pos.tolist()))
    mol.AddConformer(conf)

    return Chem.Mol(mol)

def rdkit_to_pyg(mol):
    """Convert an RDKit molecule to a PyG data object."""
    elements = []
    edge_index = []
    edge_attr = []
    for atom in mol.GetAtoms():
        elements.append(atom.GetAtomicNum())
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_attr.append(
            {
                Chem.BondType.SINGLE: 1.0,
                Chem.BondType.DOUBLE: 2.0,
                Chem.BondType.TRIPLE: 3.0,
                Chem.BondType.AROMATIC: 1.5,
            }[bond.GetBondType()]
        )
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    elements = torch.tensor(elements, dtype=torch.long)
    return Data(elements=elements, edge_index=edge_index, edata=edge_attr)

def pyg_to_topology(data):
    """ Returns an OpenMM topology from a PyG data object """

    ret = app.Topology()
    chain = ret.addChain()
    res = ret.addResidue("UNK", chain)
    for i, elem in enumerate(data.elements):
        atom = ret.addAtom(f"atom_{i}", app.Element.getByAtomicNumber(elem.item()), res)
    # add bonds
    atoms = list(ret.atoms())
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        ret.addBond(atoms[i], atoms[j])

    return ret

def lj_E(nb_params, pos):
    """Return the Lennard-Jones energy of the given positions. nb_params
    is a tensor containing charge, sigma and epsilon for each atom."""

    pos = pos.view(-1, 1, 3) / 10  # convert to nm
    dists_sq = (pos - pos.permute(1, 0, 2)).pow(2).sum(2)
    dists_sq = dists_sq + torch.eye(
        dists_sq.shape[0], device=dists_sq.device
    )  # avoid division by zero

    sig = nb_params[:, 1].view(-1, 1)
    eps = nb_params[:, 2].view(-1, 1)

    sig_ij = (sig + sig.permute(1, 0)) / 2
    eps_ij = (eps * eps.permute(1, 0)).sqrt()

    r6 = (sig_ij / dists_sq).pow(3)
    r12 = r6.pow(2)

    E = 4 * eps_ij * (r12 - r6)
    E = E.triu(diagonal=1).sum()

    return E


def coul_E(nb_params, pos):
    """Return the Coulomb energy of the given positions. nb_params
    is a tensor containing charge, sigma and epsilon for each atom."""

    pos = pos.view(-1, 1, 3) / 10  # convert to nm
    dists_sq = (pos - pos.permute(1, 0, 2)).pow(2).sum(2)
    dists_sq = dists_sq + torch.eye(
        pos.shape[0], device=pos.device
    )  # avoid division by zero

    dists = torch.sqrt(dists_sq)

    q = nb_params[:, 0].view(-1, 1)
    q_ij = q * q.permute(1, 0)

    E = ONE_4PI_EPS0 * q_ij / dists

    E = E.triu(diagonal=1).sum()
    return E


def save_sdf(mols, path):
    """Save an RDKit molecule(s) to an SDF file."""

    if not isinstance(mols, list):
        mols = [mols]

    writer = Chem.SDWriter(path)
    # save all conformers
    for mol in mols:
        for cid in range(mol.GetNumConformers()):
            writer.write(mol, confId=cid)
    writer.close()

def is_same_mol(mol1, mol2):
    """ Check if two molecules are the same (same atoms in same order) """
    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        return False
    for a1, a2 in zip(mol1.GetAtoms(), mol2.GetAtoms()):
        if a1.GetAtomicNum() != a2.GetAtomicNum():
            return False
    return True

def load_sdf(path):
    """Load an SDF file into an RDKit molecule, or a list of RDKit molecules"""
    mols = []
    for conf_mol in Chem.SDMolSupplier(path, removeHs=False):
        if conf_mol is None:
            return None
        if len(mols) == 0:
            mols.append(conf_mol)
            continue
        mol = mols[-1]
        if is_same_mol(mol, conf_mol):
            mol.AddConformer(conf_mol.GetConformer(), assignId=True)
        else:
            mols.append(conf_mol)
    if len(mols) == 1:
        return mols[0]
    return mols

def with_units(
    arg_dict, output  # mapping from either kwarg name or position to unit
):  # desired output unit
    """Wrapper for functions that can optionally take arguments with units. Numpy doesn'try
    work very nicely with OpenMM units, so just use this to wrap any function that can take units.
    This will automatically convert input types to the unitless values, and convert the output to
    the desired output Quantity. If the input values are unitless, the output will be unitless as well.
    """

    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            use_units = False
            ba = inspect.signature(f).bind(*args, **kwargs)
            ba.apply_defaults()
            for i, name in enumerate(ba.arguments):
                if i in arg_dict or name in arg_dict:
                    if isinstance(ba.arguments[name], unit.Quantity):
                        units = arg_dict.get(i, arg_dict.get(name))
                        if not units.is_compatible(ba.arguments[name].unit):
                            raise ValueError(
                                f"Argument {name} has units {ba.arguments[name].unit}, but expected {units}"
                            )
                        ba.arguments[name] = ba.arguments[name] / units
                        use_units = True

            result = f(*ba.args, **ba.kwargs)
            if use_units:
                return result * output
            else:
                return result

        return wrapped

    return wrapper


def get_nonbonded_params(system):
    """Get the nonbonded parameters from an OpenMM system."""
    nb_params = []
    nb_force = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, mm.NonbondedForce):
            nb_force = force
            break

    for i in range(system.getNumParticles()):
        charge, sigma, eps = nb_force.getParticleParameters(i)
        nb_params.append(
            (
                charge.value_in_unit(unit.elementary_charge),
                sigma.value_in_unit(unit.nanometers),
                eps.value_in_unit(unit.kilojoules_per_mole),
            )
        )
    return np.array(nb_params, dtype=np.float32)


def get_nb_matrices(nb_params, system):
    """Get the full q_sq, sigma, and eps matrices including
    the system exceptions"""

    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            nonbonded_force = force
            break
    assert nonbonded_force is not None

    q, sig, eps = nb_params.T
    q = q.view(-1, 1)
    sig = sig.view(-1, 1)
    eps = eps.view(-1, 1)

    sig_mat = (sig + sig.permute(1, 0)) / 2
    eps_mat = (eps * eps.permute(1, 0)).sqrt()
    q_sq_mat = q * q.permute(1, 0)

    for i in range(nonbonded_force.getNumExceptions()):
        idx0, idx1, q_sq_ij, sig_ij, eps_ij = nonbonded_force.getExceptionParameters(i)

        q_sq_ij = q_sq_ij.value_in_unit(q_sq_ij.unit)
        sig_ij = sig_ij.value_in_unit(sig_ij.unit)
        eps_ij = eps_ij.value_in_unit(eps_ij.unit)

        sig_mat[idx0, idx1] = sig_ij
        sig_mat[idx1, idx0] = sig_ij

        eps_mat[idx0, idx1] = eps_ij
        eps_mat[idx1, idx0] = eps_ij

        q_sq_mat[idx0, idx1] = q_sq_ij
        q_sq_mat[idx1, idx0] = q_sq_ij

    return q_sq_mat, sig_mat, eps_mat


def get_nonbonded_U(positions, q_sq_mat, sig_mat, eps_mat):
    """Return the LJ + column energy given the matrices. Expects
    positions in angstroms."""

    pos = positions.view(-1, 1, 3) / 10  # angstrom to nm

    dists_sq = (pos - pos.permute(1, 0, 2)).pow(2).sum(2)
    dists_sq = dists_sq + torch.eye(
        dists_sq.shape[0], device=dists_sq.device
    )  # avoid division by zero
    dists = torch.sqrt(dists_sq)

    r6 = (sig_mat / dists).pow(6)
    r12 = r6.pow(2)

    E_lj = 4 * eps_mat * (r12 - r6)
    E_coul = ONE_4PI_EPS0 * q_sq_mat / dists

    E_mat = E_lj + E_coul
    E = E_mat.triu(diagonal=1).sum()

    return E


def get_torsion_params(system):
    """Return the PeriodicTorsionForce parameters for the given system."""

    torsion_force = None
    for force in system.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            torsion_force = force
            break

    index0 = []
    index1 = []
    index2 = []
    index3 = []
    periodicity = []
    phase = []
    k = []

    for i in range(torsion_force.getNumTorsions()):
        idx0, idx1, idx2, idx3, _periodicity, _phase, _k = (
            torsion_force.getTorsionParameters(i)
        )

        index0.append(idx0)
        index1.append(idx1)
        index2.append(idx2)
        index3.append(idx3)
        periodicity.append(_periodicity)
        phase.append(_phase.value_in_unit(unit.radian))
        k.append(_k.value_in_unit(unit.kilojoule_per_mole))

    index0 = torch.tensor(index0, dtype=torch.long)
    index1 = torch.tensor(index1, dtype=torch.long)
    index2 = torch.tensor(index2, dtype=torch.long)
    index3 = torch.tensor(index3, dtype=torch.long)
    periodicity = torch.tensor(periodicity, dtype=torch.float)
    phase = torch.tensor(phase, dtype=torch.float)
    k = torch.tensor(k, dtype=torch.float)

    return index0, index1, index2, index3, periodicity, phase, k


def get_torsion_U(pos, index0, index1, index2, index3, periodicity, phase, k):
    """Return the torsion energy for the given positions."""

    pos = pos / 10  # angstrom to nm

    pos0 = pos[index0]
    pos1 = pos[index1]
    pos2 = pos[index2]
    pos3 = pos[index3]

    b0 = pos0 - pos1
    b1 = pos2 - pos1
    b2 = pos3 - pos2

    n0 = torch.cross(b0, b1)
    n1 = torch.cross(b2, b1)

    n0_len = torch.sqrt((n0 * n0).sum(1))
    n1_len = torch.sqrt((n1 * n1).sum(1))

    n0_unit = n0 / n0_len.view(-1, 1)
    n1_unit = n1 / n1_len.view(-1, 1)

    m0 = torch.cross(n0_unit, b1)
    m1 = torch.cross(n1_unit, b1)

    m0_len = torch.sqrt((m0 * m0).sum(1))
    m1_len = torch.sqrt((m1 * m1).sum(1))

    m0_unit = m0 / m0_len.view(-1, 1)
    # m1_unit = m1 / m1_len.view(-1, 1)

    cos_phi = (n0_unit * n1_unit).sum(1)
    sin_phi = (m0_unit * n1_unit).sum(1)

    phi = torch.atan2(sin_phi, cos_phi)

    energy = k * (1 + torch.cos(periodicity * phi - phase))
    return energy.sum()


def get_angle_params(system):
    """Return the HarmonicAngleForce parameters for the given system."""

    angle_force = None
    for force in system.getForces():
        if isinstance(force, mm.HarmonicAngleForce):
            angle_force = force
            break

    index0 = []
    index1 = []
    index2 = []
    theta0 = []
    k = []

    for i in range(angle_force.getNumAngles()):
        idx0, idx1, idx2, _theta0, _k = angle_force.getAngleParameters(i)

        index0.append(idx0)
        index1.append(idx1)
        index2.append(idx2)
        theta0.append(_theta0.value_in_unit(unit.radian))
        k.append(_k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2))

    index0 = torch.tensor(index0, dtype=torch.long)
    index1 = torch.tensor(index1, dtype=torch.long)
    index2 = torch.tensor(index2, dtype=torch.long)
    k = torch.tensor(k, dtype=torch.float)
    theta0 = torch.tensor(theta0, dtype=torch.float)

    return index0, index1, index2, theta0, k


def get_angle_U(pos, index0, index1, index2, theta0, k):
    """Return the angle energy for the given positions."""

    pos = pos / 10  # angstrom to nm

    pos0 = pos[index0]
    pos1 = pos[index1]
    pos2 = pos[index2]

    b0 = pos0 - pos1
    b1 = pos2 - pos1

    b0_len = torch.sqrt((b0 * b0).sum(1))
    b1_len = torch.sqrt((b1 * b1).sum(1))

    b0_unit = b0 / b0_len.view(-1, 1)
    b1_unit = b1 / b1_len.view(-1, 1)

    cos_theta = (b0_unit * b1_unit).sum(1)
    theta = torch.acos(cos_theta)

    energy = 0.5 * k * (theta - theta0) ** 2
    return energy.sum()


def get_bond_params(system):
    """Return the HarmonicBondForce parameters for the given system."""

    bond_force = None
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            bond_force = force
            break

    index0 = []
    index1 = []
    r0 = []
    k = []

    for i in range(bond_force.getNumBonds()):
        idx0, idx1, _r0, _k = bond_force.getBondParameters(i)

        index0.append(idx0)
        index1.append(idx1)
        r0.append(_r0.value_in_unit(unit.nanometer))
        k.append(_k.value_in_unit(unit.kilojoule_per_mole / unit.nanometers**2))

    index0 = torch.tensor(index0, dtype=torch.long)
    index1 = torch.tensor(index1, dtype=torch.long)
    r0 = torch.tensor(r0, dtype=torch.float)
    k = torch.tensor(k, dtype=torch.float)

    return index0, index1, r0, k


def get_bond_U(pos, index0, index1, r0, k):
    """Return the bond energy for the given positions."""

    pos = pos / 10  # angstrom to nm

    pos0 = pos[index0]
    pos1 = pos[index1]

    b = pos0 - pos1

    b_len = torch.sqrt((b * b).sum(1))

    energy = 0.5 * k * (b_len - r0) ** 2
    return energy.sum()


def get_constraints(system):
    """Get distance constraints from the Openmm system"""

    index0 = []
    index1 = []
    r0 = []

    for i in range(system.getNumConstraints()):
        idx0, idx1, _r0 = system.getConstraintParameters(i)

        index0.append(idx0)
        index1.append(idx1)
        r0.append(_r0.value_in_unit(unit.nanometer))

    index0 = torch.tensor(index0, dtype=torch.long)
    index1 = torch.tensor(index1, dtype=torch.long)
    r0 = torch.tensor(r0, dtype=torch.float)

    return index0, index1, r0


def save_smi(fname, smiles):
    with open(fname, "w") as f:
        for smi in smiles:
            f.write(smi + "\n")


def load_smi(fname):
    with open(fname, "r") as f:
        return f.read().splitlines()


pdbfixer_substitutions = {
    "2AS": "ASP",
    "3AH": "HIS",
    "5HP": "GLU",
    "5OW": "LYS",
    "ACL": "ARG",
    "AGM": "ARG",
    "AIB": "ALA",
    "ALM": "ALA",
    "ALO": "THR",
    "ALY": "LYS",
    "ARM": "ARG",
    "ASA": "ASP",
    "ASB": "ASP",
    "ASK": "ASP",
    "ASL": "ASP",
    "ASQ": "ASP",
    "AYA": "ALA",
    "BCS": "CYS",
    "BHD": "ASP",
    "BMT": "THR",
    "BNN": "ALA",
    "BUC": "CYS",
    "BUG": "LEU",
    "C5C": "CYS",
    "C6C": "CYS",
    "CAS": "CYS",
    "CCS": "CYS",
    "CEA": "CYS",
    "CGU": "GLU",
    "CHG": "ALA",
    "CLE": "LEU",
    "CME": "CYS",
    "CSD": "ALA",
    "CSO": "CYS",
    "CSP": "CYS",
    "CSS": "CYS",
    "CSW": "CYS",
    "CSX": "CYS",
    "CXM": "MET",
    "CY1": "CYS",
    "CY3": "CYS",
    "CYG": "CYS",
    "CYM": "CYS",
    "CYQ": "CYS",
    "DAH": "PHE",
    "DAL": "ALA",
    "DAR": "ARG",
    "DAS": "ASP",
    "DCY": "CYS",
    "DGL": "GLU",
    "DGN": "GLN",
    "DHA": "ALA",
    "DHI": "HIS",
    "DIL": "ILE",
    "DIV": "VAL",
    "DLE": "LEU",
    "DLY": "LYS",
    "DNP": "ALA",
    "DPN": "PHE",
    "DPR": "PRO",
    "DSN": "SER",
    "DSP": "ASP",
    "DTH": "THR",
    "DTR": "TRP",
    "DTY": "TYR",
    "DVA": "VAL",
    "EFC": "CYS",
    "FLA": "ALA",
    "FME": "MET",
    "GGL": "GLU",
    "GL3": "GLY",
    "GLZ": "GLY",
    "GMA": "GLU",
    "GSC": "GLY",
    "HAC": "ALA",
    "HAR": "ARG",
    "HIC": "HIS",
    "HIP": "HIS",
    "HMR": "ARG",
    "HPQ": "PHE",
    "HTR": "TRP",
    "HYP": "PRO",
    "IAS": "ASP",
    "IIL": "ILE",
    "IYR": "TYR",
    "KCX": "LYS",
    "LLP": "LYS",
    "LLY": "LYS",
    "LTR": "TRP",
    "LYM": "LYS",
    "LYZ": "LYS",
    "MAA": "ALA",
    "MEN": "ASN",
    "MHS": "HIS",
    "MIS": "SER",
    "MK8": "LEU",
    "MLE": "LEU",
    "MPQ": "GLY",
    "MSA": "GLY",
    "MSE": "MET",
    "MVA": "VAL",
    "NEM": "HIS",
    "NEP": "HIS",
    "NLE": "LEU",
    "NLN": "LEU",
    "NLP": "LEU",
    "NMC": "GLY",
    "OAS": "SER",
    "OCS": "CYS",
    "OMT": "MET",
    "PAQ": "TYR",
    "PCA": "GLU",
    "PEC": "CYS",
    "PHI": "PHE",
    "PHL": "PHE",
    "PR3": "CYS",
    "PRR": "ALA",
    "PTR": "TYR",
    "PYX": "CYS",
    "SAC": "SER",
    "SAR": "GLY",
    "SCH": "CYS",
    "SCS": "CYS",
    "SCY": "CYS",
    "SEL": "SER",
    "SEP": "SER",
    "SET": "SER",
    "SHC": "CYS",
    "SHR": "LYS",
    "SMC": "CYS",
    "SOC": "CYS",
    "STY": "TYR",
    "SVA": "SER",
    "TIH": "ALA",
    "TPL": "TRP",
    "TPO": "THR",
    "TPQ": "ALA",
    "TRG": "LYS",
    "TRO": "TRP",
    "TYB": "TYR",
    "TYI": "TYR",
    "TYQ": "TYR",
    "TYS": "TYR",
    "TYY": "TYR",
}

amino_acid_dict = {
    # Standard amino acids
    "ALA": "A",  # Alanine
    "ARG": "R",  # Arginine
    "ASN": "N",  # Asparagine
    "ASP": "D",  # Aspartic Acid
    "CYS": "C",  # Cysteine
    "GLU": "E",  # Glutamic Acid
    "GLN": "Q",  # Glutamine
    "GLY": "G",  # Glycine
    "HIS": "H",  # Histidine
    "ILE": "I",  # Isoleucine
    "LEU": "L",  # Leucine
    "LYS": "K",  # Lysine
    "MET": "M",  # Methionine
    "PHE": "F",  # Phenylalanine
    "PRO": "P",  # Proline
    "SER": "S",  # Serine
    "THR": "T",  # Threonine
    "TRP": "W",  # Tryptophan
    "TYR": "Y",  # Tyrosine
    "VAL": "V",  # Valine
    # Protonation states of amino acids
    "ASH": "D",  # Protonated Aspartic Acid
    "GLH": "E",  # Protonated Glutamic Acid
    "HID": "H",  # Histidine (delta nitrogen protonated)
    "HIE": "H",  # Histidine (epsilon nitrogen protonated)
    "HIP": "H",  # Fully protonated Histidine (both nitrogens)
    "LYN": "K",  # Deprotonated Lysine
    "CYX": "C",  # Cystine (disulfide bond)
    "CYM": "C",  # Deprotonated Cysteine
    # nonstandard/modified amino acids
    "MSE": "M",  # Selenomethionine
    "CSO": "C",  # S-hydroxycysteine
    "CME": "C",  # S,S-(2-hydroxyethyl)thiocysteine
    "CCS": "C",  # Carboxymethylcysteine
    "SCH": "C",  # S-methyl-thio-cysteine
    "YCM": "C",  # S-(2-AMINO-2-OXOETHYL)-L-CYSTEINE
    "YOF": "Y",  # 3-fluorotyrosine
    "M3L": "K",  # N-trimethyllysine
    "MLY": "K",  # N-dimethyllysine
    "MLZ": "K",  # N-methyllysine
    "ALY": "K",  # N-acetyllysine
    "NMM": "R",  # N-methylarginine
    "PTR": "Y",  # Phosphotyrosine
    "SEP": "S",  # Phosphoserine
    "TPO": "T",  # Phosphothreonine
    "PCA": "E",  # Pyroglutamic acid
    "NEP": "H",  # N1-phosphohistidine
    "AME": "M",  # N-acetylmethionine
    "FTR": "W",  # 5-fluorotryptophan
    "CMT": "C", # O-methylcysteine
    "6CW": "W", # 6-chlorotryptophan
}

ignore_residues = {
    "HOH",
    "WAT",
    "TIP3",
    "TIP4",
    "Na+",
    "Cl-",
    "K+",
    "DOD",
    "ACE",
    "NH2",
    "SIN",
    # these come from alphafold 3 crystallization aid list
    "SO4",
    "GOL",
    "EDO",
    "PO4",
    "ACT",
    "PEG",
    "DMS",
    "TRS",
    "PGE",
    "PG4",
    "FMT",
    "EPE",
    "MPD",
    "MES",
    "CD",
    "IOD",
}


def load_protein(filename):
    """Load a cif or pdb file with openmm"""

    # we already opened it
    if not isinstance(filename, str):
        return filename

    if filename.endswith(".cif"):
        return app.PDBxFile(open(filename, "r"))
    elif filename.endswith(".pdb"):
        return app.PDBFile(filename)
    else:
        raise ValueError("File format not supported")


def get_sequence(pdb, ignore_unknown=False) -> str:
    """Get the amino acid sequence from a PDB or CIF file. Also
    returns mapping from sequence index to residue index."""

    aas = []
    res_indices = []
    for residue in pdb.topology.residues():
        code = residue.name
        if code in pdbfixer_substitutions:
            code = pdbfixer_substitutions[code]
        if code in ignore_residues:
            continue
        if code not in amino_acid_dict:
            if ignore_unknown:
                continue
            else:
                raise ValueError(f"Unknown amino acid {code}")
        aas.append(amino_acid_dict[code])
        res_indices.append(residue.index)

    seq = "".join(aas)
    return seq, res_indices


def protonate_mol(mol: Chem.Mol, pH: float = 7):
    """Use openbabel to protonate an RDKit molecule."""

    # first save as sdf
    w = Chem.SDWriter(f"{get_output_dir()}/temp.sdf")
    w.write(mol)
    w.close()

    subprocess.run(
        f"obabel {get_output_dir()}/temp.sdf -O {get_output_dir()}/temp.sdf -p {pH}",
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # read back in
    suppl = Chem.SDMolSupplier(f"{get_output_dir()}/temp.sdf", removeHs=False)
    mol = suppl[0]

    return mol


def protonate_smiles(smiles):
    """Use obabel to protonate the smiles at pH 7"""
    fname = f"{get_output_dir()}/protonated.smi"
    subprocess.run(
        f'obabel -:"{smiles}" -O {fname} -p 7',
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with open(fname) as f:
        protonated_smiles = f.read().strip()
    return protonated_smiles


def remove_salt_from_smiles(smiles):
    """Use Chem.SaltRemover to remove salts from the smiles"""

    remover = SaltRemover()
    mol = Chem.MolFromSmiles(smiles)
    mol = remover.StripMol(mol)
    return Chem.MolToSmiles(mol)


def infer_bonding(elements, pos):
    """Infer the bonding from the elements and positions of the atoms"""

    # determine bonding based on distance
    # inspired by https://github.com/jensengroup/xyz2mol/blob/b9929e6e935d2d87d0092f3b1f1136a1fc9d93fe/xyz2mol.py#L600

    cov_factor = 1.3
    pt = Chem.GetPeriodicTable()
    cov_radii = torch.tensor(
        [pt.GetRcovalent(elem.item()) for elem in elements],
        device=elements.device,
    )
    cov_cutoff_mat = (cov_radii.reshape(-1, 1) + cov_radii.reshape(1, -1)) * cov_factor

    dist_mat = torch.cdist(pos, pos)
    adj_mat = dist_mat < cov_cutoff_mat

    # only take upper triangle
    adj_mat = torch.triu(adj_mat, diagonal=1)
    edge_index = torch.stack(torch.nonzero(adj_mat, as_tuple=True))

    return edge_index


def get_fp(mol, radius=4, bits=2048):
    """Get the Morgan fingerprint of a molecule."""
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, useChirality=True, radius=radius, nBits=bits
    )
    return np.array(fp, dtype=bool)


def load_env(env_fname):
    """Load a .env file into the environment."""
    with open(env_fname) as f:
        for line in f.readlines():
            key, value = line.strip().split("=")
            os.environ[key] = value


def unload_env(env_fname):
    """Unload a .env file from the environment."""
    with open(env_fname) as f:
        for line in f.readlines():
            key, value = line.strip().split("=")
            del os.environ[key]


def combine_modellers(*modellers):
    """Combine multiple openmm modellers into a single modeller."""
    modeller = app.Modeller(modellers[0].topology, modellers[0].positions)
    for m in modellers[1:]:
        modeller.add(m.topology, m.positions)
    pos = np.array(modeller.positions.value_in_unit(unit.nanometer))*unit.nanometer
    modeller.positions = pos
    return modeller


def rdkit_to_modeller(mol):
    """Convert an RDKit molecule to an OpenMM modeller."""

    ff_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)

    top = ff_mol.to_topology().to_openmm()
    pos = to_openmm(ff_mol.conformers[0])
    return app.Modeller(top, pos)


def get_CA_indices(topology):
    """Get indices of alpha carbons in the openmm topology."""

    indices = []
    for atom in topology.atoms():
        if atom.name == "CA":
            indices.append(atom.index)
    return indices

def align_chains(rec, box_vectors):
    """ Brings the COM of each chain closest to chain 0 """

    # get indices of each chain
    chain_indices = []
    for chain in rec.topology.chains():
        chain_indices.append(np.array([atom.index for atom in chain.atoms()]))

    # find the periodic image that brings all the chains closest to chain 0 (min distance)
    chain0_pos = rec.positions.value_in_unit(unit.nanometers)[chain_indices[0]]

    bv = box_vectors.value_in_unit(unit.nanometers)
    for indices in chain_indices[1:]:
        chain_pos = rec.positions.value_in_unit(unit.nanometers)[indices]

        cur_min = (np.inf, None)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    mul = np.tile(np.array([i, j, k]), 3).reshape(3,3).T
                    shift = (bv * mul).sum(axis=0)
                    chain_pos_shifted = chain_pos + shift
                    # compute min distance
                    # dists = np.linalg.norm(chain0_pos[:, None] - chain_pos_shifted, axis=2)
                    # dist = dists.min()

                    # compute COM distance
                    dist = np.linalg.norm(chain0_pos.mean(axis=0) - chain_pos_shifted.mean(axis=0))

                    if dist < cur_min[0]:
                        cur_min = (dist, chain_pos_shifted)

        rec.positions[indices] = cur_min[1] * unit.nanometers

def modeller_with_state(modeller, state):
    """Create a new modeller with the given OpenMMTools sampler state"""
    new_modeller = app.Modeller(modeller.topology, state.positions)
    new_modeller.topology.setPeriodicBoxVectors(state.box_vectors)
    if state.box_vectors is not None:
        align_chains(new_modeller, state.box_vectors)
    return new_modeller


def get_residue_atom_indices(topology, res_indices):
    """Returns a list of atom indices for each residue"""
    ret = []
    for residue in topology.residues():
        if residue.index in res_indices:
            for atom in residue.atoms():
                ret.append(atom.index)
    return np.array(ret)

def get_Ca_indices(topology, atom_indices):
    """ Returns all the CA indices that are in the given atom indices """
    return np.array([
        i for i in atom_indices if i in get_CA_indices(topology)
    ], dtype=int)

def save_modeller_pdb(mol, path):
    app.PDBFile.writeFile(mol.topology, mol.positions, open(path, "w"))


def save_system_xml(system, path):
    with open(path, "w") as f:
        f.write(mm.XmlSerializer.serialize(system))


def load_modeller_pdb(path):
    pdb = app.PDBFile(path)
    pos = np.array(pdb.positions.value_in_unit(unit.nanometer))*unit.nanometer
    return app.Modeller(pdb.topology, pos)


def load_system_xml(path):
    with open(path, "r") as f:
        return mm.XmlSerializer.deserialize(f.read())


def add_coords_to_mol(mol, coords):
    """Adds conformer to RDkit mol. Assumes coords are a simple numpy/pytorch array on the CPU"""
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, coord in enumerate(coords):
        conformer.SetAtomPosition(
            i, Point3D(float(coord[0]), float(coord[1]), float(coord[2]))
        )

    mol.AddConformer(conformer)

def save_pointcloud_pdb(to_save, path):
    """ Save a PDB file with carbon atoms in this point cloud (in nm) """

    topology = app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("UNK", chain)
    for pt in to_save:
        topology.addAtom("C", app.Element.getBySymbol("C"), res)

    if not isinstance(to_save, unit.Quantity):
        to_save = to_save * unit.nanometers

    modeller = app.Modeller(topology, to_save)
    save_modeller_pdb(modeller, path)

class FatalError(Exception):
    """Fatal error that should cause the conductor to stop an instance of this program """


def compress_mol(mols):
    """ Writes all the molecule's conformers to a gzipped compressed buffer """
    
    if mols is None:
        return None
    
    buffer = TextIOWrapper(BytesIO(), encoding="utf-8")
    writer = Chem.SDWriter(buffer)

    if not isinstance(mols, list):
        mols = [mols]

    for mol in mols:
        for conf_id in range(mol.GetNumConformers()):
            writer.write(mol, conf_id)
    writer.close()
    buffer.seek(0)
    return gzip.compress(buffer.buffer.read())

def decompress_mol(mol_bytes):
    """ Reads a gzipped compressed molecule from a buffer """

    if mol_bytes is None:
        return None

    try:
        buffer = BytesIO(gzip.decompress(mol_bytes))
        suppl = Chem.ForwardSDMolSupplier(buffer, removeHs=False)
        mols = []
        for conf_mol in suppl:
            if conf_mol is None:
                return None
            if len(mols) == 0:
                mols.append(conf_mol)
                continue
            mol = mols[-1]
            if is_same_mol(mol, conf_mol):
                mol.AddConformer(conf_mol.GetConformer(), assignId=True)
            else:
                mols.append(conf_mol)
        if len(mols) == 1:
            return mols[0]
        return mols
    except gzip.BadGzipFile:
        print("Warning: Failed to decompress molecule. Returning None.")
        return None

def rmsd_noh(mol1, mol2, mol1_conf_idx=0, mol2_conf_id=0):
    """ Removes all hydrogens and calculates RMSD """
    mol1 = Chem.RemoveAllHs(mol1)
    mol2 = Chem.RemoveAllHs(mol2)
    return AllChem.CalcRMS(mol1, mol2, mol1_conf_idx, mol2_conf_id)

def all_rmsds_noh(pred, ref):
    """ Returns a np array of the rmsds between the reference and 
    each of the predicted poses """
    ref = Chem.RemoveAllHs(ref)
    pred = Chem.RemoveAllHs(pred)
    ret = []
    for conf_id in range(pred.GetNumConformers()):
        rmsd = AllChem.CalcRMS(pred, ref, conf_id)
        ret.append(rmsd)
    return np.array(ret)

def align_mols_noh(mol1, mol2, mol1_conf_idx=0, mol2_conf_id=0):
    """ Removes all hydrogens and aligns the molecules. Returns mol1 aligned to mol2 """
    mol1_noh = Chem.RemoveAllHs(mol1)
    mol2_noh = Chem.RemoveAllHs(mol2)
    rmsd, T, atommap = AllChem.GetBestAlignmentTransform(mol1_noh, mol2_noh, mol1_conf_idx, mol2_conf_id)
    mol1_coords = mol1.GetConformer().GetPositions()
    # T is a 4x4 matrix -- extract the rotation and translation
    R = T[:3, :3]
    t = T[:3, 3]
    mol1_coords = np.dot(mol1_coords, R.T) + t
    mol1 = deepcopy(mol1)
    add_coords_to_mol(mol1, mol1_coords)
    return mol1

def add_seq_to_pdb(in_file, pdb_id, out_file):
    """ Adds SEQRES to a PDB based on the PDB ID """

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = requests.get(url)
    og_lines = io.StringIO(r.content.decode("utf-8")).readlines()

    seqres_lines = [ line for line in og_lines if line.startswith("SEQRES") ]
    
    pdb_lines = seqres_lines + open(in_file).readlines()
    with open(out_file, "w") as f:
        f.write("".join(pdb_lines))