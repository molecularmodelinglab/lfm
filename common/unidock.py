import json
import os
import subprocess
from traceback import print_exc
from rdkit import Chem
from rdkit.Chem import AllChem

from common.utils import (
    get_output_dir,
    protonate_smiles,
    remove_salt_from_smiles,
    save_modeller_pdb,
)

def save_rec_pdbqt(struct, filename):
    """ Save an openmm Modeller structure to a pdbqt file """
    out_pdb = filename.replace(".pdbqt", ".pdb")
    save_modeller_pdb(struct, out_pdb)
    subprocess.run(f"obabel {out_pdb} -O {filename} -xr", shell=True, check=True)

def save_lig_pdbqt(mol, filename):
    """Save an RDKit molecule to a PDBQT file"""

    # first save as sdf
    sdf_fname = filename.replace(".pdbqt", ".sdf")
    w = Chem.SDWriter(sdf_fname)
    w.write(mol)
    w.close()

    subprocess.run(
        f"mk_prepare_ligand.py -i {sdf_fname} -o {filename} --rigid_macrocycles",
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def get_all_lig_pdbqts(smi_list):
    """Returns a pdbqt for each molecule in the list after protonating and
    minimizing them. Returns a list of the pdbqt filenames """

    ret = []
    for i, smi in enumerate(smi_list):
        try:
            smi = remove_salt_from_smiles(smi)
            prot_smi = protonate_smiles(smi)
            mol = Chem.MolFromSmiles(prot_smi)
            mol = Chem.AddHs(mol)

            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)

            lig_pdbqt = f"{get_output_dir()}/lig_{i}.pdbqt"
            save_lig_pdbqt(mol, lig_pdbqt)
            ret.append(lig_pdbqt)
        except KeyboardInterrupt:
            raise
        except:
            print_exc()
            ret.append(None)

    return ret

def run_unidock_raw(
    rec_pdbqt,
    lig_pdbqts,
    out_dir,
    box_center,
    box_size,
    **kwargs,
):
    """Run unidock on a batch of ligands. Requires everything to be in pdbqt format.
    Returns a list of RDKit molecules with the docked conformations"""

    # first remove everything in the output directory
    subprocess.run(f"rm -rf {out_dir}/*", shell=True, check=True)

    box_center_arg = "--center_x {:.2f} --center_y {:.2f} --center_z {:.2f}".format(
        *box_center
    )
    box_size_arg = "--size_x {:.2f} --size_y {:.2f} --size_z {:.2f}".format(*box_size)

    batch_arg = " ".join([pdbqt for pdbqt in lig_pdbqts if pdbqt is not None])

    cmd = f"unidock --receptor {rec_pdbqt} --gpu_batch {batch_arg} --dir {out_dir} {box_center_arg} {box_size_arg}"
    for k, v in kwargs.items():
        cmd += f" --{k} {v}"

    print(f"Running {cmd}")
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

    # now load all the results
    mols = []
    for pdbqt in lig_pdbqts:
        try:
            if pdbqt is None:
                mols.append(None)
                continue

            out_pdbqt = (
                out_dir + "/" + pdbqt.split("/")[-1].replace(".pdbqt", "_out.pdbqt")
            )
            out_sdf = out_pdbqt.replace(".pdbqt", ".sdf")
            subprocess.run(
                f"mk_export.py {out_pdbqt} -o {out_sdf}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            mol = None
            for conf_mol in Chem.SDMolSupplier(out_sdf, removeHs=False):
                if mol is None:
                    mol = conf_mol
                else:
                    mol.AddConformer(conf_mol.GetConformer(), assignId=True)
            mols.append(mol)
        except KeyboardInterrupt:
            raise
        except:
            print_exc()
            mols.append(None)

    return mols

def get_docking_box(rec, poc_indices):
    from openmm import unit
    all_poc_pos = rec.positions[poc_indices].value_in_unit(unit.angstroms)
    # box_center = all_poc_pos.mean(axis=0)
    box_center = 0.5*(all_poc_pos.max(axis=0) + all_poc_pos.min(axis=0))
    box_size = all_poc_pos.max(axis=0) - all_poc_pos.min(axis=0)
    return box_size, box_center

def run_unidock(
    rec,
    ligs,
    poc_indices=None,
    box_center=None,
    box_size=None,
    **kwargs
):
    """ Saves everything to pdbqt and runs unidock. Can automatically
    determine box sized based on the pocket indices if we want.
    This will also automatically protonate and minimize the ligands.
    Returns a list of RDKit molecules with the docked conformations """

    if poc_indices is not None:
        assert box_center is None and box_size is None
        box_size, box_center = get_docking_box(rec, poc_indices)
    else:
        assert box_center is not None and box_size is not None

    rec_pdbqt = f"{get_output_dir()}/rec.pdbqt"
    save_rec_pdbqt(rec, rec_pdbqt)

    # convert to smiles if needed
    if isinstance(ligs[0], Chem.Mol):
        ligs = [Chem.MolToSmiles(lig) for lig in ligs]

    lig_pdbqts = get_all_lig_pdbqts(ligs)
    
    out_dir = get_output_dir() + "/unidock_out"
    os.makedirs(out_dir, exist_ok=True)

    return run_unidock_raw(
        rec_pdbqt,
        lig_pdbqts,
        out_dir,
        box_center,
        box_size,
        **kwargs,
    )

def get_unidock_score(mol):
    """Returns the unidock (vina) score of a molecule"""
    if mol is None:
        return None
    props = json.loads(mol.GetPropsAsDict()["meeko"])
    return props["free_energy"]