import json
import os
import subprocess
from traceback import print_exc
from common.gs import download_gs_file
from rdkit import Chem
from rdkit.Chem import AllChem

from common.utils import (
    CONFIG,
    get_output_dir,
    load_sdf,
    protonate_smiles,
    remove_salt_from_smiles,
    save_modeller_pdb,
    save_sdf,
)

def prepare_ligand(smi):
    """ Prepare a ligand for docking by protonating and minimizing it """
    smi = remove_salt_from_smiles(smi)
    prot_smi = protonate_smiles(smi)
    mol = Chem.MolFromSmiles(prot_smi)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)

    lig_sdf = f"{get_output_dir()}/lig.sdf"
    save_sdf(mol, lig_sdf)
    return lig_sdf

UNIPROT2SPLIT = None
SPLIT_MODELS = {}
def get_gnina_model_args(uniprot):
    """ Figured out which split to use and return the gnina model args """
    global UNIPROT2SPLIT, SPLIT_MODELS
    gnina_gs_folder = f"gs://{CONFIG.storage.bucket}/gnina"

    if UNIPROT2SPLIT is None:
        fname = download_gs_file(f"{gnina_gs_folder}/uniprot2split.json")
        UNIPROT2SPLIT = json.load(open(fname, "r"))

    if uniprot not in UNIPROT2SPLIT:
        print("Warning: Uniprot not found in gnina models")
        return ""

    split = UNIPROT2SPLIT[uniprot]
    if split not in SPLIT_MODELS:
        models = []
        for i in range(5):
            fname = download_gs_file(f"{gnina_gs_folder}/models/{split}/{i}.caffemodel")
            models.append(fname)
        SPLIT_MODELS[split] = models

    cmd = ["--cnn_weights"] + SPLIT_MODELS[split]
    cmd += ["--cnn_model"] + ["data/gnina/default2018.model"] * 5
    return " ".join(cmd)

def run_gnina(
    rec,
    lig,
    poc_indices=None,
    box_center=None,
    box_size=None,
    uniprot_id=None,
):
    """ Saves everything to pdb/sdf and runs gnina. Can automatically
    determine box sized based on the pocket indices if we want.
    This will also automatically protonate and minimize the ligands.
    Returns a list of RDKit molecules with the docked conformations.
    If uniprot_id is not None, will use a version of gnina that wasn't
    trained on that protein. """

    if poc_indices is not None:
        from openmm import unit
        assert box_center is None and box_size is None
        all_poc_pos = rec.positions[poc_indices].value_in_unit(unit.angstroms)
        # box_center = all_poc_pos.mean(axis=0)
        box_center = 0.5*(all_poc_pos.max(axis=0) + all_poc_pos.min(axis=0))
        box_size = all_poc_pos.max(axis=0) - all_poc_pos.min(axis=0)
    else:
        assert box_center is not None and box_size is not None

    rec_pdb = f"{get_output_dir()}/rec.pdb"
    save_modeller_pdb(rec, rec_pdb)

    # convert to smiles if needed
    if isinstance(lig, Chem.Mol):
        lig = Chem.MolToSmiles(lig)

    lig_sdf = prepare_ligand(lig)
    out_sdf = f"{get_output_dir()}/out.sdf"

    center_args = (
        f"--center_x {box_center[0]} --center_y {box_center[1]} --center_z {box_center[2]}"
    )
    size_args = (
        f"--size_x {box_size[0]} --size_y {box_size[1]} --size_z {box_size[2]}"
    )
    model_args = get_gnina_model_args(uniprot_id) if uniprot_id is not None else ""

    gnina_cmd = f"gnina -r {rec_pdb} -l {lig_sdf} -o {out_sdf} {center_args} {size_args} {model_args}"
    print(gnina_cmd)

    subprocess.run(gnina_cmd, shell=True)

    return load_sdf(out_sdf)

def rescore_gnina(
    rec,
    lig,
    uniprot_id=None,
):
    """ Runs gnina --minimize on a single ligand and returns the resulting pose """

    rec_pdb = f"{get_output_dir()}/rec.pdb"
    save_modeller_pdb(rec, rec_pdb)

    lig_sdf = f"{get_output_dir()}/lig.sdf"
    save_sdf(lig, lig_sdf)
    out_sdf = f"{get_output_dir()}/out.sdf"
    model_args = get_gnina_model_args(uniprot_id) if uniprot_id is not None else ""

    gnina_cmd = f"gnina --minimize -r {rec_pdb} -l {lig_sdf} -o {out_sdf} {model_args}"
    print(gnina_cmd)

    subprocess.run(gnina_cmd, shell=True)

    return load_sdf(out_sdf)