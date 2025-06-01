from copy import deepcopy
from glob import glob
import os
import shutil
import subprocess
from time import time
from common.utils import get_output_dir, load_sdf, protonate_mol, save_modeller_pdb
from rdkit import Chem
from rdkit.Chem import AllChem
from sqlalchemy.orm import Session
from common.db import DockingResult, get_baseline_model_id, get_engine, Molecule as DBMolecule, get_target_struct_and_pocket
from common.task_queue import task, task_loop


def run_diffdock(rec, smi):
    """ Run DiffDock with the given receptor and ligand. Returns the
    3D rdkit mol with all the poses """
    
    rec_fname = os.path.abspath(f"{get_output_dir()}/diffdock_rec.pdb")
    save_modeller_pdb(rec, rec_fname)

    out_dir = os.path.abspath(f"{get_output_dir()}/diffdock_out")
    shutil.rmtree(out_dir, ignore_errors=True)

    cur_dir = os.path.abspath(os.getcwd())
    try:
        os.chdir("/home/appuser/DiffDock")
        cmd = f'python -m inference --protein_path {rec_fname} --ligand "{smi}" --out_dir {out_dir}'

        print(f"Running DiffDock with command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        # rank molecules according to confidence (low to high)
        rank_and_mol = []
        for sdf in glob(f"{out_dir}/complex_0/rank*_confidence*.sdf"):
            rank = int(sdf.split("/")[-1].split("_")[0][4:]) 
            mol = load_sdf(sdf)
            rank_and_mol.append((rank, mol))

        rank_and_mol = sorted(rank_and_mol, key=lambda x: x[0])
        mols = [m for _, m in rank_and_mol]

        return mols

    finally:
        os.chdir(cur_dir)


@task(max_runtime_hours=0.5)
def diffdock_dock(target_id, mol_id, alt_struct_id):

    engine = get_engine()
    model_id = get_baseline_model_id(engine, "diffdock")
    rec, _ = get_target_struct_and_pocket(engine, target_id, alt_struct_id)

    with Session(engine) as session:
        mol = (
            session.query(DBMolecule)
            .filter(DBMolecule.id == mol_id)
            .one()
        )
        mol_rd = mol.mol # protonate_mol(mol.mol)
        mol_rd = Chem.RemoveHs(mol_rd)
        smi = Chem.MolToSmiles(mol_rd)

        prev_time = time()
        pose = run_diffdock(rec, smi)
        dt = time() - prev_time

        result = DockingResult(
            target_id=target_id,
            mol_id=mol_id,
            model_id=model_id,
            pose=pose,
            alt_structure_id=alt_struct_id,
            dt=dt,
        )
        session.add(result)
        session.commit()
        print(f"DiffDock docking took {dt:.2f} seconds")


if __name__ == "__main__":
    engine = get_engine()
    task_loop(engine, "diffdock_dock")