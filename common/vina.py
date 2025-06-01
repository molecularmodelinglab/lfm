from copy import deepcopy
import subprocess
from time import time
from traceback import print_exc
from rdkit import Chem

from sqlalchemy.orm import Session
from common.db import CoStructure, DockingResult, ScoringResult, get_baseline_model_id, get_engine, get_target_struct_and_pocket
from common.task_queue import task
from common.unidock import get_docking_box, save_lig_pdbqt, save_rec_pdbqt
from common.utils import get_output_dir, get_residue_atom_indices, load_sdf, protonate_mol
from vina import Vina

def autobox_ligand(lig, padding=2.0):
    """ Get the box size and center for the ligand """
    lig_pos = lig.GetConformer().GetPositions()
    box_center = 0.5 * (lig_pos.max(axis=0) + lig_pos.min(axis=0))
    box_size = lig_pos.max(axis=0) - lig_pos.min(axis=0)
    box_size += padding
    return box_size, box_center

def minimize_vina(rec, poc_residues, docked):
    """ Minmize docked with vina, return the minimized pose
    and the final score """

    rec_pdbqt = f"{get_output_dir()}/vina_rec.pdbqt"
    save_rec_pdbqt(rec, rec_pdbqt)

    # docked_prot = protonate_mol(docked)
    docked_prot = Chem.AddHs(docked, addCoords=True)

    # poc_indices = get_residue_atom_indices(rec.topology, poc_residues)
    # box_size, box_center = get_docking_box(rec, poc_indices)
    box_size, box_center = autobox_ligand(docked_prot)

    # compute energy for each conformer, return the best one
    poses = []
    energies  = []

    v = Vina(verbosity=0)
    v.set_receptor(rec_pdbqt)
    v.compute_vina_maps(center=box_center, box_size=box_size)

    conf_mol = deepcopy(docked_prot)
    for conformer in docked_prot.GetConformers():
        try:
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(conformer)

            lig_pdbqt = f"{get_output_dir()}/vina_lig.pdbqt"
            save_lig_pdbqt(conf_mol, lig_pdbqt)

            v.set_ligand_from_file(lig_pdbqt)

            energy_minimized = v.optimize()[0]

            if energy_minimized < min(energies, default=float("inf")):
                # Don't worry about saving non-best poses
                print(f"Found new best energy: {energy_minimized}")
                out_pdbqt = f"{get_output_dir()}/vina_out.pdbqt"
                v.write_pose(out_pdbqt, overwrite=True)

                out_sdf = out_pdbqt.replace(".pdbqt", ".sdf")
                subprocess.run(
                    f"mk_export.py {out_pdbqt} -o {out_sdf}",
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                minimized = load_sdf(out_sdf)
                energies.append(energy_minimized)
                poses.append(minimized)
        except:
            print("Error with conformer, continuing")
            print_exc()

    best_idx = energies.index(min(energies))
    minimized = poses[best_idx]
    energy_minimized = energies[best_idx]

    return minimized, energy_minimized

@task(max_runtime_hours=0.25)
def vina_rescore(target_id, result_id):
    """ Rescores a docking result with gnina """
    engine = get_engine()
    model_id = get_baseline_model_id(engine, "vina")

    with Session(engine) as session:
        prev_result = session.query(DockingResult).get(result_id)
        rec, poc_residues = get_target_struct_and_pocket(engine, target_id, prev_result.alt_structure_id)
        mol = prev_result.pose

        prev_time = time()

        try:
            docked, score = minimize_vina(rec, poc_residues, mol)
        except:
            print_exc()
            docked = None
            score = None
        dt = time() - prev_time

        result = DockingResult(
            model_id=model_id,
            target_id=target_id,
            mol_id=prev_result.mol_id,
            score=score,
            pose=docked,
            alt_structure_id=prev_result.alt_structure_id,
            dt=dt,
            docking_result_id=result_id,
        )
        session.add(result)
        session.commit()

@task(max_runtime_hours=0.25)
def vina_score_costruct(target_id, costruct_id, alt_struct_id):
    """ Scores a costruct with vina """
    engine = get_engine()
    model_id = get_baseline_model_id(engine, "vina")

    rec, poc_residues = get_target_struct_and_pocket(engine, target_id, alt_struct_id)

    with Session(engine) as session:
        costruct = session.query(CoStructure.lig_structure).filter(
            CoStructure.id == costruct_id
        ).one()

        mol = costruct.lig_structure

        prev_time = time()

        try:
            docked, score = minimize_vina(rec, poc_residues, mol)
        except:
            print_exc()
            docked = None
            score = None
        dt = time() - prev_time

        result = ScoringResult(
            model_id=model_id,
            target_id=target_id,
            costructure_id=costruct_id,
            score=score,
            pose=docked,
            alt_structure_id=alt_struct_id,
            dt=dt,
        )
        session.add(result)
        session.commit()