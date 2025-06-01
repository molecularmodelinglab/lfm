from copy import deepcopy
import os
import subprocess
from traceback import print_exc
from typing import NamedTuple, Optional
import numpy as np
from tqdm import tqdm
from common.baselines import get_baseline_model_id, get_target_struct_and_pocket
from common.db import (
    AlternativeStructure,
    PMFScoringProtocol,
    PMFScreenResult,
    exec_sql,
    get_engine,
    get_target_id_by_name,
)
from common.model_scoring import get_pmf_scorer
from common.pose_transform import Pose, add_pose_to_mol
from common.torsion import TorsionData
from common.unidock import run_unidock
from common.utils import get_residue_atom_indices
from common.wandb_utils import get_wandb_model
from openff.toolkit.topology import Molecule
from openff.units.openmm import to_openmm
from openmmforcefields.generators import EspalomaTemplateGenerator
from openmm import app, unit
import openmm as mm
from pmf_net.pose_pred import PosePred
from pmf_net.scorer import PMFScore
from torch_geometric.data import Data
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from celery_batches import Batches
from common.celery_app import celery_app
from sqlalchemy.orm import Session


PMF_Screen_Args = NamedTuple(
    "PMF_Screen_Args",
    [
        ("mol_id", int),
        ("smiles", str),
        ("target_id", int),
        ("alt_struct_id", Optional[int]),
        ("model_name", str),
        ("protocol_id", Optional[int]),
    ],
)

BATCH_SIZE = 20


@celery_app.task(
    base=Batches,
    flush_every=BATCH_SIZE,
    worker_prefetch_multiplier=BATCH_SIZE,
    acks_late=True,
    task_reject_on_worker_lost=True,
    name="pmf_screen",
)
def pmf_screen_task(tasks):
    """Runs a batch of PMF screen tasks"""
    args = [PMF_Screen_Args(*task.args, **task.kwargs) for task in tasks]
    target_ids = [arg.target_id for arg in args]
    alt_struct_ids = [arg.alt_struct_id for arg in args]
    model_names = [arg.model_name for arg in args]
    protocol_ids = [arg.protocol_id for arg in args]

    assert len(set(target_ids)) == 1, "All tasks must have the same target_id for now!"
    assert (
        len(set(alt_struct_ids)) == 1
    ), "All tasks must have the same alt_struct_id for now!"
    assert (
        len(set(model_names)) == 1
    ), "All tasks must have the same model_name for now!"
    assert len(set(protocol_ids)) == 1, "All tasks must have the same protocol for now!"

    target_id = target_ids[0]
    alt_struct_id = alt_struct_ids[0]
    model_name = model_names[0]
    protocol_id = protocol_ids[0]

    engine = get_engine()
    # docking_model_id = get_baseline_model_id(engine, "unidock")
    model_id = get_wandb_model(engine, model_name, target_id)
    scorer = get_pmf_scorer(model_name, protocol_id)

    rec, poc_residues = get_target_struct_and_pocket(engine, target_id, alt_struct_id)
    poc_indices = get_residue_atom_indices(rec.topology, poc_residues)

    mol_ids = [arg.mol_id for arg in args]
    smis = [arg.smiles for arg in args]

    with Session(engine) as session:
        docked_mols = run_unidock(rec, smis, poc_indices)
        results = []
        for mol_id, mol in zip(mol_ids, docked_mols):
            try:
                score, docked, extra_data = scorer(mol)
            except KeyboardInterrupt:
                raise
            except:
                print_exc()
                score = None
                docked = None
                extra_data = None

            result = PMFScreenResult(
                model_id=model_id,
                target_id=target_id,
                mol_id=mol_id,
                score=score,
                mol=docked,
                extra_data=extra_data,
                protocol_id=protocol_id,
            )
            results.append(result)

        session.bulk_save_objects(results)
        session.commit()


def init_screen(
    target_name,
    model_name,
    alt_struct_name,
    collection,
    protocol_name,
    redo=False,
    limit=None,
    dry_run=False,
):
    """Initialize the task queue for the PMF screen"""

    # first clear the task queue
    # celery_app.control.purge()

    engine = get_engine()
    target_id = get_target_id_by_name(engine, target_name)
    with Session(engine) as sess:
        if alt_struct_name is None:
            alt_struct_id = None
        else:
            alt_struct_id = (
                sess.query(AlternativeStructure.id)
                .filter_by(target_id=target_id, name=alt_struct_name)
                .one()
                .id
            )
            print("Using alt struct", alt_struct_name)

    model_id = get_wandb_model(engine, model_name, target_id)
    with Session(engine) as sess:
        if protocol_name is not None:
            protocol = (
                sess.query(PMFScoringProtocol).filter_by(name=protocol_name).one()
            )
            protocol_id = protocol.id
        else:
            protocol_id = None

    if redo:
        # clear existing results
        with Session(engine) as sess:
            sess.query(PMFScreenResult
            ).filter_by(
                model_id=model_id
            ).filter_by(
                target_id=target_id
            ).filter_by(
                protocol_id=protocol_id
            ).delete()
            sess.commit()

    query = f"""
        SELECT id, mol from screen_mols
        WHERE collection = '{collection}'
        AND id NOT IN (
            SELECT mol_id FROM pmf_screen_results
            WHERE model_id = {model_id}
            AND target_id = {target_id}    
            AND protocol_id {'IS NULL' if protocol_id is None else "= " + str(protocol_id)}
        )
    """

    if limit is not None:
        query += f" LIMIT {limit}"

    df = exec_sql(engine, query)

    if dry_run:
        print("Dry run, not enqueuing tasks")
        print("Would enqueue", len(df), "tasks")
        return

    for i, row in tqdm(df.iterrows(), total=len(df)):
        arg = PMF_Screen_Args(
            mol_id=row.id,
            smiles=row.mol,
            target_id=target_id,
            alt_struct_id=alt_struct_id,
            model_name=model_name,
            protocol_id=protocol_id,
        )
        pmf_screen_task.delay(*arg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target_name", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--alt_struct", type=str, default=None)
    parser.add_argument("--collection", type=str, default="UNC")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    init_screen(
        args.target_name,
        args.model_name,
        args.alt_struct,
        args.collection,
        args.protocol,
        args.redo,
        args.limit,
        args.dry_run,
    )
