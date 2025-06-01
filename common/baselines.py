# Docking and scoring baselines for activity + costructure data
from argparse import ArgumentParser
import sys
from time import time
from typing import NamedTuple, Optional

from tqdm import tqdm
from traceback import print_exc
from common.diffdock import diffdock_dock
from common.gnina import rescore_gnina
from common.task_queue import task
from common.utils import compress_mol, get_residue_atom_indices
from common.vina import vina_score_costruct
from sqlalchemy.orm import Session, aliased
from sqlalchemy import and_, exists, or_, text

from common.unidock import get_unidock_score, run_unidock

from common.db import (
    Activity,
    AlternativeStructure,
    BaselineModel,
    CoStructure,
    Decoy,
    DockingResult,
    Molecule as DBMolecule,
    ProteinTarget,
    ScoringResult,
    Target,
    add_maybe_null_to_query,
    exec_sql,
    get_alt_struct_id,
    get_baseline_model_id,
    get_engine,
    get_target_id_by_name,
    get_target_struct_and_pocket,
)

Docking_Args = NamedTuple(
    "Docking_Args",
    [("target_id", int), ("mol_id", int), ("alt_struct_id", Optional[int])],
)
Scoring_Args = NamedTuple("Scoring_Args", [("target_id", int), ("costructure_id", int)])
Rescoring_Args = NamedTuple(
    "Rescoring_Args",
    [("target_id", int), ("result_id", int), ("alt_struct_id", Optional[int])],
)


try:
    from celery_batches import Batches
    from common.celery_app import celery_app

    BATCH_SIZE = 20

    @celery_app.task(
        base=Batches,
        flush_every=BATCH_SIZE,
        worker_prefetch_multiplier=BATCH_SIZE,
        acks_late=True,
        task_reject_on_worker_lost=True,
        name="unidock_dock",
    )
    def unidock_dock_task(tasks):
        """Runs unidock on a batch of molecules and adds result to the database"""
        args = [Docking_Args(*task.args, **task.kwargs) for task in tasks]
        target_ids = [arg.target_id for arg in args]
        alt_struct_ids = [arg.alt_struct_id for arg in args]

        assert (
            len(set(target_ids)) == 1
        ), "All tasks must have the same target_id for now!"
        assert (
            len(set(alt_struct_ids)) == 1
        ), "All tasks must have the same alt_struct_id for now!"

        engine = get_engine()
        model_id = get_baseline_model_id(engine, "unidock")
        rec, poc_residues = get_target_struct_and_pocket(
            engine, target_ids[0], alt_struct_ids[0]
        )
        poc_indices = get_residue_atom_indices(rec.topology, poc_residues)

        with Session(engine) as session:
            mols = (
                session.query(DBMolecule)
                .filter(DBMolecule.id.in_([arg.mol_id for arg in args]))
                .all()
            )
            smis = [mol.mol for mol in mols]

            prev_time = time()
            docked_mols = run_unidock(rec, smis, poc_indices)
            scores = [get_unidock_score(mol) for mol in docked_mols]
            total_dt = time() - prev_time
            dt_avg = total_dt / len(docked_mols)

            results = []
            for db_mol, mol, score in zip(mols, docked_mols, scores):
                results.append(
                    DockingResult(
                        model_id=model_id,
                        target_id=target_ids[0],
                        mol_id=db_mol.id,
                        score=score,
                        pose=mol,
                        alt_structure_id=alt_struct_ids[0],
                        dt=dt_avg,
                    )
                )
            session.bulk_save_objects(results)
            session.commit()

except ModuleNotFoundError:
    # for now -- we dont' want to pip install celery for diffdock
    pass


TARGET_UNIPROTS = {}


def get_target_uniprot(target_id):
    """Returns the uniprot id (if any) for a given target"""
    if target_id in TARGET_UNIPROTS:
        return TARGET_UNIPROTS[target_id]
    engine = get_engine()
    with Session(engine) as sess:
        protein_target = (
            sess.query(ProteinTarget).filter(ProteinTarget.id == target_id).first()
        )
        if protein_target is None:
            return None
        uniprot = protein_target.uniprot
        TARGET_UNIPROTS[target_id] = uniprot
    return uniprot


# @celery_app.task(
#     acks_late=True,
#     task_reject_on_worker_lost=True,
#     name="gnina_rescore",
# )
@task(max_runtime_hours=0.25)
def gnina_rescore(target_id, result_id):
    """Rescores a docking result with gnina"""
    engine = get_engine()
    model_id = get_baseline_model_id(engine, "gnina")
    uniprot_id = get_target_uniprot(target_id)

    with Session(engine) as session:
        prev_result = session.query(DockingResult).get(result_id)
        rec, poc_residues = get_target_struct_and_pocket(
            engine, target_id, prev_result.alt_structure_id
        )

        mol = prev_result.pose

        prev_time = time()

        try:
            docked = rescore_gnina(rec, mol, uniprot_id=uniprot_id)
        except:
            print_exc()
            docked = None

        if docked is not None:
            data = docked.GetPropsAsDict()
            score = -data["CNN_VS"]
        else:
            score = None
            data = None

        dt = time() - prev_time

        result = DockingResult(
            model_id=model_id,
            target_id=target_id,
            mol_id=prev_result.mol_id,
            score=score,
            pose=docked,
            extra_data=data,
            alt_structure_id=prev_result.alt_structure_id,
            dt=dt,
            docking_result_id=result_id,
        )
        session.add(result)
        session.commit()

@task(max_runtime_hours=0.25)
def gnina_score_costruct(target_id, costruct_id, alt_struct_id):
    """ Scores a costructure with gnina """

    engine = get_engine()
    model_id = get_baseline_model_id(engine, "gnina")
    uniprot_id = get_target_uniprot(target_id)

    rec, poc_residues = get_target_struct_and_pocket(
        engine, target_id, alt_struct_id
    )
    
    with Session(engine) as session:
        costruct = session.query(CoStructure.lig_structure).filter(
            CoStructure.id == costruct_id
        ).one()
        mol = costruct.lig_structure

        prev_time = time()

        try:
            docked = rescore_gnina(rec, mol, uniprot_id=uniprot_id)
        except:
            print_exc()
            docked = None

        if docked is not None:
            data = docked.GetPropsAsDict()
            score = -data["CNN_VS"]
        else:
            score = None
            data = None

        dt = time() - prev_time

        result = ScoringResult(
            model_id=model_id,
            target_id=target_id,
            costructure_id=costruct_id,
            score=score,
            pose=docked,
            extra_data=data,
            alt_structure_id=alt_struct_id,
            dt=dt,
        )
        session.add(result)
        session.commit()

def get_unscored_costructs(engine, target_id, model_id, alt_struct_id, protocol_id=None):
    """ Gets all costructures for a target that have not been scored with the model """

    with Session(engine) as sess:
        query = sess.query(CoStructure.id).filter(
            CoStructure.target_id == target_id,
        ).outerjoin(
            ScoringResult,
            and_(
                ScoringResult.costructure_id == CoStructure.id,
                ScoringResult.model_id == model_id,
                ScoringResult.alt_structure_id == alt_struct_id,
                ScoringResult.protocol_id == protocol_id,
            ),
        ).filter(ScoringResult.id == None)

        costruct_ids = [row[0] for row in query.all()]

    return costruct_ids

def queue_gnina_costruct_baseline(engine, target_name, alt_struct_name, redo=False):
    model_id = get_baseline_model_id(engine, "gnina")

    target_id = get_target_id_by_name(engine, target_name)
    alt_struct_id = get_alt_struct_id(engine, target_id, alt_struct_name)

    with Session(engine) as sess:
        # if redo, delete all docking results for this target
        if redo:
            print("Deleting gnina scoring results for target", target_name)
            sess.query(ScoringResult).filter_by(
                target_id=target_id, model_id=model_id, alt_structure_id=alt_struct_id
            ).delete()
            sess.commit()

    costruct_ids = get_unscored_costructs(
        engine, target_id, model_id, alt_struct_id
    )
    print("Found", len(costruct_ids), "costructures to score")
    args = [
        (target_id, costruct_id, alt_struct_id)
        for costruct_id in costruct_ids
    ]

    gnina_score_costruct.delay_bulk(engine, args)

def queue_vina_costruct_baseline(engine, target_name, alt_struct_name, redo=False):
    model_id = get_baseline_model_id(engine, "vina")

    target_id = get_target_id_by_name(engine, target_name)
    alt_struct_id = get_alt_struct_id(engine, target_id, alt_struct_name)

    with Session(engine) as sess:
        # if redo, delete all docking results for this target
        if redo:
            print("Deleting vina scoring results for target", target_name)
            sess.query(ScoringResult).filter_by(
                target_id=target_id, model_id=model_id, alt_structure_id=alt_struct_id
            ).delete()
            sess.commit()

    costruct_ids = get_unscored_costructs(
        engine, target_id, model_id, alt_struct_id
    )
    print("Found", len(costruct_ids), "costructures to score")
    args = [
        (target_id, costruct_id, alt_struct_id)
        for costruct_id in costruct_ids
    ]

    vina_score_costruct.delay_bulk(engine, args)

def get_undocked_mols(
    engine, target_name, alt_struct_name, model_id, docked_baseline_id=None
):
    """Returns a list of undocked molecules for a given target, alt_structure, and baseline"""

    target_id = get_target_id_by_name(engine, target_name)
    alt_struct_id = get_alt_struct_id(engine, target_id, alt_struct_name)

    # length of an empty gzipp'd string
    empty_len = 20

    mol_ids = []
    for table_name in ["activities", "decoys", "co_structures"]:
        query = f"""
        SELECT {table_name}.mol_id
        FROM {table_name}
        JOIN molecules mol ON {table_name}.mol_id = mol.id
        LEFT JOIN docking_results dr ON {table_name}.mol_id = dr.mol_id
        AND dr.target_id = {target_id}
        AND dr.model_id = {model_id}
        """
        query = add_maybe_null_to_query(query, "dr.alt_structure_id", alt_struct_id)
        if docked_baseline_id is not None:
            query += f"""
            LEFT JOIN docking_results dr2 ON dr.docking_result_id = dr2.id
            """
        query += f"""
        WHERE {table_name}.target_id = {target_id}
        AND (dr.mol_id IS NULL
        """
        if docked_baseline_id is not None:
            query += f"""
            OR dr2.model_id != {docked_baseline_id}
            """
        query += ")"

        # print("=====")
        # print(query)
        # OR LENGTH(dr.pose) = {empty_len}

        mol_ids += exec_sql(engine, query).mol_id.tolist()

    return mol_ids


def queue_unidock_baseline(engine, target_name, alt_struct_name, redo=False):
    model_id = get_baseline_model_id(engine, "unidock")

    with Session(engine) as sess:
        target = sess.query(Target).filter_by(name=target_name).one()
        if alt_struct_name is None:
            alt_struct_id = None
        else:
            alt_struct_id = (
                sess.query(AlternativeStructure.id)
                .filter_by(target_id=target.id, name=alt_struct_name)
                .one()
                .id
            )
            print("Using alt struct", alt_struct_name)

        # if redo, delete all docking results for this target
        if redo:
            print("Deleting unidock docking results for target", target_name)
            sess.query(DockingResult).filter_by(
                target_id=target.id, model_id=model_id
            ).delete()
            sess.commit()

        target_id = target.id
        all_mol_ids = get_undocked_mols(engine, target_name, alt_struct_name, model_id)

    for mol_id in tqdm(all_mol_ids):
        unidock_dock_task.delay(target_id, mol_id, alt_struct_id)


def queue_diffdock_baseline(engine, target_name, alt_struct_name, redo=False):
    # TODO -- all these functions are very similar. Collate!
    model_id = get_baseline_model_id(engine, "diffdock")

    with Session(engine) as sess:
        target = sess.query(Target).filter_by(name=target_name).one()
        if alt_struct_name is None:
            alt_struct_id = None
        else:
            alt_struct_id = (
                sess.query(AlternativeStructure.id)
                .filter_by(target_id=target.id, name=alt_struct_name)
                .one()
                .id
            )
            print("Using alt struct", alt_struct_name)

        # if redo, delete all docking results for this target
        if redo:
            print("Deleting diffdock docking results for target", target_name)
            sess.query(DockingResult).filter_by(
                target_id=target.id, model_id=model_id
            ).delete()
            sess.commit()
            print("Dequeuing all diffdock tasks for target", target_name)
            diffdock_dock.clear_queue(engine)

        target_id = target.id
        all_mol_ids = get_undocked_mols(engine, target_name, alt_struct_name, model_id)

    args = [(target_id, mol_id, alt_struct_id) for mol_id in all_mol_ids]
    print("Docking", len(args), "molecules")
    diffdock_dock.delay_bulk(engine, args)


def queue_gnina_rescoring_baseline(
    engine, target_name, alt_struct_name, from_baseline, redo
):
    model_id = get_baseline_model_id(engine, "gnina")
    from_baseline_id = get_baseline_model_id(engine, from_baseline)
    with Session(engine) as sess:
        target = sess.query(Target).filter_by(name=target_name).one()
        alt_struct_id = get_alt_struct_id(engine, target.id, alt_struct_name)

        # if redo, delete all docking results for this target
        if redo:
            print("Deleting gnina docking results for target", target_name)
            print("TODO: only delete rescored results! Currently deleting all")
            sess.query(DockingResult).filter_by(
                target_id=target.id, model_id=model_id
            ).delete()
            sess.commit()

        # ensure that we don't get hoodwinked by currently running tasks
        sess.execute(text("ROLLBACK; BEGIN AS OF SYSTEM TIME '-1m'"))

        aliasedDR = aliased(DockingResult)
        docking_results = (
            sess.query(DockingResult.id, DBMolecule)
            .filter(DockingResult.model_id == from_baseline_id)
            .filter(DockingResult.target_id == target.id)
            .filter(DockingResult.alt_structure_id == alt_struct_id)
            .filter(DockingResult.mol_id == DBMolecule.id)
            .join(
                aliasedDR,
                (aliasedDR.docking_result_id == DockingResult.id)
                & (aliasedDR.model_id == model_id),
                isouter=True,
            )
            .filter(
                aliasedDR.id == None,
            )
            .all()
        )

    # for result in tqdm(docking_results):
    #     gnina_rescore.delay(target.id, result.id)

    args_list = [(target.id, result.id) for result in docking_results]
    print("Rescoring", len(args_list), "docking results")
    gnina_rescore.delay_bulk(engine, args_list)


def queue_vina_rescoring_baseline(
    engine, target_name, alt_struct_name, from_baseline, redo
):
    from common.vina import vina_rescore

    # also a lot of duplication with the above
    model_id = get_baseline_model_id(engine, "vina")
    from_baseline_id = get_baseline_model_id(engine, from_baseline)
    with Session(engine) as sess:
        target = sess.query(Target).filter_by(name=target_name).one()
        alt_struct_id = get_alt_struct_id(engine, target.id, alt_struct_name)


        # if redo, delete all docking results for this target
        if redo:
            print("Deleting vina docking results for target", target_name)
            print("TODO: only delete rescored results! Currently deleting all")
            sess.query(DockingResult).filter_by(
                target_id=target.id, model_id=model_id
            ).delete()
            sess.commit()

            print("also clearing vina task queue")
            vina_rescore.clear_queue(engine)

        # ensure that we don't get hoodwinked by currently running tasks
        sess.execute(text("ROLLBACK; BEGIN AS OF SYSTEM TIME '-1m'"))

        aliasedDR = aliased(DockingResult)
        docking_results = (
            sess.query(DockingResult.id, DBMolecule)
            .filter(DockingResult.model_id == from_baseline_id)
            .filter(DockingResult.target_id == target.id)
            .filter(DockingResult.alt_structure_id == alt_struct_id)
            .filter(DockingResult.mol_id == DBMolecule.id)
            .join(
                aliasedDR,
                (aliasedDR.docking_result_id == DockingResult.id)
                & (aliasedDR.model_id == model_id),
                isouter=True,
            )
            .filter(
                aliasedDR.id == None,
            )
            .all()
        )
    args_list = [(target.id, result.id) for result in docking_results]
    print("Rescoring", len(args_list), "docking results")
    vina_rescore.delay_bulk(engine, args_list)


if __name__ == "__main__":
    # action = sys.argv[1]
    # baseline_name = sys.argv[2]
    # target_names = sys.argv[3].split(",")
    # redo = "--redo" in sys.argv

    parser = ArgumentParser()
    parser.add_argument("action", choices=["dock", "rescore"])
    parser.add_argument("baseline_name")
    parser.add_argument("target_names", nargs="+")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--alt_struct", default=None)
    parser.add_argument("--from_baseline", default="unidock")
    args = parser.parse_args()

    engine = get_engine()
    if args.action == "dock":

        if args.baseline_name == "diffdock":
            for target_name in args.target_names:
                queue_diffdock_baseline(engine, target_name, args.alt_struct, args.redo)
        elif args.baseline_name == "unidock":
            for target_name in args.target_names:
                queue_unidock_baseline(engine, target_name, args.alt_struct, args.redo)
        else:
            print("Unknown baseline", args.baseline_name)

    elif args.action == "rescore":

        for target_name in args.target_names:
            if args.baseline_name == "gnina":
                queue_gnina_rescoring_baseline(
                    engine, target_name, args.alt_struct, args.from_baseline, args.redo
                )
            elif args.baseline_name == "vina":
                queue_vina_rescoring_baseline(
                    engine, target_name, args.alt_struct, args.from_baseline, args.redo
                )
            else:
                print("Unknown baseline", args.baseline_name)
    else:
        print("Unknown action", args.action)
