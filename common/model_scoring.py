from functools import lru_cache
import sys
from time import time
from traceback import print_exc

from tqdm import tqdm
from common.baselines import get_unscored_costructs
from common.celery_app import celery_app
from common.db import (
    CoStructure,
    DockingResult,
    PMFModel,
    PMFScoringProtocol,
    ScoringResult,
    SolvationScoringProtocol,
    Target,
    get_alt_struct_id,
    get_engine,
    Molecule as DBMolecule,
    get_target_id_by_name,
    get_baseline_model_id,
    get_target_struct_and_pocket,
)
from common.task_queue import task
from pmf_net.scorer import PMFScore
from pmf_net.solv_scorer import PMFSolvScore
from sqlalchemy.orm import Session, aliased
from sqlalchemy import and_, exists, text
from omegaconf import OmegaConf
from common.utils import CONFIG
from common.wandb_utils import (
    get_old_model,
    get_old_run,
    load_old_run_config,
    get_wandb_model,
)


@lru_cache(maxsize=None)
def get_pmf_scorer(model_name, protocol_id):
    """Get a PMFScore object for the given model name"""

    model = get_old_model(model_name)
    model.eval()
    model.derivative = False

    if protocol_id is None:
        # default protocol
        scorer = PMFScore(model, "cuda")
    else:
        engine = get_engine()
        with Session(engine) as session:
            protocol = session.query(PMFScoringProtocol).get(protocol_id)

            if isinstance(protocol, SolvationScoringProtocol):
                scorer = PMFSolvScore(
                    model,
                    "cuda",
                    n_steps=protocol.n_steps,
                    n_burnin=protocol.n_burnin,
                    report_interval=protocol.report_interval,
                )
            else:
                scorer = PMFScore(
                    model,
                    "cuda",
                    minmize_mm=protocol.minimize_mm,
                    minimize_pmf=protocol.minimize_pmf,
                    mm_restraint_k=protocol.mm_restraint_k,
                )

    return scorer


@task(max_runtime_hours=0.5)
def pmf_rescore(target_id, result_id, model_name, protocol_id):
    """Rescore a docking result with a PMF model"""

    engine = get_engine()
    model_id = get_wandb_model(engine, model_name, target_id)
    scorer = get_pmf_scorer(model_name, protocol_id)

    with Session(engine) as session:
        prev_result = session.query(DockingResult).get(result_id)
        mol = prev_result.pose

        prev_time = time()
        try:
            score, docked, extra_data = scorer(mol)
        except KeyboardInterrupt:
            raise
        except:
            print_exc()
            score = None
            docked = None
            extra_data = None
        dt = time() - prev_time

        result = DockingResult(
            model_id=model_id,
            target_id=target_id,
            mol_id=prev_result.mol_id,
            score=score,
            pose=docked,
            extra_data=extra_data,
            protocol_id=protocol_id,
            docking_result_id=result_id,
            dt=dt,
        )
        session.add(result)
        session.commit()


@task(max_runtime_hours=0.5)
def pmf_score_costruct(target_id, costruct_id, model_name, protocol_id):
    """Rescore a docking result with a PMF model"""

    engine = get_engine()
    model_id = get_wandb_model(engine, model_name, target_id)
    scorer = get_pmf_scorer(model_name, protocol_id)

    with Session(engine) as session:
        costruct = (
            session.query(CoStructure.lig_structure)
            .filter(CoStructure.id == costruct_id)
            .one()
        )
        mol = costruct.lig_structure

        prev_time = time()
        try:
            score, docked, extra_data = scorer(mol)
        except KeyboardInterrupt:
            raise
        except:
            print_exc()
            score = None
            docked = None
            extra_data = None
        dt = time() - prev_time

        result = ScoringResult(
            model_id=model_id,
            target_id=target_id,
            costructure_id=costruct_id,
            score=score,
            pose=docked,
            extra_data=extra_data,
            protocol_id=protocol_id,
            dt=dt,
        )
        session.add(result)
        session.commit()


def queue_pmf_costruct(
    engine, target_name, model_name, protocol_name, redo=False
):
    """ Queue PMF scoring tasks for all unscored costructures of a target
    """

    target_id = get_target_id_by_name(engine, target_name)
    model_id = get_wandb_model(engine, model_name, target_id)

    with Session(engine) as sess:

        if protocol_name is not None:
            protocol = (
                sess.query(PMFScoringProtocol).filter_by(name=protocol_name).one()
            )
            protocol_id = protocol.id
        else:
            protocol_id = None

        # if redo, delete all docking results for this target
        if redo:
            print("Deleting pmf scoring results for target", target_name)
            sess.query(ScoringResult).filter_by(
                target_id=target_id, model_id=model_id, alt_structure_id=None, protocol_id=protocol_id
            ).delete()
            sess.commit()

    costruct_ids = get_unscored_costructs(engine, target_id, model_id, None, protocol_id=protocol_id)
    print("Found", len(costruct_ids), "costructures to score")
    args = [(target_id, costruct_id, model_name, protocol_id) for costruct_id in costruct_ids]

    pmf_score_costruct.delay_bulk(engine, args)


def queue_pmf_rescoring(
    engine,
    target_name,
    model_name,
    from_baseline,
    protocol_name,
    alt_struct_name,
    limit,
    redo,
    immediate,
):
    """Queue rescore tasks for all docking results for a target"""
    target_id = get_target_id_by_name(engine, target_name)
    model_id = get_wandb_model(engine, model_name, target_id)
    from_baseline_id = get_baseline_model_id(engine, from_baseline)
    alt_struct_id = get_alt_struct_id(engine, target_id, alt_struct_name)
    with Session(engine) as sess:
        if protocol_name is not None:
            protocol = (
                sess.query(PMFScoringProtocol).filter_by(name=protocol_name).one()
            )
            protocol_id = protocol.id
        else:
            protocol_id = None

        # if redo, delete all docking results for this target
        if redo:
            print("Deleting pmf docking results for target", target_name)
            print("TODO: only delete rescored results! Currently deleting all")
            sess.query(DockingResult).filter_by(
                target_id=target_id,
                model_id=model_id,
                protocol_id=protocol_id,
                alt_structure_id=alt_struct_id,
            ).delete()
            sess.commit()

        # aliasedDR = aliased(DockingResult)
        # query = (
        #     sess.query(DockingResult.id)
        #     .filter(DockingResult.model_id == from_baseline_id)
        #     .filter(DockingResult.target_id == target_id)
        #     .filter(DockingResult.mol_id == DBMolecule.id)
        #     .filter(DockingResult.alt_structure_id == alt_struct_id)
        #     .filter(
        #         ~exists().where(
        #             and_(
        #                 aliasedDR.model_id == model_id,
        #                 aliasedDR.mol_id == DBMolecule.id,
        #                 aliasedDR.target_id == target_id,
        #                 aliasedDR.protocol_id == protocol_id,
        #             )
        #         )
        #     )
        # )

        # ensure that we don't get hoodwinked by currently running tasks
        sess.execute(text("ROLLBACK; BEGIN AS OF SYSTEM TIME '-1m'"))

        aliasedDR = aliased(DockingResult)
        query = (
            sess.query(DockingResult.id)
            .filter(DockingResult.model_id == from_baseline_id)
            .filter(DockingResult.target_id == target_id)
            .filter(DockingResult.alt_structure_id == alt_struct_id)
            .filter(DockingResult.mol_id == DBMolecule.id)
            .join(
                aliasedDR,
                (aliasedDR.docking_result_id == DockingResult.id)
                & (aliasedDR.model_id == model_id)
                & (aliasedDR.target_id == target_id)
                & (aliasedDR.protocol_id == protocol_id),
                isouter=True,
            )
            .filter(
                aliasedDR.id == None,
            )
        )

        if limit is not None:
            query = query.limit(limit)
        docking_results = query.all()

    print("Queuing pmf rescore tasks for", len(docking_results), "docking results")

    args = [
        (target_id, result_id, model_name, protocol_id)
        for (result_id,) in docking_results
    ]
    if immediate:
        for args in tqdm(args):
            pmf_rescore(*args)
    else:
        pmf_rescore.delay_bulk(engine, args)
    # for (result_id,) in tqdm(docking_results):
    #     pmf_rescore_task.delay(target_id, result_id, model_name, protocol_id)


if __name__ == "__main__":
    from argparse import ArgumentParser

    engine = get_engine()
    # target_name = sys.argv[1]
    # model_name = sys.argv[2]
    # from_baseline = sys.argv[3]
    # redo = "--redo" in sys.argv

    parser = ArgumentParser()
    parser.add_argument("target_name")
    parser.add_argument("model_name")
    parser.add_argument("from_baseline")
    parser.add_argument("--alt_struct", default=None)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--immediate", action="store_true")
    args = parser.parse_args()

    queue_pmf_rescoring(
        engine,
        args.target_name,
        args.model_name,
        args.from_baseline,
        protocol_name=args.protocol,
        alt_struct_name=args.alt_struct,
        limit=args.limit,
        redo=args.redo,
        immediate=args.immediate,
    )
