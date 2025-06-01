import argparse
import sys
from traceback import print_exc
import numpy as np
from tqdm import tqdm
from common.db import (
    CoStructure,
    DockingResult,
    add_maybe_null_to_query,
    exec_sql,
    get_alt_struct_id,
    get_engine,
    get_target_id_by_name,
    PMFDataset as DBPMFDataset,
    get_baseline_model_id,
)
from common.efb import calc_best_efb
from common.wandb_utils import get_wandb_model
from pmf_net.pose_pred import predict_poses_parellel
from pmf_net.scorer import PMFScore
from common.utils import CONFIG, all_rmsds_noh, decompress_mol, load_config
from scipy import stats
from sqlalchemy.orm import Session
import torch


def get_docking_metrics(pred_poses, true_poses, rmsd_cutoffs=[2, 5]):
    """This requires a list of mols (with possibly multiple poses) and the true poses.
    Returns a dict of metrics"""

    top_1_rmsds = []
    top_any_rmsds = []

    for pred, true in zip(pred_poses, true_poses):
        if pred is None:
            all_rmsds = [np.inf]
        try:
            all_rmsds = all_rmsds_noh(pred, true)
        except KeyboardInterrupt:
            raise
        except:
            # print_exc()
            all_rmsds = [np.inf]
        top_1_rmsds.append(all_rmsds[0])
        top_any_rmsds.append(min(all_rmsds))

    # just in case, remove any nan values
    top_1_rmsds = [r for r in top_1_rmsds if not np.isnan(r)]
    top_any_rmsds = [r for r in top_any_rmsds if not np.isnan(r)]

    ret = {}
    for cutoff in rmsd_cutoffs:
        top_1_acc = np.mean(np.array(top_1_rmsds) < cutoff)
        top_any_acc = np.mean(np.array(top_any_rmsds) < cutoff)
        ret[f"top_1_acc_{cutoff}"] = top_1_acc
        ret[f"top_any_acc_{cutoff}"] = top_any_acc

    return ret


def rescore_poses(model, df):
    """Returns a list of rescores poses for each docked pose in the dataframe"""
    old_deriv = model.derivative
    model.derivative = False

    new_poses = []
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = PMFScore(model, device)

        for pose in tqdm(df.pose):
            try:
                score, new_pose, _ = scorer(decompress_mol(pose))
            except KeyboardInterrupt:
                raise
            except:
                print_exc()
                new_pose = None
            new_poses.append(new_pose)

    finally:
        model.derivative = old_deriv

    return new_poses


def rescore_docked_costructures(
    engine, model, target_id, baseline_id, alt_struct_id, limit=None, step=1
):
    """Rescores the docked costructures from the baseline model with the current model.
    Returns the predicted poses and the true poses"""

    act_query = f"""
    SELECT d.pose, c.lig_structure FROM docking_results d
    JOIN co_structures c ON c.mol_id = d.mol_id
    WHERE d.target_id = {target_id}
    AND c.target_id = {target_id}
    AND d.model_id = {baseline_id}
    """
    act_query = add_maybe_null_to_query(act_query, "d.alt_structure_id", alt_struct_id)

    if limit is not None:
        act_query += f" LIMIT {limit}"

    act_df = exec_sql(engine, act_query)
    act_df = act_df.iloc[::step]

    if len(act_df) == 0:
        raise ValueError("No docked costructures found")

    true_poses = [decompress_mol(pose) for pose in act_df["lig_structure"]]

    return rescore_poses(model, act_df), true_poses


def get_model_docking_metrics(engine, model_id, target_id, alt_struct_id, baseline_id):
    """Assumes model has been run and the results saved to the docking_results table"""

    with Session(engine) as sess:
        results = (
            sess.query(CoStructure.lig_structure, DockingResult.pose)
            .filter(CoStructure.target_id == target_id)
            .join(DockingResult, CoStructure.mol_id == DockingResult.mol_id)
            .filter(DockingResult.target_id == target_id)
            .filter(DockingResult.model_id == model_id)
            .filter(DockingResult.alt_structure_id == alt_struct_id)
        ).all()

        pred_poses = [r.pose for r in results]
        true_poses = [r.lig_structure for r in results]

    return get_docking_metrics(pred_poses, true_poses)


def run_model_docking_metrics(model, target_id, baseline, alt_struct_id):
    """Returns the docking metrics for the model on the dataset when
    rescored from the baseline model. If only_rescore, then only rescore
    baseline docked poses"""

    ret = {}

    engine = get_engine()
    baseline_id = get_baseline_model_id(engine, baseline)

    pred_poses, true_poses = rescore_docked_costructures(
        engine, model, target_id, baseline_id, alt_struct_id
    )
    rescore_metrics = get_docking_metrics(pred_poses, true_poses)

    for key, val in rescore_metrics.items():
        ret[f"{baseline}_{key}"] = val

    if not CONFIG.eval.docking.only_rescore:
        pred_poses = predict_poses_parellel(
            model,
            true_poses,
            n_poses=CONFIG.eval.docking.n_poses,
            n_embed=CONFIG.eval.docking.n_embed,
        )
        docked_metrics = get_docking_metrics(pred_poses, true_poses)

        for key, val in docked_metrics.items():
            ret[key] = val

    return ret

if __name__ == "__main__":
    load_config("configs/default.yaml", False)

    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--alt_struct", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)

    args = parser.parse_args()
    if args.model_name.startswith("wandb:"):
        model_type, model_name = args.model_name.split(":")
    else:
        model_type = "baseline"
        model_name = args.model_name

    engine = get_engine()

    target_id = get_target_id_by_name(engine, args.target)
    alt_struct_id = get_alt_struct_id(engine, target_id, args.alt_struct)
    if args.baseline is not None:
        baseline_id = get_baseline_model_id(engine, args.baseline)
    else:
        baseline_id = None

    match model_type:
        case "baseline":
            model_id = get_baseline_model_id(engine, model_name)
        case "wandb":
            model_id = get_wandb_model(engine, model_name, target_id)
        case _:
            print(f"Unknown model type: {model_type}")
            sys.exit(1)

    metrics = get_model_docking_metrics(engine, model_id, target_id, alt_struct_id, baseline_id)

    print(f"Docking metrics for {model_name} on {args.target}")
    for key, val in metrics.items():
        print(f"  {key}: {val}")