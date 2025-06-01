import argparse
from functools import partial
import sys
from traceback import print_exc
import numpy as np
from tqdm import tqdm
from common.db import (
    add_maybe_null_to_query,
    exec_sql,
    get_alt_struct_id,
    get_engine,
    get_target_id_by_name,
    PMFDataset as DBPMFDataset,
    get_baseline_model_id,
)
from common.efb import calc_best_efb, calc_efb
from common.metric_utils import calc_ef, roc_auc
from common.wandb_utils import get_wandb_model
from pmf_net.scorer import PMFScore
from common.utils import decompress_mol, load_config
from scipy import stats
from sqlalchemy.orm import Session
import torch


def bootstrap_metric(
    metric,
    *args,
    n_resamples=1000,
    method="percentile",
    full_bootstrap_dist=False,
    **kwargs,
):
    """Bootstraps metric on args, return result, low, and high"""
    val = metric(*args)
    result = stats.bootstrap(
        args, metric, n_resamples=n_resamples, method=method, **kwargs
    )
    if full_bootstrap_dist:
        return (
            val,
            result.confidence_interval.low,
            result.confidence_interval.high,
            result.bootstrap_distribution,
        )
    else:
        return val, result.confidence_interval.low, result.confidence_interval.high


def spearmanr(x, y):
    return stats.spearmanr(x, y).statistic


def pearsonr(x, y):
    return stats.pearsonr(x, y).statistic


def get_scoring_metrics(
    act_preds,
    decoy_preds,
    act_pKs,
    act_cutoff=5,
    negate_preds=True,
    full_bootstrap_dist=False,
    remove_nan=False,
):
    """Returns spearman, correlation, and efb with confidence intervals"""

    if len(act_preds) == 0:
        return {}

    act_preds = np.array(act_preds)
    decoy_preds = np.array(decoy_preds)

    if negate_preds:
        act_preds = -act_preds
        decoy_preds = -decoy_preds

    act_nan_mask = np.isnan(act_preds) | np.isnan(act_pKs)
    if remove_nan:
        # Remove NaN values
        act_preds = act_preds[~act_nan_mask]
        act_pKs = act_pKs[~act_nan_mask]
        decoy_preds = decoy_preds[~np.isnan(decoy_preds)]
        act_nan_mask = np.zeros_like(act_preds, dtype=bool)
    else:
        # NaN to smallest value
        act_preds[act_nan_mask] = np.min(act_preds[~act_nan_mask])
        decoy_preds[np.isnan(decoy_preds)] = np.min(decoy_preds[~np.isnan(decoy_preds)])

    act_mask = act_pKs > act_cutoff

    if (~act_nan_mask).sum() > 1:
        spearman, *spearman_rest = bootstrap_metric(
            spearmanr,
            act_preds[~act_nan_mask],
            act_pKs[~act_nan_mask],
            paired=True,
            full_bootstrap_dist=full_bootstrap_dist,
        )
        correlation, *correlation_rest = bootstrap_metric(
            pearsonr,
            act_preds[~act_nan_mask],
            act_pKs[~act_nan_mask],
            paired=True,
            full_bootstrap_dist=full_bootstrap_dist,
        )

        correlation_low = correlation_rest[0]
        correlation_high = correlation_rest[1]
        spearman_low = spearman_rest[0]
        spearman_high = spearman_rest[1]
        
    else:
        spearman_rest = None
        correlation_rest = None
        spearman = spearman_low = spearman_high = np.nan
        correlation = correlation_low = correlation_high = np.nan

    if act_mask.sum() > 1 and len(decoy_preds) > 1:
        # EFB_max
        efb, *efb_rest = bootstrap_metric(
            calc_best_efb, act_preds[act_mask], decoy_preds, full_bootstrap_dist=full_bootstrap_dist
        )

        # EFB 1%
        calc_efb_1 = partial(calc_efb, select_frac=0.01)
        efb_1, *efb_1_rest = bootstrap_metric(
            calc_efb_1, act_preds[act_mask], decoy_preds, full_bootstrap_dist=full_bootstrap_dist
        )

        # EF 1%
        calc_ef_1 = partial(calc_ef, select_frac=0.01)
        ef_1, *ef_1_rest = bootstrap_metric(
            calc_ef_1, act_preds[act_mask], decoy_preds, full_bootstrap_dist=full_bootstrap_dist
        )

        # AUC
        auc, *auc_rest = bootstrap_metric(
            roc_auc, act_preds[act_mask], decoy_preds, full_bootstrap_dist=full_bootstrap_dist
        )

        efb_low = efb_rest[0]
        efb_high = efb_rest[1]
        efb_1_low = efb_1_rest[0]
        efb_1_high = efb_1_rest[1]
        ef_1_low = ef_1_rest[0]
        ef_1_high = ef_1_rest[1]
        auc_low = auc_rest[0]
        auc_high = auc_rest[1]

    else:
        efb_rest = None
        efb_1_rest = None
        ef_1_rest = None
        auc_rest = None
        efb = efb_low = efb_high = np.nan
        efb_1 = efb_1_low = efb_1_high = np.nan
        ef_1 = ef_1_low = ef_1_high = np.nan
        auc = auc_low = auc_high = np.nan

    ret = {
        "spearman": spearman,
        "spearman_low": spearman_low,
        "spearman_high": spearman_high,
        "correlation": correlation,
        "correlation_low": correlation_low,
        "correlation_high": correlation_high,
        "EFB": efb,
        "EFB_low": efb_low,
        "EFB_high": efb_high,
        "EFB_1": efb_1,
        "EFB_1_low": efb_1_low,
        "EFB_1_high": efb_1_high,
        "EF_1": ef_1,
        "EF_1_low": ef_1_low,
        "EF_1_high": ef_1_high,
        "auc": auc,
        "auc_low": auc_low,
        "auc_high": auc_high,
    }

    if full_bootstrap_dist:
        ret["spearman_dist"] = spearman_rest[2] if spearman_rest is not None else None
        ret["correlation_dist"] = correlation_rest[2] if correlation_rest is not None else None
        ret["EFB_dist"] = efb_rest[2] if efb_rest is not None else None
        ret["EFB_1_dist"] = efb_1_rest[2] if efb_1_rest is not None else None
        ret["EF_1_dist"] = ef_1_rest[2] if ef_1_rest is not None else None
        ret["auc_dist"] = auc_rest[2] if auc_rest is not None else None

    return ret


def get_all_scoring_metrics(
    act_preds,
    costruct_preds,
    decoy_preds,
    act_pKs,
    costruct_pKs,
    act_cutoff=5,
    negate_preds=True,
    full_bootstrap_dist=False,
):
    """Returns scoring metrics for actives vs decoys and co-structures vs decoys"""

    dock_scoring_metrics = get_scoring_metrics(
        act_preds, decoy_preds, act_pKs, act_cutoff, negate_preds, full_bootstrap_dist
    )
    costruct_scoring_metrics = get_scoring_metrics(
        costruct_preds, decoy_preds, costruct_pKs, act_cutoff, negate_preds, full_bootstrap_dist
    )

    ret = {**dock_scoring_metrics}
    for key, val in costruct_scoring_metrics.items():
        ret[f"crystal_{key}"] = val

    return ret


def get_model_scoring_metrics(
    engine, model_id, target_id, alt_struct_id, baseline_id=None, negate_preds=True
):
    """Assumes model has been run and the results saved to the docking_results table"""

    act_query = f"""
    SELECT d.score, a."pK", mol.mol AS smiles FROM docking_results d
    JOIN activities a ON a.mol_id = d.mol_id
    JOIN molecules mol ON mol.id = d.mol_id
    WHERE d.target_id = {target_id}
    AND a.target_id = {target_id}
    AND d.model_id = {model_id}
    """
    act_query = add_maybe_null_to_query(act_query, "d.alt_structure_id", alt_struct_id)

    act_df = exec_sql(engine, act_query)
    # make sure smiles are unique
    act_df = act_df.drop_duplicates(subset="smiles")

    crystal_query = f"""
    SELECT d.score, c."pK", mol.mol AS smiles FROM docking_results d
    JOIN co_structures c ON c.mol_id = d.mol_id
    JOIN molecules mol ON mol.id = d.mol_id
    WHERE d.target_id = {target_id}
    AND c.target_id = {target_id}
    AND d.model_id = {model_id}
    AND c."pK" IS NOT NULL
    """
    crystal_query = add_maybe_null_to_query(
        crystal_query, "d.alt_structure_id", alt_struct_id
    )

    crystal_df = exec_sql(engine, crystal_query)
    # make sure smiles are unique
    crystal_df = crystal_df.drop_duplicates(subset="smiles")

    decoy_query = f"""
    SELECT d.score, mol.mol AS smiles FROM docking_results d
    JOIN decoys a ON a.mol_id = d.mol_id
    JOIN molecules mol ON mol.id = d.mol_id
    WHERE d.target_id = {target_id}
    AND a.target_id = {target_id}
    AND d.model_id = {model_id}
    """
    decoy_query = add_maybe_null_to_query(
        decoy_query, "d.alt_structure_id", alt_struct_id
    )

    decoy_df = exec_sql(engine, decoy_query)
    # make sure smiles are unique
    decoy_df = decoy_df.drop_duplicates(subset="smiles")

    act_preds = act_df.score
    costruct_preds = crystal_df.score
    decoy_preds = decoy_df.score
    act_pKs = act_df.pK
    costruct_preds = crystal_df.score

    return get_all_scoring_metrics(
        act_preds,
        costruct_preds,
        decoy_preds,
        act_pKs,
        crystal_df.pK,
        negate_preds=negate_preds,
    )


def rescore_mols(model, df):
    """Returns a np array of scores for the rescored poses in the dataframe"""
    old_deriv = model.derivative
    model.derivative = False

    scores = []
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = PMFScore(model, device)

        for pose in tqdm(df.pose):
            try:
                score, *rest = scorer(decompress_mol(pose))
            except KeyboardInterrupt:
                raise
            except:
                print_exc()
                score = np.nan
            scores.append(score)

    finally:
        model.derivative = old_deriv

    return np.array(scores)


def rescore_actives(
    engine, model, target_id, baseline_id, alt_struct_id, limit=None, step=1
):
    """Rescores the active docking results from the baseline model with
    the current model. Also returns the pKs for the actives."""

    act_query = f"""
    SELECT d.pose, a."pK" FROM docking_results d
    JOIN activities a ON a.mol_id = d.mol_id
    WHERE d.target_id = {target_id}
    AND a.target_id = {target_id}
    AND d.model_id = {baseline_id}
    """
    act_query = add_maybe_null_to_query(act_query, "d.alt_structure_id", alt_struct_id)
    if limit is not None:
        act_query += f" LIMIT {limit}"

    act_df = exec_sql(engine, act_query)
    act_df = act_df.iloc[::step]

    return rescore_mols(model, act_df), act_df["pK"]


def score_co_structures(engine, model, target_id, limit=None, step=1):
    """Minimizes + scores the cocystalized ligands for a given target"""

    act_query = f"""
    SELECT c.lig_structure AS pose, c."pK" FROM co_structures c
    WHERE c.target_id = {target_id}
    AND c."pK" IS NOT NULL
    """
    if limit is not None:
        act_query += f" LIMIT {limit}"

    act_df = exec_sql(engine, act_query)
    act_df = act_df.iloc[::step]

    return rescore_mols(model, act_df), act_df["pK"]


def rescore_decoys(
    engine, model, target_id, baseline_id, alt_struct_id, limit=None, step=1
):
    """Rescores the decoy docking results from the baseline model with
    the current model."""

    decoy_query = f"""
    SELECT d.pose FROM docking_results d
    JOIN decoys a ON a.mol_id = d.mol_id
    WHERE d.target_id = {target_id}
    AND a.target_id = {target_id}
    AND d.model_id = {baseline_id}
    """
    decoy_query = add_maybe_null_to_query(
        decoy_query, "d.alt_structure_id", alt_struct_id
    )
    if limit is not None:
        decoy_query += f" LIMIT {limit}"

    decoy_df = exec_sql(engine, decoy_query)
    decoy_df = decoy_df.iloc[::step]

    return rescore_mols(model, decoy_df)


def run_model_scoring_metrics(model, target_id, baseline, alt_struct_id, decoy_step):
    """Returns the (re-)scoring metrics model with the given dataset_id,
    useful when evalling during model training"""

    engine = get_engine()
    baseline_id = get_baseline_model_id(engine, baseline)

    act_scores, act_pK = rescore_actives(
        engine, model, target_id, baseline_id, alt_struct_id
    )
    costruct_scores, costruct_pK = score_co_structures(engine, model, target_id)
    decoy_scores = rescore_decoys(
        engine, model, target_id, baseline_id, step=decoy_step
    )
    return get_all_scoring_metrics(
        act_scores, costruct_scores, decoy_scores, act_pK, costruct_pK
    )


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

    metrics = get_model_scoring_metrics(
        engine, model_id, target_id, alt_struct_id, baseline_id
    )

    print(f"Scoring metrics for {model_name} on {args.target}")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
