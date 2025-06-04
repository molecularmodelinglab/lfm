
from copy import deepcopy
from functools import partial
import os
import pickle
import sys
import pandas as pd

import numpy as np
from tqdm import tqdm
from common.docking_metrics import get_docking_metrics
from common.metric_utils import median_metric, to_str
from common.scoring_metrics import bootstrap_metric, get_all_scoring_metrics
from common.utils import all_rmsds_noh, decompress_mol, get_output_dir

def get_all_preds(
    act_df,
    dec_df,
    costruct_df,
    model_pair,
    poc_com_dist_cutoff=8.0,
):
    """Get all the predictions for a given model pair, cutting off
    the ones that are too far from the pocket"""

    model = "_".join(model_pair.split("_")[:-1])

    act_preds = np.array(act_df[model_pair])
    dec_preds = np.array(dec_df[model_pair])
    costruct_preds = np.array(costruct_df[model])

    act_poc_com_dists = np.array(act_df[model_pair + "_poc_com_dist"])
    dec_poc_com_dists = np.array(dec_df[model_pair + "_poc_com_dist"])
    costruct_poc_com_dists = np.array(costruct_df[model + "_poc_com_dist"])

    act_preds[act_poc_com_dists > poc_com_dist_cutoff] = np.nan
    dec_preds[dec_poc_com_dists > poc_com_dist_cutoff] = np.nan
    costruct_preds[costruct_poc_com_dists > poc_com_dist_cutoff] = np.nan

    return act_preds, dec_preds, costruct_preds



def get_screen_metrics(
    results, all_model_pairs, target_names, force=False, old_screen_metrics=None
):

    cache_fname = f"{get_output_dir()}/screen_metrics.pkl"
    if os.path.exists(cache_fname) and not force:
        print(f"Loading cached results from {cache_fname}")
        with open(cache_fname, "rb") as f:
            screen_metrics = pickle.load(f)
        return screen_metrics
    else:

        screen_metrics = {}
        for target_name in target_names:
            print("Evaluating target", target_name)
            act_df = results["activities"]
            dec_df = results["decoys"]
            costruct_df = results["co_structures_min"]
            act_df = act_df[act_df.target_name == target_name].reset_index(drop=True)
            dec_df = dec_df[dec_df.target_name == target_name].reset_index(drop=True)
            costruct_df = costruct_df[
                costruct_df.target_name == target_name
            ].reset_index(drop=True)
            act_pKs = act_df.pK
            costruct_pKs = costruct_df.pK


            for model_pair in tqdm(all_model_pairs):
                try:
                    key = (target_name, model_pair)
                    if key in old_screen_metrics:
                        print(
                            f"Using cached metrics for {key} from old results")
                        screen_metrics[key] = deepcopy(old_screen_metrics[key])
                        continue

                    act_preds, dec_preds, costruct_preds = get_all_preds(
                        act_df, dec_df, costruct_df, model_pair
                    )

                    if np.sum(~np.isnan(act_preds)) == 0:
                        continue

                    if np.sum(~np.isnan(dec_preds)) == 0:
                        continue

                    if np.sum(~np.isnan(costruct_preds)) == 0:
                        continue

                    screen_metrics[key] = get_all_scoring_metrics(
                        act_preds,
                        costruct_preds,
                        dec_preds,
                        act_pKs,
                        costruct_pKs,
                        negate_preds=True,
                        full_bootstrap_dist=True,
                    )

                except KeyboardInterrupt:
                    raise
                except:
                    # raise
                    pass
                    # print("Error evaluating target", target_name, model_pair)
                    # print_exc()

        with open(cache_fname, "wb") as f:
            pickle.dump(screen_metrics, f)
        return screen_metrics

def get_all_docking_metrics(results, all_model_pairs, target_names, force=False):

    cache_fname = f"{get_output_dir()}/docking_metrics.pkl"
    if os.path.exists(cache_fname) and not force:
        print(f"Loading cached results from {cache_fname}")
        with open(cache_fname, "rb") as f:
            docking_metrics = pickle.load(f)
        return docking_metrics

    docking_metrics = {}
    for target_name in target_names:
        print("Evaluating target", target_name)
        costruct_df = results["co_structures"]
        costruct_df = costruct_df[costruct_df.target_name == target_name].reset_index(
            drop=True
        )
        for model_pair in tqdm(all_model_pairs):
            pred_poses = [
                decompress_mol(x) if isinstance(x, bytes) else None
                for x in costruct_df[model_pair + "_pred_pose"].values
            ]
            true_poses = [
                decompress_mol(x) if isinstance(x, bytes) else None
                for x in costruct_df[model_pair + "_true_pose"].values
            ]
            key = (target_name, model_pair)
            docking_metrics[key] = get_docking_metrics(pred_poses, true_poses)

    with open(cache_fname, "wb") as f:
        pickle.dump(docking_metrics, f)

    return docking_metrics


def get_median_docking_metrics(results, all_model_pairs, target_names, force=False):

    cache_fname = f"{get_output_dir()}/median_docking_metrics.pkl"
    if os.path.exists(cache_fname) and not force:
        print(f"Loading cached results from {cache_fname}")
        with open(cache_fname, "rb") as f:
            median_docking_df = pickle.load(f)
        return median_docking_df

    rows = []
    for model_pair in tqdm(all_model_pairs):
        print("Computing median docking metrics for", model_pair)

        cur_preds = []
        for target_name in target_names:
            costruct_df = results["co_structures"]
            costruct_df = costruct_df[
                costruct_df.target_name == target_name
            ].reset_index(drop=True)
            pred_poses = costruct_df[model_pair + "_pred_pose"].values
            true_poses = costruct_df[model_pair + "_true_pose"].values

            top_1_rmsds = []
            top_any_rmsds = []

            for pred, true in zip(pred_poses, true_poses):
                pred = decompress_mol(pred) if isinstance(pred, bytes) else None
                true = decompress_mol(true) if isinstance(true, bytes) else None
                if pred is None:
                    all_rmsds = [np.inf]
                try:
                    all_rmsds = all_rmsds_noh(pred, true)
                except KeyboardInterrupt:
                    raise
                except:
                    # raise
                    # print_exc()
                    all_rmsds = [np.inf]
                top_1_rmsds.append(all_rmsds[0])
                top_any_rmsds.append(min(all_rmsds))

            if len(top_1_rmsds) < 2:
                continue

            cur_preds.append(top_1_rmsds)
            cur_preds.append(top_any_rmsds)

        metrics = {}
        cutoff = 2
        metrics[f"top_1_acc_{cutoff}"] = lambda top_1_rmsds, _: np.mean(
            np.array(top_1_rmsds) < cutoff
        )
        metrics[f"top_any_acc_{cutoff}"] = lambda _, top_any_rmsds: np.mean(
            np.array(top_any_rmsds) < cutoff
        )

        median_metrics = {
            metric: partial(median_metric, metrics[metric]) for metric in metrics
        }
        row = {"model": model_pair}
        for metric_name, metric_func in median_metrics.items():
            m, m_low, m_high = bootstrap_metric(metric_func, *cur_preds)
            row[metric_name] = m
            row[f"{metric_name}_low"] = m_low
            row[f"{metric_name}_high"] = m_high
        rows.append(row)

    median_docking_df = pd.DataFrame(rows)

    with open(cache_fname, "wb") as f:
        pickle.dump(median_docking_df, f)

    return median_docking_df

def print_median_df(
    median_df,
    median_docking_df,
    all_docking_models,
    mets_to_show,
    symmetric=["auc", "crystal_auc", "top_1_acc_2"],
    sigfigs=2,
):

    for docking_model in all_docking_models:

        rows_str = []

        model_names = {
            "Vina": f"vina_{docking_model}",
            "GNINA": f"gnina_{docking_model}",
            "LFM": f"mf_50_5_{docking_model}",
        }

        maxes = {}
        for model_name, model_key in model_names.items():
            row = median_df[median_df.model == model_key].iloc[0].to_dict()
            row.update(
                median_docking_df[median_docking_df.model == model_key]
                .iloc[0]
                .to_dict()
            )
            for metric in mets_to_show:
                val = row[metric]
                if metric in maxes:
                    maxes[metric] = max(maxes[metric], val)
                else:
                    maxes[metric] = val

        for model_name, model_key in model_names.items():
            row = median_df[median_df.model == model_key].iloc[0].to_dict()
            row.update(
                median_docking_df[median_docking_df.model == model_key]
                .iloc[0]
                .to_dict()
            )

            if model_name == "GNINA":
                model_name = "\\textsc{Gnina}"

            postfix = {
                "unidock": "UD",
                "diffdock": "DD",
            }

            out_row = {"model": f"{model_name} ({postfix[docking_model]})"}

            for metric in mets_to_show:
                m = row[metric]
                m_low = row[f"{metric}_low"]
                m_high = row[f"{metric}_high"]

                if metric in symmetric:
                    plus_minus = 0.5 * ((m_high - m) + (m - m_low))
                    if metric == "top_1_acc_2":
                        # percent
                        m_str = f"{to_str(m*100, sigfigs)} \\textpm \ {to_str(plus_minus*100, 2)} \%"
                    else:
                        m_str = (
                            f"{to_str(m, sigfigs)} \\textpm \ {to_str(plus_minus, 1)}"
                        )
                else:
                    m_str = f"{to_str(m, sigfigs)} [{to_str(m_low, sigfigs)}, {to_str(m_high, sigfigs)}]"

                max_m = maxes[metric]
                if m == max_m:
                    m_str = f"\\textbf{{{m_str}}}"

                out_row[metric] = f"{m_str}"

            rows_str.append(out_row)

        median_df_str = pd.DataFrame(rows_str)
        # print(docking_model)
        latex = median_df_str.to_latex(index=False)
        for line in latex.split("\n")[4:7]:
            print(line)
        if docking_model == "unidock":
            print("\midrule")

if __name__ == "__main__":

    FORCE_RECOMPUTE = False

    target_names = [
        "MCL1",
        "ESR1",
        "MDM2",
        "CDK2",
        "HSP90",
        "BRD4"
    ]

    results_folder = sys.argv[1]
    results = {
        key: pd.read_parquet(os.path.join(results_folder, f"{key}.parquet"))
        for key in ["activities", "decoys", "co_structures_min", "co_structures"]
    }
    all_model_pairs = [ c for c in results["decoys"].columns[2:] if "dt" not in c and "poc_com" not in c ]


    screen_metrics = get_screen_metrics(
        results, all_model_pairs, target_names, force=FORCE_RECOMPUTE
    )

    median_dists = {}
    rows = []
    for model_pair in all_model_pairs:

        cur_row = {
            "model": model_pair,
        }
        for metric in  ["EFB", "crystal_EFB", "EFB_1", "crystal_EFB_1", "auc", "crystal_auc"]:
            preds = []
            dists = []
            for target in target_names:
                pred = screen_metrics[(target, model_pair)][metric]
                if not np.isnan(pred):
                    dist = screen_metrics[(target, model_pair)][metric + "_dist"]
                    preds.append(pred)
                    dists.append(dist)

            d = np.median(np.stack(dists), axis=0)
            median_dists[(model_pair, metric)] = d

            confidence_level = 0.95
            alpha = ((1 - confidence_level)/2)
            lower = np.percentile(d, 100*alpha)
            upper = np.percentile(d, 100*(1-alpha))
            cur_row[metric] = np.median(preds)
            cur_row[metric + "_low"] = lower
            cur_row[metric + "_high"] = upper

        rows.append(cur_row)
    median_df = pd.DataFrame(rows)

    median_docking_df = get_median_docking_metrics(results, all_model_pairs, target_names, force=FORCE_RECOMPUTE)

    all_docking_models = ["unidock", "diffdock"]
    print("Table 1")
    print("==" * 20)
    print_median_df(
        median_df,
        median_docking_df,
        all_docking_models,
        ["EFB", "EFB_1", "auc"],
    )

    print("\n\n")
    print("Table 2")
    print("==" * 20)

    print_median_df(
        median_df,
        median_docking_df,
        all_docking_models,
        ["crystal_EFB", "crystal_EFB_1", "crystal_auc", "top_1_acc_2"],
    )