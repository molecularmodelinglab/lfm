# collates all the files created by datagen into a nice ML dataset
from functools import partial
import os
import sys
from traceback import print_exc
import numpy as np
from common.metrics import calc_dataset_metrics
from sqlalchemy.orm import Session
from common.db import (
    exec_sql,
    get_db_gs_path,
    get_engine,
    get_target_id_by_name,
    PMFDataset as DBPMFDataset,
)
from common.gs import GS_FS
from common.tanimoto import batch_tanimoto_faster, get_tanimoto_matrix
from common.utils import get_fp, get_output_dir
from datagen.pmf_datagen import PMFSimResult, get_datagen_for_target
from openmm import unit
from scipy import sparse
from rdkit import Chem
import h5py
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import graph_tool as gt
from graph_tool import topology
from common.celery_app import celery_app

REPORT_INTERVAL = 10.0 * unit.picoseconds


def get_ligand_restraint_forces(lig_pos, restraint_com, k):
    """Version 1 of the ligand restraint force calculation."""

    deltas = lig_pos - restraint_com
    Nlig = deltas.shape[1]
    lig_restraint_F = -2 * k * deltas / Nlig
    return lig_restraint_F


def get_single_datapoint(
    gs_path, n_burnin, n_production, prod_lig_restraint_k, lig_restraint_version
):
    """Returns a tuple pd_row, h5_row if the folder contains
    a valid data point, otherwise None.
    """

    try:
        result = PMFSimResult.from_gs(gs_path)

        if np.isnan(result.forces).any():
            raise ValueError("Forces contain NaNs")

        if (np.abs(result.forces) > 1e4).any():
            raise ValueError("Forces are too large")

        if np.isnan(result.energies).any():
            raise ValueError("Energies contain NaNs")

        if np.isnan(result.lig_positions).any():
            raise ValueError("Positions contain NaNs")

        if np.isnan(result.lig_nbparams).any():
            raise ValueError("nb_params contain NaNs")

        mol = result.lig_rd
        # by default all atoms are Hs (smh some of the datapoints
        # have implicit Hs)
        elements = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)

        assert (
            len(result.lig_positions) == mol.GetNumAtoms()
        ), f"Number of atoms mismatch: {len(result.lig_positions)} vs {mol.GetNumAtoms()}"

        bonds = np.array(
            [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        )
        bond_types = np.array([b.GetBondTypeAsDouble() for b in mol.GetBonds()])

        formal_charges = np.array([a.GetFormalCharge() for a in mol.GetAtoms()])

        smi = Chem.MolToSmiles(mol)
        fp = get_fp(mol)

        final_index = (n_burnin + n_production) if n_production is not None else None
        prod_forces = result.forces[n_burnin:final_index]

        n_atoms = len(elements)
        force_cov = np.cov(
            result.forces[n_burnin:final_index].reshape(-1, n_atoms * 3).T
        )

        # if we used a restraint, keep use full array of
        # forces and MD positions (and ignore hessians)
        if prod_lig_restraint_k is None:
            mean_hessians = np.mean(result.hessians[n_burnin:final_index], axis=0)
            lig_pos = result.lig_positions
            prod_forces = prod_forces.mean(axis=0)
        else:
            assert lig_restraint_version == 1, "Only version 1 is supported right now"
            mean_hessians = None
            lig_pos = result.lig_md_positions[n_burnin:final_index]

            # we forgot to subtract off the ligand restraint forces in datagen
            # doing that now
            restraint_F = get_ligand_restraint_forces(
                lig_pos, result.restraint_com, prod_lig_restraint_k
            )
            prod_forces = prod_forces - restraint_F

        pd_row = {
            "id": gs_path.split("/")[-1],
            "smiles": smi,
        }

        h5_row = {
            "positions": lig_pos,
            "forces": prod_forces,
            "nb_params": result.lig_nbparams,
            "elements": elements,
            "formal_charges": formal_charges,
            "bonds": bonds,
            "bond_types": bond_types,
            "force_cov": force_cov,
            "fingerprint": fp,
        }
        if mean_hessians is not None:
            h5_row["hessians"] = mean_hessians

        if prod_lig_restraint_k is not None:
            h5_row["lig_restraint_k"] = prod_lig_restraint_k
            h5_row["lig_restraint_com"] = result.restraint_com

        return pd_row, h5_row
    except KeyboardInterrupt:
        raise
    except:
        print("Error in", gs_path)
        print_exc()
        return None


def get_datagen_dir(datagen, burnin_time, production_time) -> str:
    """Get the directory where data generation files are stored."""
    prod_suffix = (
        f"_{production_time.value_in_unit(unit.nanoseconds)}"
        if production_time is not None
        else ""
    )
    ret = os.path.join(
        get_output_dir(),
        "datagens",
        str(datagen.id),
        str(burnin_time.value_in_unit(unit.nanoseconds)) + prod_suffix,
    )
    os.makedirs(ret, exist_ok=True)
    return ret


def collate_md_data(datagen, burnin_time, production_time, n_workers=8):
    """Locally creates the h5 and csv files for the given datagen"""

    datagen_dir = get_datagen_dir(datagen, burnin_time, production_time)

    folders = GS_FS.ls(datagen.get_gs_path(), detail=False)
    folders = [f"gs://{folder}" for folder in folders]
    print(f"Collating {len(folders)} datapoints")

    df_fname = os.path.join(datagen_dir, "all.csv")
    h5_fname = os.path.join(datagen_dir, "all.h5")

    df = pd.DataFrame()

    if os.path.exists(h5_fname):
        os.remove(h5_fname)

    h5_file = h5py.File(h5_fname, "w")
    # since all fps are the same shape, we can create a single dataset
    h5_file.create_dataset(
        "fingerprint", maxshape=(None, 2048), shape=(0, 2048), dtype=bool
    )

    n_burnin = int(burnin_time / REPORT_INTERVAL)
    n_production = (
        int(production_time / REPORT_INTERVAL) if production_time is not None else None
    )
    single_datapoint_fn = partial(
        get_single_datapoint,
        n_burnin=n_burnin,
        n_production=n_production,
        prod_lig_restraint_k=datagen.prod_ligand_restraint_k,
        lig_restraint_version=datagen.ligand_restraint_version,
    )

    with Pool(n_workers) as p:
        for result in tqdm(p.imap(single_datapoint_fn, folders), total=len(folders)):
            # for folder in tqdm(folders):
            # result = single_datapoint_fn(folder)
            if result is None:
                continue
            pd_row, h5_row = result

            df = pd.concat([df, pd.DataFrame([pd_row])], axis=0)

            group = h5_file.create_group(pd_row["id"])
            for key, value in h5_row.items():
                group.create_dataset(key, data=value)

            h5_file["fingerprint"].resize((len(h5_file["fingerprint"]) + 1, 2048))
            h5_file["fingerprint"][-1] = h5_row["fingerprint"]

    print("Successfully collated", len(df), "datapoints")

    df.to_csv(df_fname, index=False)


def get_active_fps(engine, target_id):
    """Returns the fingerprints of the active molecules for the given target"""
    query = f"""
    SELECT mol.mol FROM
    molecules mol
    JOIN activities act ON act.mol_id = mol.id
    WHERE act.target_id = {target_id}
    """
    df = exec_sql(engine, query)
    return np.array([get_fp(Chem.MolFromSmiles(mol)) for mol in df.mol])


def get_dataset_tanimoto_matrix(datagen_dir, fps, force=False):
    """Get the tanimoto similarity matrix for the dataset"""

    mat_file = os.path.join(datagen_dir, "tanimoto_matrix.npz")
    if os.path.exists(mat_file) and not force:
        return sparse.load_npz(mat_file)

    mat = get_tanimoto_matrix(fps)
    sparse.save_npz(mat_file, mat)
    return mat


def get_active_max_tanimoto(datagen_dir, fps, act_fps, force=False):
    """Get the maximum tanimoto similarity between the ligands
    and any of the known binders.
    """

    out_fname = os.path.join(datagen_dir, "act_tanimoto.npz")
    if os.path.exists(out_fname) and not force:
        return np.load(out_fname)["arr_f0"]

    if len(act_fps) == 0:
        return np.zeros(len(fps))

    act_tanimoto = np.zeros((len(fps), len(act_fps)))
    for i, act_fp in enumerate(tqdm(act_fps)):
        act_tanimoto[:, i] = batch_tanimoto_faster(act_fp, fps)

    max_tanimoto = np.max(act_tanimoto, axis=1)
    np.savez(out_fname, max_tanimoto)
    return max_tanimoto


def get_dataset_splits(datagen_dir, fps, act_fps, has_test, cutoff=0.4, force=False):
    """Return a mapping from split to dataset indices. We split by tanimoto
    similarity cluster according to cutoff, and ensure that all the datapoints
    in the same cluster as the actives are in the test set
    """

    split_fname = os.path.join(datagen_dir, "splits.npz")
    if os.path.exists(split_fname) and not force:
        return dict(np.load(split_fname))

    tan_mat = get_dataset_tanimoto_matrix(datagen_dir, fps, force=force)

    act_tanimoto = get_active_max_tanimoto(datagen_dir, fps, act_fps, force=force)
    act_mask = act_tanimoto > cutoff

    tan_mask = tan_mat.data > cutoff
    tan_graph = gt.Graph(
        list(zip(tan_mat.row[tan_mask], tan_mat.col[tan_mask])), directed=False
    )

    labels, *rest = topology.label_components(tan_graph)
    labels = np.array(labels.a)
    num_labels = labels.max() + 1

    # splits 0-7 are train, 8 is val, 9 is test
    label_to_split = np.random.randint(0, 10, num_labels)

    # put molecules similar to act in the test set
    act_labels = np.unique(labels[act_mask])
    label_to_split[act_labels] = 9

    train_labels = np.where(np.in1d(label_to_split, np.arange(8)))[0]
    val_labels = np.where(label_to_split == 8)[0]
    test_labels = np.where(label_to_split == 9)[0]

    if not has_test:
        train_labels = np.concatenate([train_labels, test_labels])
        test_labels = np.array([], dtype=int)

    # mapping from split to data indices
    splits = {
        "train": np.where(np.in1d(labels, train_labels))[0],
        "val": np.where(np.in1d(labels, val_labels))[0],
        "test": np.where(np.in1d(labels, test_labels))[0],
    }

    print("Finished creating splits. Split fractions:")
    for split, idxs in splits.items():
        split_frac = len(idxs) / len(labels)
        print(f"  {split}: {split_frac:.2f}")

    # save the splits
    np.savez(split_fname, **splits)

    return splits


def split_dataset(engine, datagen, burnin_time, production_time, has_test, force=False):
    """Split the dataset (assumes it's already collated)"""

    datagen_dir = get_datagen_dir(datagen, burnin_time, production_time)

    df_fname = os.path.join(datagen_dir, "all.csv")
    h5_fname = os.path.join(datagen_dir, "all.h5")

    df = pd.read_csv(df_fname)

    h5_file = h5py.File(h5_fname, "r")
    fps = h5_file["fingerprint"][:]
    act_fps = get_active_fps(engine, datagen.target_id)

    splits = get_dataset_splits(datagen_dir, fps, act_fps, has_test, force=force)

    for split, indices in splits.items():

        split_df_fname = os.path.join(datagen_dir, f"{split}.csv")
        split_h5_fname = os.path.join(datagen_dir, f"{split}.h5")

        print(f"Creating {split}, saving to {split_df_fname} and {split_h5_fname}")

        if os.path.exists(split_h5_fname):
            os.remove(split_h5_fname)

        split_df = df.iloc[indices]
        split_df.to_csv(split_df_fname, index=False)

        split_h5_file = h5py.File(split_h5_fname, "w")

        split_h5_file.create_dataset("fingerprint", data=fps[indices])

        for id in tqdm(split_df.id):
            in_group = h5_file[id]
            out_group = split_h5_file.create_group(id)
            for key in in_group.keys():
                out_group.create_dataset(key, data=in_group[key])


def upload_dataset(
    engine, datagen, dataset_name, burnin_time, production_time, has_test
):
    """Upload the dataset to the database and google storage"""

    datagen_dir = get_datagen_dir(datagen, burnin_time, production_time)
    all_df = pd.read_csv(os.path.join(datagen_dir, "all.csv"))

    burnin_time_ns = burnin_time.value_in_unit(unit.nanoseconds)
    production_time_ns = (
        production_time.value_in_unit(unit.nanoseconds)
        if production_time is not None
        else None
    )

    with Session(engine) as session:
        dataset = DBPMFDataset(
            datagen_id=datagen.id,
            target_id=datagen.target_id,
            size=len(all_df),
            name=dataset_name,
            burnin_time=burnin_time_ns,
            production_time=production_time_ns,
            has_test_split=has_test,
        )
        session.add(dataset)
        session.commit()
        dataset_id = dataset.id

    gs_folder = get_db_gs_path("pmf_datasets", dataset_id)

    for split in ["all", "train", "val", "test"]:
        for suffix in ["csv", "h5"]:
            local_path = os.path.join(datagen_dir, f"{split}.{suffix}")
            gs_path = f"{gs_folder}/{split}.{suffix}"
            GS_FS.put(local_path, gs_path)

    # now upload the metrics
    metrics_path = os.path.join(gs_folder, "metrics.npz")
    dataset_metrics = calc_dataset_metrics(dataset_id)
    np.savez(GS_FS.open(metrics_path, "wb"), **dataset_metrics)


def create_dataset(
    engine,
    target_id,
    datagen_name,
    sim_name,
    dataset_name,
    burnin_time,
    production_time,
    has_test,
):
    """Create the dataset for the given target and burnin time"""

    burnin_time_ns = burnin_time.value_in_unit(unit.nanoseconds)
    production_time_ns = (
        production_time.value_in_unit(unit.nanoseconds)
        if production_time is not None
        else None
    )
    datagen = get_datagen_for_target(engine, datagen_name, target_id, sim_name)

    with Session(engine) as session:
        # first check to see if the dataset already exists
        dataset = (
            session.query(DBPMFDataset)
            .filter(
                DBPMFDataset.datagen_id == datagen.id,
                DBPMFDataset.burnin_time == burnin_time_ns,
                DBPMFDataset.production_time == production_time_ns,
                DBPMFDataset.has_test_split == has_test,
                DBPMFDataset.name == dataset_name,
            )
            .first()
        )
        if dataset is not None:
            raise ValueError(f"Dataset already exists with name {dataset.name}")

    collate_md_data(datagen, burnin_time, production_time)
    split_dataset(engine, datagen, burnin_time, production_time, has_test)
    upload_dataset(
        engine, datagen, dataset_name, burnin_time, production_time, has_test
    )


@celery_app.task(
    worker_prefetch_multiplier=1,
    name="collate_dataset_task",
)
def collate_dataset_task(
    target_id,
    datagen_name,
    sim_name,
    dataset_name,
    burnin_time,
    production_time,
    has_test,
):
    engine = get_engine()
    create_dataset(
        engine,
        target_id,
        datagen_name,
        sim_name,
        dataset_name,
        burnin_time * unit.nanoseconds,
        production_time * unit.nanoseconds if production_time is not None else None,
        has_test,
    )


if __name__ == "__main__":
    engine = get_engine()
    action = sys.argv[1]
    target_name = sys.argv[2]
    datagen_name = sys.argv[3]
    sim_name = sys.argv[4] if sys.argv[4] != "None" else None
    dataset_name = sys.argv[5]
    burnin_time = float(sys.argv[6])
    # production_time = (
    #     float(sys.argv[7]) * unit.nanoseconds if len(sys.argv) > 7 else None
    # )
    production_time = None
    target_id = get_target_id_by_name(engine, target_name)

    has_test = not ("--no-test" in sys.argv)

    match action:
        case "run":
            create_dataset(
                engine,
                target_id,
                datagen_name,
                sim_name,
                dataset_name,
                burnin_time * unit.nanoseconds,
                (
                    production_time * unit.nanoseconds
                    if production_time is not None
                    else None
                ),
                has_test,
            )
        case "queue":
            collate_dataset_task.delay(
                target_id,
                datagen_name,
                sim_name,
                dataset_name,
                burnin_time,
                production_time,
                has_test,
            )
        case _:
            raise ValueError(f"Unknown action {action}")
