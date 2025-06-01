""" Utilities for dealing with PLINDER """

from collections import defaultdict
from copy import deepcopy
import os
import shutil
from traceback import print_exc
import pandas as pd
import numpy as np
from common.alignment import find_rigid_alignment
import openmm as mm
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from fsspec.implementations.zip import ZipFileSystem
from openmm import app
from openmm import unit
import graph_tool as gt
from graph_tool import topology
from Bio import pairwise2
import h5py
from common.tanimoto import get_tanimoto_matrix
from common.utils import (
    CONFIG,
    get_fp,
    get_output_dir,
    get_sequence,
    load_smi,
    protonate_mol,
    save_smi,
)


def get_rec_structure(sys_id):
    """Returns a Modeller object so we can manage without
    unzipping the whole directory like the above function."""

    # sys_id = entry.system_id
    dir_id = sys_id[1:3]

    sys_dir = os.path.join(CONFIG.plinder_dir, "systems")

    # if we've already unzipped the file, we can just use that
    unzipped_file = os.path.join(sys_dir, dir_id, sys_id, "receptor.cif")
    if os.path.exists(unzipped_file):
        pdb = app.PDBxFile(unzipped_file)
        return app.Modeller(
            pdb.topology,
            np.array(pdb.positions.value_in_unit(unit.nanometer)) * unit.nanometer,
        )

    zip_file = os.path.join(sys_dir, dir_id + ".zip")

    fs = ZipFileSystem(zip_file)
    with fs.open(f"{sys_id}/receptor.cif", "r") as f:
        pdb = app.PDBxFile(f)
        ret = app.Modeller(
            pdb.topology,
            np.array(pdb.positions.value_in_unit(unit.nanometer)) * unit.nanometer,
        )

    return ret


def get_lig_mol(sys_id, lig_id, protonate=False):
    """Returns an RDKit object so we can manage without unzipping"""

    # sys_id = entry.system_id
    dir_id = sys_id[1:3]

    sys_dir = os.path.join(CONFIG.plinder_dir, "systems")
    zip_file = os.path.join(sys_dir, dir_id + ".zip")

    fs = ZipFileSystem(zip_file)
    with fs.open(f"{sys_id}/ligand_files/{lig_id}.sdf", "rb") as f:
        rd_mol = next(iter(Chem.ForwardSDMolSupplier(f, removeHs=False)))
        if protonate:
            rd_mol = protonate_mol(rd_mol)

    return rd_mol


def get_plinder_for_rec(uniprot):
    """Get the subset of the PLINDER database that contains
    the current protein."""

    plinder_fname = os.path.join(
        CONFIG.plinder_dir, "index/annotation_table_nonredundant.parquet"
    )
    plinder = pd.read_parquet(plinder_fname)
    plinder = plinder[plinder["system_pocket_UniProt"] == uniprot]

    return plinder


def get_plinder_benchmark_dir(uniprot):
    ret = get_output_dir() + "/" + uniprot
    os.makedirs(ret, exist_ok=True)
    return ret


def get_plinder_ligands(uniprot, force=False):
    """Get all ligands that are known to bind to the current
    protein from the PLINDER database.
    """

    smi_file = os.path.join(get_plinder_benchmark_dir(uniprot), "plinder_ligands.smi")
    if os.path.exists(smi_file) and not force:
        return load_smi(smi_file)

    plinder = get_plinder_for_rec(uniprot)

    smi = list(plinder.ligand_rdkit_canonical_smiles.dropna().unique())
    save_smi(smi_file, smi)
    return smi


def get_plinder_fps(uniprot, force=False):
    """Get the fingerprints of all the ligands that are known to
    bind to the current protein from the PLINDER database.
    """

    fps_file = os.path.join(get_plinder_benchmark_dir(uniprot), "plinder_fps.npy")
    if os.path.exists(fps_file) and not force:
        return np.load(fps_file)
    smi = get_plinder_ligands(force=force)

    fps = np.array([get_fp(Chem.MolFromSmiles(smi)) for smi in smi])
    np.save(fps_file, fps)
    return fps


def get_full_dataset_df(uniprot, force=False):
    """Get the full (unfiltered) dataset dataframe"""

    fname = os.path.join(get_plinder_benchmark_dir(uniprot), "all_unfiltered.csv")
    if os.path.exists(fname) and not force:
        return pd.read_csv(fname)

    pl_df = get_plinder_for_rec(uniprot)

    mask = []
    seqs = []
    for i, row in tqdm(pl_df.iterrows(), total=len(pl_df)):
        try:
            rec = get_rec_structure(row.system_id)
            seqs.append(get_sequence(rec)[0])
            mask.append(True)
        except KeyboardInterrupt:
            raise
        except ValueError:
            print_exc()
            mask.append(False)

    pl_df = pl_df.iloc[mask]
    out_df = pd.DataFrame(
        {
            "sys_id": pl_df.system_id,
            "lig_id": pl_df.system_ligand_chains,
            "resolution": pl_df.entry_resolution,
            "rec_seq": seqs,
            "lig_smiles": pl_df.ligand_rdkit_canonical_smiles,
            "pK": pl_df.ligand_binding_affinity,
        }
    )

    out_df.to_csv(fname, index=False)

    return out_df


def filter_mw(df, min_weight=100, max_weight=600):
    """Filter out ligands that are too small."""
    indices = []
    for i, smi in enumerate(df.lig_smiles):
        mol = Chem.MolFromSmiles(smi)
        mw = Descriptors.MolWt(mol)
        if mw > min_weight and mw < max_weight:
            indices.append(i)
    return df.iloc[indices]


def same_seq_gaps(s1, s2):
    """Returns true iff the sequences are the same except for gaps"""
    alignment = pairwise2.align.globalxx(s1, s2)[0]
    for res1, res2 in zip(alignment[0], alignment[1]):
        if res1 == "-" or res2 == "-":
            continue
        if res1 != res2:
            return False
    return True


def filter_rec_seqs(df, max_len=None, allow_gaps=False):
    """Finds the minimum common sequence in the dataset and filters out
    everything that doesn't match it. Also filters out sequences that are
    too long"""

    seq2indices = defaultdict(list)
    for seq in tqdm(df.rec_seq.unique()):
        for i, seq2 in enumerate(df.rec_seq):
            if max_len is not None and len(seq2) > max_len:
                continue
            if seq in seq2:
                seq2indices[seq].append(i)
            elif allow_gaps:
                if same_seq_gaps(seq, seq2):
                    seq2indices[seq].append(i)

    # get the most common sequence
    rec_seq = max(seq2indices, key=lambda x: len(seq2indices[x]))
    good_indices = seq2indices[rec_seq]

    return df.iloc[good_indices]


def remove_duplicates(df):
    """Consolidates duplicate lig_smiles by choosing the one with the highest resolution"""

    df["has_pK"] = ~df.pK.isna()
    df = df.sort_values("has_pK", ascending=False)
    df = df.drop_duplicates("lig_smiles", keep="first")
    del df["has_pK"]
    return df


def get_curated_dataset_df(uniprot, min_mw=100, max_mw=600, force=False):
    """Get the curated dataset dataframe"""

    fname = os.path.join(get_plinder_benchmark_dir(uniprot), "all_curated.csv")
    if os.path.exists(fname) and not force:
        return pd.read_csv(fname)

    df = get_full_dataset_df(uniprot)  # , force=force)
    print("Initial length: ", len(df))
    df = filter_mw(df, min_mw, max_mw)
    print("Length after MW filtering: ", len(df))

    df = filter_rec_seqs(df)
    print("Length after sequence filtering: ", len(df))

    df = remove_duplicates(df)
    print("Length after duplicate removal: ", len(df))

    df = df.reset_index(drop=True)
    df.to_csv(fname, index=False)

    return df


def split_benchmark(df, cutoff=0.3, force=False):
    """Split the dataset according to tanimoto similarity"""

    split_names = ["val", "test"]
    split_fnames = {
        split: os.path.join(get_plinder_benchmark_dir(), f"{split}.csv")
        for split in split_names
    }

    if all([os.path.exists(fname) for fname in split_fnames.values()]) and not force:
        return {split: pd.read_csv(fname) for split, fname in split_fnames.items()}

    fps = get_plinder_fps()
    all_smi = get_plinder_ligands()

    fp_indices = [all_smi.index(smi) for smi in df.lig_smiles]
    fps = fps[fp_indices]

    tan_mat = get_tanimoto_matrix(fps)
    tan_mask = tan_mat.data > cutoff

    tan_graph = gt.Graph(
        list(zip(tan_mat.row[tan_mask], tan_mat.col[tan_mask])), directed=False
    )

    labels, *rest = topology.label_components(tan_graph)
    labels = np.array(labels.a)
    num_labels = labels.max() + 1
    label_to_split = np.random.randint(0, 2, num_labels)

    val_labels = np.where(label_to_split[labels] == 0)[0]
    test_labels = np.where(label_to_split[labels] == 1)[0]

    # mapping from split to data indices
    splits = {
        "val": np.where(np.in1d(labels, val_labels))[0],
        "test": np.where(np.in1d(labels, test_labels))[0],
    }

    for split in split_names:
        w_ba = (~np.isnan(df.iloc[splits[split]].pK)).sum()
        print(f"{split} size: {len(splits[split])} (w/ binding affinity: {w_ba})")

    ret = {}
    for split in split_names:
        split_df = df.iloc[splits[split]]
        ret[split] = split_df
        split_df.to_csv(split_fnames[split], index=False)

    return ret


def get_residue_mapping(seq, idx, ref_seq, ref_idx):
    """Return mapping from protein1 and protein2 residues and vice versa"""
    alignment = pairwise2.align.globalxx(seq, ref_seq)[0]

    # mapping between r1 residues and r2 residues
    rec2ref = {}
    ref2rec = {}
    s1 = 0
    s2 = 0
    for res1, res2 in zip(alignment[0], alignment[1]):
        if res1 != "-" and res2 != "-":
            rec2ref[idx[s1]] = ref_idx[s2]
            ref2rec[ref_idx[s2]] = idx[s1]
        if res1 != "-":
            s1 += 1
        if res2 != "-":
            s2 += 1

    return rec2ref, ref2rec


def get_pocket_residues(
    uniprot,
    df,
    ref_index,
    rec_ref,
    cutoff_dist=4,
    min_jaccard=0.4,  # pocket must have at least 40% overlap with reference pocket
    overlap_cutoff=2,  # but if it adds less than 2 residues, we don't care (may just be small)
    force=False,
):
    """Once we've found the reference receptor, get a list of pocket indices
    mapped to its sequence"""

    fname = os.path.join(get_plinder_benchmark_dir(uniprot), "pocket_residues.npy")
    df_csv = os.path.join(get_plinder_benchmark_dir(uniprot), "pocket_filtered.csv")
    if os.path.exists(fname) and os.path.exists(df_csv) and not force:
        return pd.read_csv(df_csv), np.load(fname)

    ref_seq, ref_idx = get_sequence(rec_ref)

    all_poc_residues = set()
    ref_poc_residues = None

    # make sure we start from the reference receptor
    all_indices = list(df.index)
    all_indices.remove(ref_index)

    good_indices = []
    for i, row in tqdm(df.loc[[ref_index] + all_indices].iterrows(), total=len(df)):
        rec = get_rec_structure(row.sys_id)
        lig = get_lig_mol(row.sys_id, row.lig_id)

        seq, idx = get_sequence(rec)
        rec2ref, ref2rec = get_residue_mapping(seq, idx, ref_seq, ref_idx)

        lig_pos = lig.GetConformer().GetPositions()
        rec_pos = rec.positions.value_in_unit(unit.angstrom)

        dist_mat = np.linalg.norm(lig_pos[:, np.newaxis] - rec_pos, axis=-1)
        lig_atoms, rec_atoms = np.where(dist_mat < cutoff_dist)

        atom2residue = {
            atom.index: residue.index
            for residue in rec.topology.residues()
            for atom in residue.atoms()
        }
        pocket_residues = {atom2residue[atom] for atom in rec_atoms}
        ref_residues = {rec2ref[res] for res in pocket_residues if res in rec2ref}

        if ref_poc_residues is None:
            ref_poc_residues = ref_residues
        else:
            to_add = len(ref_residues - ref_poc_residues)
            overlap = len(ref_residues & ref_poc_residues)
            total = len(ref_residues | ref_poc_residues)
            jaccard = overlap / total
            if (min_jaccard is not None and jaccard < min_jaccard) and (
                overlap_cutoff is not None and to_add > overlap_cutoff
            ):
                print(f"Skipping {row.sys_id} {row.lig_id} as pocket is too different")
                continue

        all_poc_residues.update(ref_residues)
        good_indices.append(i)

    ref_poc_residues = np.array(list(ref_poc_residues))
    np.save(fname, ref_poc_residues)

    good_df = df.loc[good_indices]
    good_df.to_csv(df_csv, index=False)

    return good_df, ref_poc_residues


def residue_to_alpha_indices(rec, res_indices):
    """Get the alpha carbon indices of the residues"""
    ret = []
    all_residues = list(rec.topology.residues())
    for idx in res_indices:
        for atom in all_residues[idx].atoms():
            if atom.name == "CA":
                ret.append(atom.index)
                break
    return np.array(ret)


def save_aligned_structures(
    uniprot, df, rec_ref, ref_poc_residues, force=False, max_poc_dist=15 * unit.angstrom
):
    """Save the aligned structures for each protein and ligand in the dataset. Returns
    final df after all the failed entries are removed. Also saves a random conformer for
    each ligand, and filters out all ligands that are too far away from the pocket center
    """

    out_fname = os.path.join(get_plinder_benchmark_dir(uniprot), "all.csv")
    if os.path.exists(out_fname) and not force:
        return pd.read_csv(out_fname)

    ref_seq, ref_idx = get_sequence(rec_ref)
    # ref_poc_indices = residue_to_alpha_indices(rec_ref, ref_poc_residues)
    # ref_poc_pos = rec_ref.positions[ref_poc_indices].value_in_unit(unit.nanometer)

    out_folder = os.path.join(get_plinder_benchmark_dir(uniprot), "structures")
    shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder, exist_ok=True)

    good_indices = []
    for row_id, row in tqdm(df.iterrows(), total=len(df)):
        try:
            rec = get_rec_structure(row.sys_id)

            lig = get_lig_mol(row.sys_id, row.lig_id, protonate=True)
            lig_pos = lig.GetConformer().GetPositions() * unit.angstrom

            seq, idx = get_sequence(rec)

            rec2ref, ref2rec = get_residue_mapping(seq, idx, ref_seq, ref_idx)

            rec_poc_residues = [ref2rec[i] for i in ref_poc_residues if i in ref2rec]
            rec_poc_indices = residue_to_alpha_indices(rec, rec_poc_residues)

            cur_poc_residues = [rec2ref[i] for i in rec_poc_residues]
            cur_poc_indices = residue_to_alpha_indices(rec_ref, cur_poc_residues)
            ref_poc_pos = rec_ref.positions[cur_poc_indices].value_in_unit(
                unit.nanometer
            )

            rec_pos = rec.positions.value_in_unit(unit.nanometer)
            rec_poc_pos = rec.positions[rec_poc_indices].value_in_unit(unit.nanometer)
            rec_poc_com = np.mean(rec_poc_pos, axis=0)
            ref_poc_com = np.mean(ref_poc_pos, axis=0)

            R, t = find_rigid_alignment(
                rec_poc_pos - rec_poc_com, ref_poc_pos - ref_poc_com
            )

            rec_pos_aligned = np.dot(rec_pos - rec_poc_com, R.T) + ref_poc_com
            rec_aligned = rec
            rec_aligned.positions = rec_pos_aligned * unit.nanometer

            lig_pos = lig_pos.value_in_unit(unit.nanometer)
            lig_pos_aligned = np.dot(lig_pos - rec_poc_com, R.T) + ref_poc_com
            lig_aligned = deepcopy(lig)
            conf = lig_aligned.GetConformer()
            for i, pos in enumerate(lig_pos_aligned):
                # convert to angstrom
                conf.SetAtomPosition(i, [float(x * 10) for x in pos])

            # filter out ligands that are too far away from the pocket cente#
            lig_poc_dist = np.linalg.norm(lig_pos_aligned - ref_poc_com, axis=1)
            if max_poc_dist is not None and np.max(
                lig_poc_dist
            ) > max_poc_dist.value_in_unit(unit.nanometer):
                print(
                    f"Skipping {row.sys_id} {row.lig_id} as ligand is too far from pocket"
                )
                continue

            # embed molecule

            lig_rd = deepcopy(lig)
            lig_rd.RemoveAllConformers()
            AllChem.EmbedMolecule(lig_rd)
            AllChem.UFFOptimizeMolecule(lig_rd, maxIters=1000)

            rec_fname = os.path.join(out_folder, f"rec_{row.sys_id}.pdb")
            lig_fname = os.path.join(out_folder, f"lig_{row.sys_id}_{row.lig_id}.sdf")
            rand_fname = os.path.join(
                out_folder, f"rand_lig_{row.sys_id}_{row.lig_id}.sdf"
            )

            # save the aligned structures

            app.PDBFile.writeFile(
                rec_aligned.topology, rec_aligned.positions, open(rec_fname, "w")
            )

            with open(lig_fname, "w") as f:
                f.write(Chem.MolToMolBlock(lig_aligned))

            with open(rand_fname, "w") as f:
                f.write(Chem.MolToMolBlock(lig_rd))

            good_indices.append(row_id)

        except KeyboardInterrupt:
            raise
        except:
            print_exc()

    good_df = df.loc[good_indices]
    good_df.to_csv(out_fname, index=False)

    return good_df.reset_index(drop=True)
