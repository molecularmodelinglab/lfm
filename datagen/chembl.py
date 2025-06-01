# Get the actives and property-matched decoys from ChEMBL

from collections import defaultdict, namedtuple
from functools import partial
from multiprocessing import Pool
from traceback import print_exc
from itertools import product
import os
import pickle
import sqlite3
import struct
import subprocess
import numpy as np
import pandas as pd
import h5py
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from openmm import unit
import openmm as mm
from tqdm import tqdm
import graph_tool as gt
from graph_tool.topology import max_independent_vertex_set

from common.tanimoto import batch_tanimoto_faster, get_tanimoto_matrix, tanimoto
from common.utils import CONFIG, get_fp, get_nonbonded_params


def download_chembl():
    version = CONFIG.chembl.version
    dir = CONFIG.chembl_dir
    os.makedirs(dir, exist_ok=True)

    fname = f"{dir}/chembl_{version}/chembl_{version}_sqlite/chembl_{version}.db"

    if not os.path.exists(fname):
        url = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_{version}_sqlite.tar.gz"
        tar_fname = f"{dir}/chembl_{version}_sqlite.tar.gz"

        if not os.path.exists(tar_fname):
            print(f"Downloading ChEMBL {version}")
            os.system(f"wget {url} -O {tar_fname}")

        print(f"Extracting ChEMBL {version}")
        os.system(f"tar -xvf {tar_fname} -C {dir}")

    return fname


def get_chembl_con():
    return sqlite3.connect(download_chembl())


def protonate_smiles(smiles):
    """Use obabel to protonate the smiles at pH 7"""
    subprocess.run(
        f'obabel -:"{smiles}" -O output/protonated.smi -p 7',
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with open("output/protonated.smi") as f:
        protonated_smiles = f.read().strip()
    return protonated_smiles


def get_chembl_activities(con, target, force=False):
    query = f"""

    SELECT md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.potential_duplicate,
    COALESCE(act.data_validity_comment, 'valid') as data_validity_comment,
    a.confidence_score,
    a.description AS assay_description,
    td.chembl_id AS target_chembl_id,
    td.target_type,
    c.accession as protein_accession,
    a.chembl_id as assay_id,
    prop.alogp as alogp,
    prop.cx_logp as cx_logp,
    prop.hba as hba,
    prop.hbd as hbd,
    prop.rtb as rtb,
    prop.full_mwt as full_mwt

    FROM target_dictionary td
    JOIN assays a ON td.tid = a.tid
    JOIN activities act ON a.assay_id = act.assay_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN compound_properties prop ON md.molregno = prop.molregno
    JOIN target_type tt ON td.target_type = tt.target_type
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences c ON tc.component_id = c.component_id
    AND tt.target_type = 'SINGLE PROTEIN'
    AND act.pchembl_value IS NOT NULL
    AND c.accession = '{target}';"""

    out_fname = f"{CONFIG.chembl_dir}/{target}_{CONFIG.chembl.version}_all.csv"

    if not os.path.exists(out_fname) or force:
        print(f"Extracting ChEMBL activities for {target}")
        df = pd.read_sql_query(query, con)
        df.to_csv(out_fname, index=False)

    return pd.read_csv(out_fname)


def query_property_matched_mols(
    con,
    row,
    mwt_range=50,
    logp_range=0.5,
    hba_range=1,
    hbd_range=1,
    rtb_range=1,
    act_cutoff=5,
    max_mols=100,
):

    min_mwt = row["full_mwt"] - mwt_range
    max_mwt = row["full_mwt"] + mwt_range
    min_logp = row["alogp"] - logp_range
    max_logp = row["alogp"] + logp_range
    min_hba = row["hba"] - hba_range
    max_hba = row["hba"] + hba_range
    min_hbd = row["hbd"] - hbd_range
    max_hbd = row["hbd"] + hbd_range
    min_rtb = row["rtb"] - rtb_range
    max_rtb = row["rtb"] + rtb_range

    query = f"""
    SELECT md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.potential_duplicate,
    COALESCE(act.data_validity_comment, 'valid') as data_validity_comment,
    a.confidence_score,
    a.description AS assay_description,
    td.chembl_id AS target_chembl_id,
    td.target_type,
    c.accession as protein_accession,
    a.chembl_id as assay_id,
    prop.alogp as alogp,
    prop.cx_logp as cx_logp,
    prop.hba as hba,
    prop.hbd as hbd,
    prop.rtb as rtb,
    prop.full_mwt as full_mwt

    FROM target_dictionary td
    JOIN assays a ON td.tid = a.tid
    JOIN activities act ON a.assay_id = act.assay_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN compound_properties prop ON md.molregno = prop.molregno
    JOIN target_type tt ON td.target_type = tt.target_type
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences c ON tc.component_id = c.component_id
    AND tt.target_type = 'SINGLE PROTEIN'
    AND act.pchembl_value > {act_cutoff}
    AND prop.full_mwt >= {min_mwt}
    AND prop.full_mwt <= {max_mwt}
    AND prop.alogp >= {min_logp}
    AND prop.alogp <= {max_logp}
    AND prop.hba >= {min_hba}
    AND prop.hba <= {max_hba}
    AND prop.hbd >= {min_hbd}
    AND prop.hbd <= {max_hbd}
    AND prop.rtb >= {min_rtb}
    AND prop.rtb <= {max_rtb}

    LIMIT {max_mols};"""

    return pd.read_sql_query(query, con)


def query_all_molecules(
    con,
    act_cutoff=5,
    force=False,
    chunk_size=1000,
):
    """Find all molecules in the database that have a pchembl value greater than act_cutoff for a single protein target"""

    query = f"""
    SELECT md.chembl_id AS compound_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.potential_duplicate,
    COALESCE(act.data_validity_comment, 'valid') as data_validity_comment,
    a.confidence_score,
    a.description AS assay_description,
    td.chembl_id AS target_chembl_id,
    td.target_type,
    c.accession as protein_accession,
    a.chembl_id as assay_id,
    prop.alogp as alogp,
    prop.cx_logp as cx_logp,
    prop.hba as hba,
    prop.hbd as hbd,
    prop.rtb as rtb,
    prop.full_mwt as full_mwt

    FROM target_dictionary td
    JOIN assays a ON td.tid = a.tid
    JOIN activities act ON a.assay_id = act.assay_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN compound_properties prop ON md.molregno = prop.molregno
    JOIN target_type tt ON td.target_type = tt.target_type
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences c ON tc.component_id = c.component_id
    AND tt.target_type = 'SINGLE PROTEIN'
    AND act.pchembl_value > {act_cutoff};"""

    out_fname = f"{CONFIG.chembl_dir}/all_{act_cutoff}.csv"

    if not os.path.exists(out_fname) or force:
        print(f"Extracting ChEMBL activities for all molecules")
        # iteratively save to csv
        chunks = pd.read_sql_query(query, con, chunksize=chunk_size)
        for chunk in tqdm(chunks):
            chunk.to_csv(
                out_fname, mode="a", index=False, header=not os.path.exists(out_fname)
            )

    return pd.read_csv(out_fname)


def bsearch_range(ranges, value):
    """Binary search for a range that contains the value"""
    low = 0
    high = len(ranges)
    while low < high:
        mid = (low + high) // 2
        if ranges[mid][0] <= value <= ranges[mid][1]:
            return mid
        elif ranges[mid][0] > value:
            high = mid
        else:
            low = mid + 1
    return None

def get_property_df(smi_list):
    """Get the properties of a list of SMILES"""

    mols = [Chem.MolFromSmiles(smi) for smi in smi_list]
    mols = [mol for mol in mols if mol is not None]
    mwt = [Descriptors.MolWt(mol) for mol in mols]
    logp = [Descriptors.MolLogP(mol) for mol in mols]
    hba = [Descriptors.NumHAcceptors(mol) for mol in mols]
    hbd = [Descriptors.NumHDonors(mol) for mol in mols]
    rtb = [Descriptors.NumRotatableBonds(mol) for mol in mols]

    return pd.DataFrame(
        {
            "canonical_smiles": smi_list,
            "full_mwt": mwt,
            "alogp": logp,
            "hba": hba,
            "hbd": hbd,
            "rtb": rtb,
        }
    )

def get_all_propset_bins(full_df):
    """Get the ranges for the property bins used for catalogging decoys"""

    mwt_spans = np.linspace(10, 50, 3)  # 5)
    logp_spans = np.linspace(0.1, 1, 3)  # 5)
    hba_spans = [0, 1]  # , 1, 2]
    hbd_spans = [0, 1]  # , 1, 2]
    rtb_spans = [0, 1]  # , 1, 2]

    all_spans = {
        "full_mwt": mwt_spans,
        "alogp": logp_spans,
        "hba": hba_spans,
        "hbd": hbd_spans,
        "rtb": rtb_spans,
    }

    all_bins = {}

    for key, spans in all_spans.items():
        bins = []
        for span in spans:
            if span == 0:
                bounds = np.sort(full_df[key].unique())
            else:
                bounds = np.arange(full_df[key].min(), full_df[key].max(), span)
            bins.append([(x, x + span) for x in bounds])

        all_bins[key] = bins

    return all_bins


def get_all_propsets(row, all_bins):
    """Returns a list of all property sets that the current row belongs to"""

    row_bins = {}

    for key, binss in all_bins.items():
        row_bins[key] = []
        for i, bins in enumerate(binss):
            idx = bsearch_range(bins, row[key])
            if idx is not None:
                row_bins[key].append((i, idx))

    return [tuple(bins) for bins in product(*row_bins.values())]


def encode_propset(propset):
    """Encode a property set as bytes"""
    propset_flat = tuple([x for bins in propset for x in bins])
    prop_str = struct.pack("i" * len(propset_flat), *propset_flat)
    return prop_str


def decode_indices(encoded):
    """Decode the bytearray into a list of indices"""
    length = len(encoded) // 4
    indices = []
    for i in range(length):
        indices.append(struct.unpack("i", encoded[i * 4 : i * 4 + 4])[0])
    return indices


def get_propset_indices(full_df, act_cutoff=5, force=False):
    """Get a DBM db mapping property sets to indices in full_df. Used
    for generating decoys"""

    # dbm_fname = f"{CONFIG.chembl_dir}/propset_indices_{act_cutoff}.dbm"
    # db = dbm.open(dbm_fname, "c")
    # if "completed" in db:
    #     return db

    # for key in list(db.keys()):
    #     del db[key]

    cache_fname = f"{CONFIG.chembl_dir}/propset_indices_{act_cutoff}.pkl"
    if os.path.exists(cache_fname) and not force:
        with open(cache_fname, "rb") as f:
            db = pickle.load(f)
            return db

    db = {}

    all_bins = get_all_propset_bins(full_df)

    for i, row in tqdm(full_df.iterrows(), total=len(full_df)):
        propsets = get_all_propsets(row, all_bins)
        for propset in propsets:
            prop_key = encode_propset(propset)
            cur_val = db[prop_key] if prop_key in db else b""
            cur_val += struct.pack("i", i)
            db[prop_key] = cur_val

    # db["completed"] = b"1"

    with open(cache_fname, "wb") as f:
        pickle.dump(db, f)

    return db


def get_possible_decoys(row, propset_bins, propset_indices):
    """Yeilds indices of possible decoys for a given row. Futher filtering
    is needed after this"""

    seen = set()
    propsets = get_all_propsets(row, propset_bins)
    for propset in propsets:
        key = encode_propset(propset)
        if key in propset_indices:
            indices = decode_indices(propset_indices[key])
            for index in indices:
                if index not in seen:
                    seen.add(index)
                    yield index


def filter_activities(
    activities_unfiltered,
    min_mwt=200,
    max_mwt=600,
    min_logp=-1,
    max_logp=5,
    max_hba=10,
    max_hbd=5,
    max_rtb=10,
    assay_exclusions=[],
):

    # remove mixtures
    activities = activities_unfiltered[
        ~activities_unfiltered["canonical_smiles"].str.contains("\.")
    ].reset_index(drop=True)

    # remove anything chembl thinks could be sketchy
    activities = activities.query(
        "potential_duplicate == 0 and data_validity_comment == 'valid' and confidence_score >= 8 and standard_relation == '='"
    )

    assay_exclusions = assay_exclusions + ["mutant", "mutation"]
    # remove assays with "mutant" or "mutation" in description
    for exc in assay_exclusions:
        activities = activities[
            ~activities["assay_description"].str.contains(exc, case=False)
        ]

    # in order of preference
    allowed_types = ["Kd", "Ki", "IC50"]
    activities = activities[activities["standard_type"].isin(allowed_types)]

    # we don't have these values for everything after deduping so just drop em
    activities = activities.drop(
        columns=[
            "potential_duplicate",
            "data_validity_comment",
            "confidence_score",
            "target_chembl_id",
            "target_type",
            "assay_id",
        ]
    )

    # filter by properties
    activities = activities[
        (activities["full_mwt"] >= min_mwt)
        & (activities["full_mwt"] <= max_mwt)
        & (activities["alogp"] >= min_logp)
        & (activities["alogp"] <= max_logp)
        & (activities["hba"] <= max_hba)
        & (activities["hbd"] <= max_hbd)
        & (activities["rtb"] <= max_rtb)
    ]

    # dedupe -- some compounds have multiple activities. Prefer Kd > Ki > IC50
    # take median pchembl value within a type

    dup_indexes = activities.duplicated(keep=False, subset=["compound_chembl_id"])
    dup_df = activities[dup_indexes]

    dup_rows = defaultdict(list)
    for _, row in dup_df.iterrows():
        dup_rows[row["compound_chembl_id"]].append(row)

    deduped_rows = []
    for _, rows in dup_rows.items():
        rows = sorted(rows, key=lambda x: allowed_types.index(x["standard_type"]))
        deduped_row = rows[0].copy()
        deduped_row["pchembl_value"] = pd.Series(
            [
                row["pchembl_value"]
                for row in rows
                if row["standard_type"] == deduped_row["standard_type"]
            ]
        ).median()
        deduped_row["standard_units"] = "nM"
        deduped_row["standard_value"] = (10 ** -deduped_row["pchembl_value"]) * 1e9
        deduped_rows.append(deduped_row)

    deduped_df = pd.DataFrame(deduped_rows)
    activities = pd.concat([activities[~dup_indexes], deduped_df], sort=False)

    return activities.reset_index(drop=True)


def filter_all_mols(activities_unfiltered):
    """Filters the results of querying for all the active molecules"""

    # remove mixtures
    activities = activities_unfiltered[
        ~activities_unfiltered["canonical_smiles"].str.contains("\.")
    ].reset_index(drop=True)

    # remove anything chembl thinks could be sketchy
    activities = activities.query(
        "potential_duplicate == 0 and data_validity_comment == 'valid' and confidence_score >= 8 and standard_relation == '='"
    )

    # in order of preference
    allowed_types = ["Kd", "Ki", "IC50"]
    activities = activities[activities["standard_type"].isin(allowed_types)]

    # we don't have these values for everything after deduping so just drop em
    activities = activities.drop(
        columns=[
            "potential_duplicate",
            "data_validity_comment",
            "confidence_score",
            "target_chembl_id",
            "target_type",
            "assay_id",
        ]
    )

    # remove duplicates
    activities = activities.drop_duplicates(subset=["canonical_smiles"])

    return activities.reset_index(drop=True)


def sample_activities(df, tan_cutoff=0.4):
    """Subsamples the dataframe so that no two rows have a tanimoto similarity greater than tan_cutoff"""

    fps = np.array(
        [get_fp(Chem.MolFromSmiles(smi)) for smi in tqdm(df.canonical_smiles)]
    )
    tan_mat = get_tanimoto_matrix(fps)

    tan_mask = tan_mat.data > tan_cutoff
    tan_graph = gt.Graph(
        list(zip(tan_mat.row[tan_mask], tan_mat.col[tan_mask])), directed=False
    )

    diverse_mask = np.array(list(max_independent_vertex_set(tan_graph)), dtype=bool)

    return df[diverse_mask].reset_index(drop=True)


def add_protonated_smiles(df, max_len=None):

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        prot_smi = protonate_smiles(row["canonical_smiles"])
        mol = Chem.MolFromSmiles(prot_smi)
        # smh sometimes obabel fails
        if mol is None:
            continue

        row["protonated_smiles"] = prot_smi

        rows.append(row)
        if max_len is not None and i > max_len:
            break

    return pd.DataFrame(rows)


def get_final_benchmark_df(con, force=False):
    """Get the final benchmark dataframe after filtering and sampling"""

    target = CONFIG.datagen.uniprot
    out_fname = f"{CONFIG.chembl_dir}/{target}_{CONFIG.chembl.version}_final.csv"

    if os.path.exists(out_fname) and not force:
        return pd.read_csv(out_fname)

    df = get_chembl_activities(con, CONFIG.datagen.uniprot, force=False)
    df = filter_activities(df)
    df = sample_activities(df)
    df = add_protonated_smiles(df)

    df.to_csv(out_fname, index=False)

    return df


def get_decoy_df(
    con,
    target,
    df,
    target_decoys=50,
    tan_cutoff=0.3,
    decoy_tan_cutoff=0.3,
    force=False,
    force_propset=False,
    cache_suffix="",
):
    """Get the property-matched decoys for the benchmark"""

    out_fname = (
        f"{CONFIG.chembl_dir}/{target}_{CONFIG.chembl.version}{cache_suffix}_decoys.csv"
    )

    if os.path.exists(out_fname) and not force:
        decoy_df = pd.read_csv(out_fname)
        return decoy_df

    full_df = query_all_molecules(con)
    full_df = filter_all_mols(full_df)

    propset_bins = get_all_propset_bins(full_df)
    propset_indices = get_propset_indices(full_df, force=force_propset)

    all_ref_fps = np.asarray(
        [get_fp(Chem.MolFromSmiles(smiles)) for smiles in tqdm(df["canonical_smiles"])]
    )

    decoy_rows = []
    decoy_fps = []

    for i, row in tqdm(df.iterrows(), total=len(df)):

        ref_mol = Chem.MolFromSmiles(row["canonical_smiles"])
        ref_fp = get_fp(ref_mol)

        prot_ref = Chem.MolFromSmiles(row["protonated_smiles"])
        ref_charge = Chem.GetFormalCharge(prot_ref)

        counter = 0
        for idx in get_possible_decoys(row, propset_bins, propset_indices):
            smi = full_df.iloc[idx]["canonical_smiles"]

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            fp = get_fp(mol)
            tan = tanimoto(ref_fp, fp)

            if tan > tan_cutoff:
                continue

            full_tan = batch_tanimoto_faster(fp, all_ref_fps).max()
            if full_tan > tan_cutoff:
                continue

            prot_smi = protonate_smiles(smi)
            prot_mol = Chem.MolFromSmiles(prot_smi)
            if prot_mol is None:
                continue

            charge = Chem.GetFormalCharge(prot_mol)
            if charge != ref_charge:
                continue

            if len(decoy_fps) > 0:
                decoy_tan = batch_tanimoto_faster(fp, np.array(decoy_fps)).max()
                if decoy_tan > decoy_tan_cutoff:
                    continue

            decoy_row = full_df.iloc[idx].to_dict()
            decoy_row["protonated_smiles"] = prot_smi
            decoy_row["parent_index"] = i

            decoy_rows.append(decoy_row)
            decoy_fps.append(fp)

            counter += 1

            if counter >= target_decoys:
                break

    decoy_df = pd.DataFrame(decoy_rows)
    decoy_df.to_csv(out_fname, index=False)

    return decoy_df


def save_h5(folder, df, save_systems=True, dataset_len=None):
    """Save h5 file and structures + systems folders for the benchmark"""

    system_folder = f"{folder}/systems"
    struct_folder = f"{folder}/structures"
    os.makedirs(system_folder, exist_ok=True)
    os.makedirs(struct_folder, exist_ok=True)

    # save them in parellel in a pool with tqdm
    if save_systems:
        save_single = partial(
            save_single_structure, folder, system_folder, struct_folder
        )
        with Pool(16) as p:
            list(
                tqdm(
                    p.imap(save_single, enumerate(df.protonated_smiles)), total=len(df)
                )
            )

    h5_fname = f"{folder}/all.h5"
    if os.path.exists(h5_fname):
        os.remove(h5_fname)

    h5_file = h5py.File(h5_fname, "w")

    for i, smi in enumerate(tqdm(df.protonated_smiles)):

        sdf_fname = f"{struct_folder}/{i}.sdf"
        sys_file = f"{system_folder}/{i}.xml"

        if not os.path.exists(sdf_fname) or not os.path.exists(sys_file):
            continue

        lig = MolData.from_file(sdf_fname)

        system = mm.XmlSerializer.deserialize(open(sys_file).read())

        pos = lig.positions.value_in_unit(unit.nanometer)
        nb_params = get_nonbonded_params(system)

        elems = np.array([a.element.atomic_number for a in lig.topology.atoms()])

        rd_mol = lig.rd_mols[0]
        bonds = np.array(
            [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in rd_mol.GetBonds()]
        )
        bond_types = np.array([b.GetBondTypeAsDouble() for b in rd_mol.GetBonds()])

        formal_charges = np.array([a.GetFormalCharge() for a in rd_mol.GetAtoms()])

        group = h5_file.create_group(f"{i}")
        group.create_dataset("pos", data=pos)
        group.create_dataset("nb_params", data=nb_params)
        group.create_dataset("elements", data=elems)
        group.create_dataset("formal_charges", data=formal_charges)
        group.create_dataset("bonds", data=bonds)
        group.create_dataset("bond_types", data=bond_types)

        if dataset_len is not None and i >= dataset_len:
            break


def make_benchmark(force=False):
    """Get the benchmark dataframe and the property-matched decoys"""

    con = get_chembl_con()

    df = get_final_benchmark_df(con, force=force)
    decoy_df = get_decoy_df(con, df, force=force)

    target = CONFIG.datagen.uniprot
    act_folder = f"{CONFIG.chembl_dir}/{target}_{CONFIG.chembl.version}"
    os.makedirs(act_folder, exist_ok=True)
    decoy_folder = f"{CONFIG.chembl_dir}/{target}_{CONFIG.chembl.version}_decoys"
    os.makedirs(decoy_folder, exist_ok=True)

    print("Benchmark size: ", len(df))
    print("Decoy size: ", len(decoy_df))

    save_systems = False
    save_h5(act_folder, df, save_systems=save_systems)
    save_h5(decoy_folder, decoy_df, save_systems=save_systems)

    return df, decoy_df


if __name__ == "__main__":
    make_benchmark()
