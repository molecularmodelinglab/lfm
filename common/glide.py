import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools


def clean_glide_outputs(smi_file, docked_mae, out_folder, max_ligands=None):
    """Cleans up the glide output mae file, extracting the scores
    to a new csv file with the (minimum) glide score per compound to
    {output_folder}/results.csv and saves the numbered sdf files to the
    folder as well. E.g. 1.sdf, 2.sdf, etc.

    :param smi_file: smi file we gave to glide for input. This can
        also be a list of smiles
    :param docked_mae: mae file output by glide
    :param out_folder: folder to put the cleaned outputs in
    :param max_ligands: The number of ligands we actually docked
    (sometimes less than the total number)

    """

    os.makedirs(out_folder, exist_ok=True)

    # first convert the mae file to a big sdf file in out_folder
    big_sdf = out_folder + "/all_docked.sdf"
    cmd = f"$SCHRODINGER/utilities/structconvert {docked_mae} {big_sdf}"

    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    df = PandasTools.LoadSDF(big_sdf, removeHs=False)

    # load in the smiles
    if isinstance(smi_file, list):
        og_smiles = smi_file
    else:
        og_smiles = []
        with open(smi_file, "r") as f:
            for line in f:
                og_smiles.append(line.strip())
        if max_ligands is not None:
            og_smiles = og_smiles[:max_ligands]

    filenames = [None for _ in range(len(og_smiles))]

    # save individual sdf files
    scores_and_mols = defaultdict(list)
    for i, row in df.iterrows():
        # first row is the rec structure
        if i == 0:
            continue
        mol = row.ROMol

        # sometimes I'm docking things with only one ligand and without
        # any lig prep...
        if len(og_smiles) == 1:
            og_index = 0
        else:
            og_index = int(row.s_lp_Variant.split(":")[-1].split("-")[0]) - 1

        filename = f"{og_index}.sdf"
        filenames[og_index] = filename

        for key, val in row.items():
            mol.SetProp(key, str(val))

        score = float(row.r_i_docking_score)
        scores_and_mols[og_index].append((score, mol))

    # order according to increasing score
    for index, arr in scores_and_mols.items():
        filename = filenames[index]
        full_filename = out_folder + "/" + filename
        arr = list(sorted(arr, key=lambda x: x[0]))

        writer = Chem.SDWriter(full_filename)
        for score, mol in arr:
            writer.write(mol)
        writer.close()


def prep_rec(in_file, out_file):
    """Runs prepwizard to prepare the receptor file for docking."""
    cmd = f"$SCHRODINGER/utilities/prepwizard -NOJOBID {in_file} {out_file}"
    subprocess.run(cmd, shell=True, check=True)


def prep_ligs(in_file, out_file, in_format="smi"):
    """Runs ligprep (with -i 2 to generate pronation states!)"""

    # smh ligprep needs to run from the out directory
    out_dir = os.path.dirname(out_file)
    # cur_dir = os.getcwd()
    try:
        # os.chdir(out_dir)
        cmd = f"ligprep -i{in_format} {in_file} -omae {out_file} -i 2 -WAIT"
        subprocess.run(cmd, shell=True, check=True)
    finally:
        pass
        # os.chdir(cur_dir)


def make_grid(rec_mae, poc_center, poc_size, out_file, inner_scale=0.5):
    """Runs the grid generation script to create the grid file for docking."""

    in_file = out_file.replace(".zip", ".in")

    with open(in_file, "w") as f:
        f.write(
            f"""INNERBOX {int(poc_size[0]*inner_scale)}, {int(poc_size[1]*inner_scale)}, {int(poc_size[2]*inner_scale)}
ACTXRANGE {poc_size[0]}
ACTYRANGE {poc_size[1]}
ACTZRANGE {poc_size[2]}
OUTERBOX {poc_size[0]}, {poc_size[1]}, {poc_size[2]}
GRID_CENTER {poc_center[0]}, {poc_center[1]}, {poc_center[2]}
GRIDFILE {out_file}
RECEP_FILE {rec_mae}
"""
        )

    cmd = f"$SCHRODINGER/glide -NOJOBID {in_file}"
    subprocess.run(cmd, shell=True, check=True)

    os.remove(in_file)


def glide_dock(lig_file, grid_file, out_folder):
    """Runs glide to dock the ligands into the rec specified
    by the grid file. Places output in out_folder."""

    cur_dir = os.getcwd()

    os.makedirs(out_folder, exist_ok=True)
    in_file = os.path.join(out_folder, "dock.in")
    with open(in_file, "w") as f:
        f.write(
            f"""GRIDFILE {grid_file}
LIGANDFILE {lig_file}
"""
        )

    cmd = f"$SCHRODINGER/glide -NOJOBID -OVERWRITE {os.path.basename(in_file)}"

    os.chdir(out_folder)

    try:
        subprocess.run(cmd, shell=True, check=True)
    finally:
        os.chdir(cur_dir)

    os.remove(in_file)


def collate_glide_results(glide_csvs, out_file):
    """Takes the csvs from separate glide runs
    and collates them into a single csv that Hidden
    Gem can read"""

    smiles = []
    scores = []

    for glide_csv in tqdm(glide_csvs):
        df = pd.read_csv(glide_csv)
        smiles.extend(df["SMILES"])
        scores.extend(df["r_i_docking_score"])

    out_df = pd.DataFrame({"SMILES": smiles, "Score": scores})

    out_df.to_csv(out_file, index=False)


def plot_score_distribution(glide_csv, out_file):
    """Plots the score distribution of the glide results. Assumes
    the csv was created by collate_glide_results. (that is, it has
    a Score column rather than a r_i_docking_score column.)"""

    df = pd.read_csv(glide_csv)
    mask = df.Score < 0
    print(
        f"Found {mask.sum()} negative scores in the glide results ({100*mask.mean()}% of the results)"
    )

    fig, ax = plt.subplots()
    ax.hist(df["Score"][mask], bins=500)
    ax.set_xlabel("Glide Score")
    ax.set_ylabel("Frequency")
    fig.savefig(out_file)


def save_top_n(glide_csv, out_file, n):
    """Saves the top n ligands from the glide results to a new csv.
    Again, this assumes the csv was created by collate_glide_results."""

    df = pd.read_csv(glide_csv)
    top_n = df.nsmallest(n, "Score")
    top_n.to_csv(out_file, index=False)
