# Script to add a well-studied target to the database withh PLINDER and ChEMBL

import os

import numpy as np
from tqdm import tqdm
from common.db import (
    Activity,
    ActivityType,
    CoStructure,
    Decoy,
    Molecule,
    ProteinTarget,
    Target,
    TargetType,
    get_engine,
)
import pandas as pd
from common.utils import get_sequence
from datagen.chembl import (
    add_protonated_smiles,
    filter_activities,
    get_chembl_activities,
    get_chembl_con,
    get_decoy_df,
    get_property_df,
    sample_activities,
)
from datagen.plinder import (
    get_curated_dataset_df,
    get_plinder_benchmark_dir,
    get_pocket_residues,
    save_aligned_structures,
)
from datagen.setup import get_fixed_structure
from sqlalchemy.orm import Session
from openmm import app, unit
from rdkit import Chem

# disable rdkit warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def add_target(
    target_name,
    uniprot_id,
    min_jaccard=0.4,
    max_poc_dist=15 * unit.angstroms,
    min_mw=100,
    max_mw=600,
    overlap_cutoff=2,
):
    print(f"Adding target {target_name} with uniprot ID {uniprot_id}")

    plinder_df = get_curated_dataset_df(
        uniprot_id, min_mw=min_mw, max_mw=max_mw, force=False
    )
    ref_index = plinder_df.rec_seq.str.len().argmin()

    pdb, _, idchain, _ = plinder_df.sys_id.iloc[ref_index].split("__")
    chain = idchain.split(".")[1]

    # TODO: for GPCRs, use the OPM database rather than the PDB so that calling addMembrane() works correctly
    rec_ref = get_fixed_structure(pdb, chain)
    rec_seq = get_sequence(rec_ref)

    plinder_df, poc_residues = get_pocket_residues(
        uniprot_id,
        plinder_df,
        ref_index,
        rec_ref,
        min_jaccard=min_jaccard,
        overlap_cutoff=overlap_cutoff,
        force=False,
    )

    plinder_df = save_aligned_structures(
        uniprot_id,
        plinder_df,
        rec_ref,
        poc_residues,
        max_poc_dist=max_poc_dist,
        force=False,
    )

    out_folder = get_plinder_benchmark_dir(uniprot_id)
    ref_file = os.path.join(out_folder, f"ref.pdb")
    app.PDBFile.writeFile(rec_ref.topology, rec_ref.positions, ref_file)

    print("Total #structures: ", len(plinder_df))
    print("#structures with pK: ", len(plinder_df[~np.isnan(plinder_df.pK)]))

    con = get_chembl_con()
    chembl_df = get_chembl_activities(con, uniprot_id, force=False)
    chembl_df = filter_activities(chembl_df)
    chembl_df = sample_activities(chembl_df)
    chembl_df = add_protonated_smiles(chembl_df)
    print("Total #activities: ", len(chembl_df))

    decoy_df = get_decoy_df(con, uniprot_id, chembl_df)
    print("#decoys per active: ", len(decoy_df) // len(chembl_df))

    plinder_active_df = plinder_df  # [~np.isnan(plinder_df.pK)]
    plinder_prop_df = get_property_df(plinder_active_df.lig_smiles)
    plinder_prop_df = add_protonated_smiles(plinder_prop_df)
    n_decoys = 1
    plinder_decoy_df = get_decoy_df(
        con,
        uniprot_id,
        plinder_prop_df,
        target_decoys=n_decoys,
        cache_suffix=f"_plinder_{n_decoys}",
    )

    print("Adding to the database")
    engine = get_engine()

    # first drop the target and everything associated with it if it exists

    with Session(engine) as session:
        target = session.query(Target).filter(Target.name == target_name).first()
        if target is not None:
            protein = session.query(ProteinTarget).filter(ProteinTarget.id == target.id)
            activities = session.query(Activity).filter(Activity.target_id == target.id)
            decoys = session.query(Decoy).filter(Decoy.target_id == target.id)
            costructures = session.query(CoStructure).filter(
                CoStructure.target_id == target.id
            )
            mols = session.query(Molecule).filter(
                Molecule.id.in_(
                    [a.mol_id for a in activities]
                    + [d.mol_id for d in decoys]
                    + [c.mol_id for c in costructures]
                )
            )

            # drop everything
            activities.delete()
            decoys.delete()
            costructures.delete()
            mols.delete()
            protein.delete()

            session.delete(target)

        session.commit()

    target = Target(
        name=target_name,
        type=TargetType.PROTEIN,
        pdb=pdb,
        structure=rec_ref,
        binding_site=poc_residues,
    )
    protein_target = ProteinTarget(
        target=target,
        uniprot=uniprot_id,
        sequence=rec_seq,
    )

    items = [target, protein_target]

    struct_folder = os.path.join(get_plinder_benchmark_dir(uniprot_id), "structures")

    plinder_idx2mol = {}
    for i, row in tqdm(plinder_df.iterrows(), total=len(plinder_df)):
        if not isinstance(row.lig_smiles, str):
            continue

        mol = Molecule(mol=row.lig_smiles)
        plinder_idx2mol[i] = mol

        rec_fname = os.path.join(struct_folder, f"rec_{row.sys_id}.pdb")
        lig_fname = os.path.join(struct_folder, f"lig_{row.sys_id}_{row.lig_id}.sdf")

        rec = app.PDBFile(rec_fname)
        lig = Chem.SDMolSupplier(lig_fname, removeHs=False)[0]

        struct = CoStructure(
            target=target,
            mol=mol,
            pdb=row.sys_id.split("__")[0],
            rec_structure=rec,
            lig_structure=lig,
            pK=None if np.isnan(row.pK) else row.pK,
        )
        items += [mol, struct]

    for (i, decoy_row), (pi, row) in zip(
        tqdm(plinder_decoy_df.iterrows(), total=len(plinder_decoy_df)),
        plinder_active_df.iterrows(),
    ):
        if not isinstance(decoy_row.protonated_smiles, str):
            continue
        decoy_mol = Molecule(mol=decoy_row.protonated_smiles)
        parent_mol = plinder_idx2mol[pi]

        decoy = Decoy(
            target=target,
            mol=decoy_mol,
            parent=parent_mol,
        )
        items += [decoy_mol, decoy]

    for i, row in tqdm(chembl_df.iterrows(), total=len(chembl_df)):
        if not isinstance(row.protonated_smiles, str):
            continue

        mol = Molecule(mol=row.protonated_smiles)
        activity = Activity(
            type=ActivityType(row.standard_type),
            target=target,
            mol=mol,
            pK=row.pchembl_value,
        )
        items += [mol, activity]

    for i, row in tqdm(decoy_df.iterrows(), total=len(decoy_df)):
        if not isinstance(row.protonated_smiles, str):
            continue

        mol = Molecule(mol=row.protonated_smiles)
        decoy = Decoy(
            target=target,
            mol=mol,
        )
        items += [mol, decoy]

    with Session(engine) as session:
        session.add_all(items)
        session.commit()


# def fix_costructure_decoys(target_name):
#     """The original add_target code put costructures as their own decoys. This adds actual decoys"""

#     engine = get_engine()

#     with Session(engine) as ses:
#         target = ses.query(Target).filter(Target.name == target_name).one()
#         target_id = target.id
#         uniprot = (
#             ses.query(ProteinTarget).filter(ProteinTarget.id == target.id).one().uniprot
#         )
#         co_structures = (
#             ses.query(CoStructure)
#             .filter(CoStructure.target_id == target.id)
#             # .filter(CoStructure.pK != None)
#         ).all()

#         con = get_chembl_con()
#         plinder_active_df = pd.DataFrame(
#             [
#                 {"mol_id": co.mol_id, "smiles": Chem.MolToSmiles(co.lig_structure)}
#                 for co in co_structures
#             ]
#         )
#         plinder_prop_df = get_property_df(plinder_active_df.smiles)
#         plinder_prop_df = add_protonated_smiles(plinder_prop_df)
#         n_decoys = 1
#         plinder_decoy_df = get_decoy_df(
#             con,
#             uniprot,
#             plinder_prop_df,
#             target_decoys=n_decoys,
#             cache_suffix=f"_plinder_{n_decoys}",
#             force=True,
#         )

#         parent_ids = plinder_active_df.mol_id[plinder_decoy_df.parent_index]
#         decoy_smiles = plinder_decoy_df.protonated_smiles

#         with Session(engine) as ses:
#             # first delete all the old decoys
#             ses.query(Decoy).filter(Decoy.target_id == target_id).filter(
#                 Decoy.parent_id != None
#             ).delete()
#             to_add = []
#             for decoy_smi, parent_id in zip(decoy_smiles, parent_ids):
#                 decoy_mol = Molecule(mol=decoy_smi)
#                 decoy = Decoy(target_id=target_id, mol=decoy_mol, parent_id=parent_id)
#                 to_add.append(decoy)
#             ses.add_all(to_add)
#             ses.commit()


if __name__ == "__main__":
    print(
        "Warning: this will add target info to the database without checking if the data is reasonable. Make sure to debug with notebooks/add_uniprot first."
    )

    min_jaccard = 0.4
    overlap_cutoff = 2

    # add_target("P53", "P04637")
    # add_target("MAPK1", "P28482", min_jaccard=0.2, max_poc_dist=20 * unit.angstroms)
    # fix_costructure_decoys("P53")
    # fix_costructure_decoys("MAPK1")
    # fix_costructure_decoys("BRD4_V2")
    # fix_costructure_decoys("RORg")
    # add_target("HSP90", "P07900")
    # add_target("cMET", "P08581", min_jaccard=0.1)
    # add_target("TNKS2", "Q9H2K2", min_jaccard=0.1)
    # add_target("MCL1", "Q07820")

    # target_name = "CDK2"
    # uniprot_id = "P24941"
    # min_jaccard = 0.1

    # target_name = "ESR1"
    # uniprot_id = "P03372"
    # min_jaccard = None
    # overlap_cutoff = None

    target_name = "MDM2"
    uniprot_id = "Q00987"

    add_target(
        target_name,
        uniprot_id,
        min_jaccard=min_jaccard,
        overlap_cutoff=overlap_cutoff,
    )
