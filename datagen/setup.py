# Initializes everything need for a new pmf simulation run, as well
# as creating the validation sets

import requests
import pandas as pd
from pdbfixer import PDBFixer
from openmm import app, unit
import numpy as np


def get_best_pdb_structure(uniprot_id):
    """Returns the PDB ID and chain of the highest-res structure in the PDB"""

    response = requests.get(
        f"https://rest.uniprot.org/uniprotkb/search?query={uniprot_id}&fields=structure_3d"
    )
    response.raise_for_status()
    data = response.json()
    rows = []
    for item in data["results"][0]["uniProtKBCrossReferences"]:
        row = {
            "database": item["database"],
            "id": item["id"],
        }
        for prop in item["properties"]:
            row[prop["key"]] = prop["value"]

        try:
            row["Resolution"] = float(row["Resolution"].split()[0])
        except ValueError:
            row["Resolution"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Resolution")

    pdb_id = df.iloc[0]["id"]
    chain_id = df.iloc[0].Chains.split("=")[0].split("/")[0]

    return pdb_id, chain_id


def fix_structure(fixer, max_loop_len=8):
    last_idx = len(list(fixer.topology.residues()))
    fixer.findMissingResidues()

    # first remove missing loops that are longer than max_loop_len
    to_remove = []
    for key, value in fixer.missingResidues.items():
        if len(value) > max_loop_len:
            to_remove.append(key)

    for key in to_remove:
        del fixer.missingResidues[key]

    # we don't want long dangling ends
    to_remove = []
    for chain in fixer.topology.chains():

        last_idx = len(list(chain.residues()))

        # we don't want long dangling ends
        if (chain.index, 0) in fixer.missingResidues:
            del fixer.missingResidues[(chain.index, 0)]
        if (chain.index, last_idx) in fixer.missingResidues:
            del fixer.missingResidues[(chain.index, last_idx)]

        # gdi sometimes we have some dangling residues on
        # their own chain... remove!
        if len(list(chain.residues())) < 2:
            to_remove.append(chain.index)

    fixer.removeChains(to_remove)

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    positions = np.array([ v.value_in_unit(unit.nanometers) for v in fixer.positions ]) * unit.nanometers

    return app.Modeller(fixer.topology, positions)

def get_fixed_structure(pdb_id, chain_id, max_loop_len=8, membrane=False):
    """Returns a Modeller with the fixed PDB structure"""

    if membrane:
        # download from OPM instead
        url = f"https://opm-assets.storage.googleapis.com/pdb/{pdb_id.lower()}.pdb"
        fixer = PDBFixer(url=url)
    else:
        fixer = PDBFixer(pdbid=pdb_id)
        
    to_remove = {
        i for i, chain in enumerate(fixer.topology.chains()) if chain.id != chain_id
    }

    fixer.removeChains(to_remove)

    return fix_structure(fixer, max_loop_len=max_loop_len)
