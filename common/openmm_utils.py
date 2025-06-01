from openmm import app, unit
import numpy as np
import openmm as mm
import os

try:
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import EspalomaTemplateGenerator
except ModuleNotFoundError:
    print("Espaloma and/or openff.toolkit not installed")

# todo: move a bunch of stuff from utils to here


def make_system(
    structure,
    solvent="tip3p",  # either "tip[n]p", "gbn[1-2]", "obc[1-2]", or "none". If "none", then we are in a vacuum. If "tip3p", then we are in explicit solvent and this will create the solvent box.
    mols=[],  # list of rdkit molecules
    constraints=app.HBonds,
    nonbonded_method=None,  # PME if solvent is explicit, NoCutoff if vacuum
    nonbonded_cutoff=0.8 * unit.nanometer,
    include_barostat=None,  # if True, then we include a MonteCarloBarostat, by default we only include if using explicit solvent
    box_vectors=None,  # if not None, then this is the box vectors for the system
    box_padding=0.8 * unit.nanometer,
    P=1.0 * unit.atmosphere,  # only used if include_barostat is True
    T=300 * unit.kelvin,
    ion_conc=0.0 * unit.molar,  # salt concentration
    cache_dir=None,  # where to store espaloma cache files
    box_shape="dodecahedron",  # shape of the box if we're adding solvent
    add_waters=True,  # if True, then we add solvent to the system (if using tip3p)
    H_mass=1.5 * unit.amu,  # mass of hydrogen
):
    """Returns both mm System and the modeller object with the system added to it."""

    explicit_solvent = "tip" in solvent

    if include_barostat is None:
        include_barostat = explicit_solvent

    if nonbonded_method is None:
        nonbonded_method = app.PME if explicit_solvent else app.NoCutoff

    modeller = app.Modeller(structure.topology, structure.positions)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    forcefield_kwargs = {
        "nonbondedMethod": nonbonded_method,
        "nonbondedCutoff": nonbonded_cutoff,
        "constraints": constraints,
        "hydrogenMass": H_mass,
    }

    ffs = [
        "amber/ff14SB.xml",
        "amber/tip3p_HFE_multivalent.xml",
    ]
    if solvent != "none":
        prefix = "amber14" if explicit_solvent else "implicit"
        ffs.append(f"{prefix}/{solvent}.xml")

    if mols is None:
        mols = []
    elif not isinstance(mols, list):
        mols = [mols]

    off_mols = [Molecule.from_rdkit(mol, allow_undefined_stereo=True) for mol in mols]

    forcefield = app.ForceField(*ffs)
    generator = EspalomaTemplateGenerator(
        off_mols,
        cache=None if cache_dir is None else f"{cache_dir}/espaloma.json",
    )
    forcefield.registerTemplateGenerator(generator.generator)

    if explicit_solvent and add_waters:
        modeller.addSolvent(
            forcefield,
            model=solvent,
            padding=box_padding,
            positiveIon="Na+",
            negativeIon="Cl-",
            ionicStrength=ion_conc,
            neutralize=True,
            boxShape=box_shape,
        )

    if box_vectors is not None:
        modeller.topology.setPeriodicBoxVectors(box_vectors)

    system = forcefield.createSystem(modeller.topology, **forcefield_kwargs)

    if include_barostat:
        system.addForce(mm.MonteCarloBarostat(P, T))

    modeller.positions = (
        np.array([v.value_in_unit(unit.nanometer) for v in modeller.positions])
        * unit.nanometer
    )

    return system, modeller


def add_solvent_to_modeller(
    modeller,
    box_padding=0.8 * unit.nanometer,
    box_shape="dodecahedron",
):
    forcefield = app.ForceField("amber/ff14SB.xml", "amber14/tip3p.xml")
    ret = app.Modeller(modeller.topology, modeller.positions)
    ret.addSolvent(forcefield, model="tip3p", padding=box_padding, boxShape=box_shape)

    ret.positions = (
        np.array([v.value_in_unit(unit.nanometer) for v in ret.positions])
        * unit.nanometer
    )

    return ret
