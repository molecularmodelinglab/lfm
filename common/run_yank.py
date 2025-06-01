import shutil
import subprocess
import numpy as np
import os
from common.alignment import find_rigid_alignment
from common.tanimoto import get_tanimoto_matrix, sample_diverse_set
from common.task_queue import task, task_loop
from rdkit import Chem
from common.utils import (
    get_CA_indices,
    get_fp,
    get_output_dir,
    get_residue_atom_indices,
    modeller_with_state,
    rdkit_to_modeller,
    save_modeller_pdb,
    save_system_xml,
)
from openmm import unit, app
import yaml
from sqlalchemy.orm import Session
from common.db import CoStructure, Decoy, Target, YankResult, YankInput, get_engine


def get_yank_config(lig_resname, time=4 * unit.nanoseconds):

    restraint_lambdas = np.linspace(0, 1, 3).tolist()
    coul_lambdas = np.linspace(1, 0, 8)[1:].tolist()
    vdw_lambdas = np.linspace(1, 0, 21)[1:].tolist()

    full_restraint_lambdas = restraint_lambdas + [1.0] * (
        len(coul_lambdas) + len(vdw_lambdas)
    )
    full_coul_lambdas = (
        [1.0] * len(restraint_lambdas) + coul_lambdas + [0.0] * len(vdw_lambdas)
    )
    full_vdw_lambdas = [1.0] * (
        len(restraint_lambdas) + len(coul_lambdas)
    ) + vdw_lambdas

    solv_coul_lambdas = coul_lambdas + [0.0] * len(vdw_lambdas)
    solv_vdw_lambdas = [1.0] * len(coul_lambdas) + vdw_lambdas

    return {
        "options": {
            "minimize": True,
            "verbose": True,
            "default_number_of_iterations": time.value_in_unit(unit.picoseconds),
            "temperature": "300*kelvin",
            "pressure": "1*atmosphere",
            "output_dir": "yank_output",
        },
        "systems": {
            "my-complex": {
                "phase1_path": ["complex.pdb", "complex.xml"],
                "phase2_path": ["ligand.pdb", "ligand.xml"],
                "ligand_dsl": f"resname '{lig_resname}'",  # https://github.com/mdtraj/mdtraj/issues/1126
            }
        },
        "protocols": {
            "absolute-binding": {
                "complex": {
                    "alchemical_path": {
                        "lambda_electrostatics": full_coul_lambdas,
                        "lambda_sterics": full_vdw_lambdas,
                        "lambda_restraints": full_restraint_lambdas,
                    }
                },
                "solvent": {
                    "alchemical_path": {
                        "lambda_electrostatics": solv_coul_lambdas,
                        "lambda_sterics": solv_vdw_lambdas,
                    }
                },
            }
        },
        "experiments": {
            "system": "my-complex",
            "protocol": "absolute-binding",
            "restraint": {"type": "Harmonic"},
        },
    }


def run_yank_cmd(yank_input):
    """Runs yank from the command line"""

    cur_dir = os.getcwd()
    out_dir = get_output_dir()
    os.chdir(out_dir)

    print(f"Running Yank in {out_dir}")

    try:
        shutil.rmtree("yank_output", ignore_errors=True)

        save_modeller_pdb(yank_input.complex_structure, "complex.pdb")
        save_system_xml(yank_input.complex_system, "complex.xml")
        save_modeller_pdb(yank_input.ligand_structure, "ligand.pdb")
        save_system_xml(yank_input.ligand_system, "ligand.xml")

        with open("yank_cfg.yaml", "w") as f:
            yaml.dump(yank_input.cfg, f)

        cmd = f"yank script -y yank_cfg.yaml"
        subprocess.run(cmd, shell=True, check=True)
    finally:
        os.chdir(cur_dir)


def get_yank_result(input_id, store_directory):
    """After we've run Yank, return the results to the DB"""
    from yank.analyze import get_analyzer, ExperimentAnalyzer

    report = ExperimentAnalyzer(store_directory)
    report.auto_analyze()
    fe_data = report.get_experiment_free_energy_data()

    delta_f = fe_data["free_energy_diff_unit"].value_in_unit(unit.kilocalories_per_mole)
    delta_f_err = fe_data["free_energy_diff_error_unit"].value_in_unit(
        unit.kilocalories_per_mole
    )

    diagnostics = {}

    for phase in ["complex", "solvent"]:
        storage_path = os.path.join(store_directory, phase + ".nc")
        analyzer = get_analyzer(storage_path)
        mixing_stats = analyzer.generate_mixing_statistics()

        diagnostics[phase] = {
            "transition_mat": mixing_stats.transition_matrix,
            "max_transition_mat": mixing_stats.transition_matrix.max(),
            "perron_eigenvalue": mixing_stats.eigenvalues[1],
        }

    ret = YankResult(
        input_id=input_id,
        dG=delta_f,
        dG_uncertainty=delta_f_err,
        complex_transition_mat=diagnostics["complex"]["transition_mat"],
        complex_max_transition_mat=diagnostics["complex"]["max_transition_mat"],
        complex_perron_eigenvalue=diagnostics["complex"]["perron_eigenvalue"],
        solvent_transition_mat=diagnostics["solvent"]["transition_mat"],
        solvent_max_transition_mat=diagnostics["solvent"]["max_transition_mat"],
        solvent_perron_eigenvalue=diagnostics["solvent"]["perron_eigenvalue"],
    )

    return ret


@task(prefetch=1)
def yank_task(yank_input_id):
    engine = get_engine()
    with Session(engine) as ses:
        yank_input = ses.query(YankInput).get(yank_input_id)
        assert yank_input is not None

    store_dir = os.path.join(get_output_dir(), "yank_output", "experiments")

    # does this help?
    engine = get_engine()
    with Session(engine) as ses:
        run_yank_cmd(yank_input)
        ret = get_yank_result(yank_input_id, store_dir)
        ses.add(ret)
        ses.commit()


def get_yank_inputs(
    target_id,
    mol_id,
    costructure_id,
    rec_ref,
    reporter,
    db_traj,
    poc_indices,
    poc_alpha_indices,
    lig_rd,
    sim_time,
    dock=False,
):
    """Runs a quick alchemical sim to create input structures and systems
    for use with yank"""

    from common.openmm_utils import make_system
    from datagen.pmf_sim import PMFSim
    from common.unidock import run_unidock

    rand_idx = np.random.randint(20, len(reporter))
    state = reporter[rand_idx]

    rec = modeller_with_state(db_traj.initial_structure, state)
    #  needed to ensure we don't include cofactors in the yanking
    for residue in rec.topology.residues():
        if residue.name == "UNK":
            residue.name = "COF"
    for residue in rec_ref.topology.residues():
        if residue.name == "UNK":
            residue.name = "COF"
    

    if dock:
        # use unidock to dock the ligand
        rec_nowat = app.Modeller(rec_ref.topology, rec.positions[:len(rec_ref.positions)])
        lig_rd = run_unidock(rec_nowat, [lig_rd], poc_indices)[0]
        lig = rdkit_to_modeller(lig_rd)

    else:
        # we already know the structure. align ligand to be in reference frame of rec
        rec_poc = rec.positions[poc_alpha_indices].value_in_unit(unit.nanometers)
        ref_rec_poc = rec_ref.positions[poc_alpha_indices].value_in_unit(unit.nanometers)

        rec_origin = rec_poc.mean(axis=0)
        ref_rec_origin = ref_rec_poc.mean(axis=0)

        R, t = find_rigid_alignment(ref_rec_poc - ref_rec_origin, rec_poc - rec_origin)

        lig = rdkit_to_modeller(lig_rd)
        lig_pos = lig.positions.value_in_unit(unit.nanometers)
        lig.positions = (
            R.dot((lig_pos - ref_rec_origin).T).T + rec_origin
        ) * unit.nanometers

    # alchemically add ligand to rec to create input systems
    pmf_sim = PMFSim(rec, lig, lig_rd, rec_ref, poc_indices)
    state = pmf_sim.run_alchemical_lr_sim()

    # assert False, "AAAAA we're currently freezing the ligand!"
    lr = modeller_with_state(pmf_sim.mols["lr"], state)
    lr_sys = pmf_sim.systems["lr"]

    cache_dir = f"{get_output_dir()}/cache"

    lig_sys, lig = make_system(
        pmf_sim.mols["lig_ion"],
        mols=[lig_rd],
        box_padding=1.2 * unit.nanometers,  # extra in case box size goes down
        cache_dir=cache_dir,
    )

    lig_resname = list(lig.topology.residues())[0].name
    cfg = get_yank_config(lig_resname, sim_time)

    return YankInput(
        target_id=target_id,
        mol_id=mol_id,
        costructure_id=costructure_id,
        complex_structure=lr,
        ligand_structure=lig,
        complex_system=lr_sys,
        ligand_system=lig_sys,
        cfg=cfg,
    )

def get_yank_inputs_screen(
    target_id,
    screen_mol_id,
    rec_ref,
    reporter,
    db_traj,
    poc_indices,
    poc_alpha_indices,
    lig_rd,
    sim_time,
    cofactors=None
):
    """Assumes that the ligand is already docked to the (reference) receptor.
    Creates an input for postprocessing the screen results"""

    from common.openmm_utils import make_system
    from datagen.pmf_sim import PMFSim

    rand_idx = np.random.randint(20, len(reporter))
    state = reporter[rand_idx]

    rec = modeller_with_state(db_traj.initial_structure, state)

    # remove any costructure ligands from the rec
    ref_residues = list(rec_ref.topology.residues())
    initial_residues = list(rec.topology.residues())

    to_remove = []
    for residue in initial_residues[len(ref_residues) :]:
        if residue.name in ["HOH", "Na+", "Cl-", "NA", "CL"]:
            continue
        to_remove.append(residue)
    rec.delete(to_remove)

    rec.positions = (
        np.array([v.value_in_unit(unit.nanometer) for v in rec.positions])
        * unit.nanometer
    )

    # we already know the structure. align ligand to be in reference frame of rec
    rec_poc = rec.positions[poc_alpha_indices].value_in_unit(unit.nanometers)
    ref_rec_poc = rec_ref.positions[poc_alpha_indices].value_in_unit(unit.nanometers)

    rec_origin = rec_poc.mean(axis=0)
    ref_rec_origin = ref_rec_poc.mean(axis=0)

    R, t = find_rigid_alignment(ref_rec_poc - ref_rec_origin, rec_poc - rec_origin)

    lig = rdkit_to_modeller(lig_rd)
    lig_pos = lig.positions.value_in_unit(unit.nanometers)
    lig.positions = (
        R.dot((lig_pos - ref_rec_origin).T).T + rec_origin
    ) * unit.nanometers

    for residue in rec.topology.residues():
        if residue.name == "UNK":
            residue.name = "COF"
    for residue in rec_ref.topology.residues():
        if residue.name == "UNK":
            residue.name = "COF"

    # alchemically add ligand to rec to create input systems
    pmf_sim = PMFSim(rec, lig, lig_rd, rec_ref, poc_indices, extra_mols=cofactors)
    state = pmf_sim.run_alchemical_lr_sim()

    # assert False, "AAAAA we're currently freezing the ligand!"
    lr = modeller_with_state(pmf_sim.mols["lr"], state)
    lr_sys = pmf_sim.systems["lr"]

    cache_dir = f"{get_output_dir()}/cache"

    lig_sys, lig = make_system(
        pmf_sim.mols["lig_ion"],
        mols=[lig_rd],
        box_padding=1.2 * unit.nanometers,  # extra in case box size goes down
        cache_dir=cache_dir,
    )

    lig_resname = list(lig.topology.residues())[0].name
    cfg = get_yank_config(lig_resname, sim_time)

    return YankInput(
        target_id=target_id,
        screen_mol_id=screen_mol_id,
        complex_structure=lr,
        ligand_structure=lig,
        complex_system=lr_sys,
        ligand_system=lig_sys,
        cfg=cfg,
    )


def queue_yank_costructure_tasks(
    target_names, max_tasks_per_target=8, time_per_task=4.0 * unit.nanoseconds
):
    """Test run on known binders for known targets"""
    from common.md_sim import GSReporter, get_target_md_traj, make_system
    from datagen.pmf_sim import PMFSim

    engine = get_engine()
    for target_name in target_names:

        # first find all the costructures for the target
        with Session(engine) as ses:
            target = ses.query(Target).filter(Target.name == target_name).one()
            target_id = target.id
            poc_res_indices = target.binding_site
            rec_ref = target.structure

            # now delete all the yank inputs and results for this target
            # todo: this is bad! We need a better way to manage different
            # yank output versions
            cur_inputs = ses.query(YankInput).filter(
                YankInput.target_id == target.id
            )
            cur_results = ses.query(YankResult).filter(
                YankResult.input_id.in_([i.id for i in cur_inputs])
            )
            cur_results.delete()
            cur_inputs.delete()
            ses.commit()

            # get all costructures with pK values
            co_structures = (
                ses.query(CoStructure)
                .filter(CoStructure.target_id == target.id)
                .filter(CoStructure.pK != None)
            ).all()

        poc_indices = get_residue_atom_indices(rec_ref.topology, poc_res_indices)
        poc_alpha_indices = [
            i for i in poc_indices if i in get_CA_indices(rec_ref.topology)
        ]

        db_traj = get_target_md_traj(engine, target_id)
        reporter = GSReporter(db_traj.path, db_traj.initial_structure, "r")

        # filter out costructures with formal charge != 0
        co_structures = [
            cs for cs in co_structures if Chem.GetFormalCharge(cs.lig_structure) == 0
        ]
        fps = np.array([get_fp(cs.lig_structure) for cs in co_structures])
        tan_mat = get_tanimoto_matrix(fps)
        diverse_indices = sample_diverse_set(tan_mat, 0.3)

        print("Queueing up to", len(diverse_indices), "yank tasks for target", target_name)

        for idx in diverse_indices[:max_tasks_per_target]:
            co_structure = co_structures[idx]
            lig_rd = co_structure.lig_structure

            yank_input = get_yank_inputs(
                co_structure.target_id,
                co_structure.mol_id,
                co_structure.id,
                rec_ref,
                reporter,
                db_traj,
                poc_indices,
                poc_alpha_indices,
                lig_rd,
                time_per_task,
            )

            # now get the yank input for the decoy
            with Session(engine) as ses:
                decoy = (
                    ses.query(Decoy)
                    .filter(Decoy.parent_id == co_structure.mol_id)
                ).one()
                decoy_mol_id = decoy.mol_id
                decoy_mol = decoy.mol.mol

            yank_decoy_input = get_yank_inputs(
                co_structure.target_id,
                decoy_mol_id,
                None,
                rec_ref,
                reporter,
                db_traj,
                poc_indices,
                poc_alpha_indices,
                decoy_mol,
                time_per_task,
                dock=True,
            )          

            # save the yank input to the DB
            with Session(engine) as ses:
                ses.add(yank_input)
                ses.add(yank_decoy_input)
                ses.commit()
                yank_task.delay(yank_input.id)
                yank_task.delay(yank_decoy_input.id)


if __name__ == "__main__":
    import sys

    action = sys.argv[1]

    if action == "queue":
        target_names = sys.argv[2:]
        queue_yank_costructure_tasks(target_names)
    elif action == "run":
        engine = get_engine()
        task_loop(engine, "yank_task")

    # store_dir = sys.argv[1]
    # ret = get_yank_result(0, store_dir)
    # for key, val in ret.__dict__.items():
    #     print(key, val)
