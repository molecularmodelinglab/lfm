import os
import sys

from tqdm import tqdm, trange
from common.alignment import find_rigid_alignment
from common.db import CoStructure, Target, TargetMDTrajectory, db_gs_dir
from common.gs import GS_FS, download_gs_file
from datagen.pmf_sim import PMFSim
import zarr
from common.utils import (
    CONFIG,
    get_CA_indices,
    get_residue_atom_indices,
    modeller_with_state,
    rdkit_to_modeller,
)
from openmmtools import states
from openmm import unit, app
import numpy as np
import openmm as mm
from sqlalchemy.orm import Session
from common.celery_app import celery_app
from common.db import get_engine
from common.openmm_utils import add_solvent_to_modeller, make_system

try:
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import EspalomaTemplateGenerator
except ModuleNotFoundError:
    print("Espaloma and/or openff.toolkit not installed")

GCLOUD_TOKEN = os.path.abspath(CONFIG.storage.token_dir)

# this object will be useful for google cloud storage IO
BUCKET_FS = None


def reload_bucket_fs():
    global BUCKET_FS
    BUCKET_FS = zarr.storage.normalize_store_arg(
        f"gs://{CONFIG.storage.bucket}",
        storage_options={"token": GCLOUD_TOKEN},
        mode="w",
    )


# reload_bucket_fs()


def global_gs_path(path):
    """Assumes path is relative to the bucket root; returns the full path"""
    return f"gs://{CONFIG.storage.bucket}/{path}"


def local_gs_path(path):
    """Removes the gs://bucket/ prefix"""
    return path.replace(f"gs://{CONFIG.storage.bucket}/", "")


def delete_global_gs_path(path):
    """Deletes path (even if directory!) from gs"""
    reload_bucket_fs()
    local_path = local_gs_path(path)
    try:
        del BUCKET_FS[local_path]
    except KeyError:
        print(f"Looks like the path {path} was already deleted")


def zarr_open_gs(path, *args, use_gs=True, **kwargs):
    """open a zarr file from google cloud storage"""
    return zarr.open(
        path,
        *args,
        **kwargs,
        storage_options={"token": GCLOUD_TOKEN} if use_gs else None,
    )


def zarr_save_gs(path, arr):
    """save a zarr object to google cloud storage"""
    store = zarr.storage.normalize_store_arg(
        path, storage_options={"token": GCLOUD_TOKEN}, mode="w"
    )
    zarr.save(store, arr)


class GSReporter:
    """A class to store zarr archives of the frames (and box vectors)
    for a trajectory."""

    def __init__(
        self, path: str, modeller: app.Modeller, mode: str = "a", download=False
    ):

        # this messes up gs smh
        if path[-1] == "/":
            path = path[:-1]

        if download:
            downloaded_path = download_gs_file(path)
            path = downloaded_path

        self.topology = modeller.topology
        n_atoms = modeller.positions.shape[0]
        self.frames = zarr_open_gs(
            f"{path}/frames.zarr",
            mode=mode,
            shape=(0, n_atoms, 3),
            chunks=(10, n_atoms, 3),
            dtype="f4",
            use_gs=not download,
        )
        if modeller.topology.getPeriodicBoxVectors() is not None:
            self.box_vectors = zarr_open_gs(
                f"{path}/box_vectors.zarr",
                mode=mode,
                shape=(0, 3, 3),
                chunks=(1000, 3, 3),
                dtype="f4",
                use_gs=not download,
            )
        else:
            self.box_vectors = None

    def __len__(self):
        return len(self.frames)

    def append(self, state: states.SamplerState):
        """Append a state to the trajectory"""
        self.frames.append(state.positions.value_in_unit(unit.nanometers)[None])
        if self.box_vectors is not None:
            self.box_vectors.append(
                state.box_vectors.value_in_unit(unit.nanometers)[None]
            )

    def __getitem__(self, idx: int) -> states.SamplerState:
        """Get a state from the trajectory"""
        if idx >= len(self):
            raise IndexError

        pos = self.frames[idx]
        box = self.box_vectors[idx] if self.box_vectors is not None else None
        return states.SamplerState(
            positions=pos * unit.nanometers,
            box_vectors=box * unit.nanometers if box is not None else None,
        )

    def save_dcd(self, fname: str):
        """Write the trajectory to a DCD file"""
        with open(fname, "wb") as f:
            dcd = app.DCDFile(f, self.topology, 1.0 * unit.picoseconds)
            for state in tqdm(self):
                dcd.writeModel(state.positions, periodicBoxVectors=state.box_vectors)


def get_target_md_traj_by_name(engine, target_id, name):
    """Returns the TargetMDTrajectory for the target by name"""

    with Session(engine) as ses:
        db_traj = (
            ses.query(TargetMDTrajectory)
            .filter(TargetMDTrajectory.target_id == target_id)
            .filter(TargetMDTrajectory.name == name)
            .one_or_none()
        )
        # access these attributes to ensure that they are loaded
        system = db_traj.system
        modeller = db_traj.initial_structure

    return db_traj

def get_target_md_traj(engine, target_id, co_structure_id=None, name=None):
    """Returns the TargetMDTrajectory for the target. Creates one if it doesn't exist."""

    with Session(engine) as ses:
        db_traj = (
            ses.query(TargetMDTrajectory)
            .filter(TargetMDTrajectory.target_id == target_id)
            .filter(TargetMDTrajectory.co_structure_id == co_structure_id)
            .one_or_none()
        )
        if db_traj is None:

            # get the target
            target = ses.query(Target).get(target_id)
            rec_ref = target.structure

            # get the co_structure ligand (if applicable)
            # and alchemically add it to the simulation

            if target.cofactors is not None:
                if isinstance(target.cofactors, list):
                    mols = target.cofactors
                else:
                    mols = [target.cofactors]
            else:
                mols = []

            if co_structure_id is not None:

                # get a frame from the apo trajectory
                db_traj = get_target_md_traj(engine, target_id)
                reporter = GSReporter(db_traj.get_gs_path(), db_traj.initial_structure, "r")
                state = reporter[5]
                rec = modeller_with_state(db_traj.initial_structure, state)

                co_structure = (
                    ses.query(CoStructure.lig_structure)
                    .filter(CoStructure.id == co_structure_id)
                    .one()
                )
                lig_rd = co_structure.lig_structure

                poc_res_indices = target.binding_site
                poc_indices = get_residue_atom_indices(
                    rec_ref.topology, poc_res_indices
                )
                poc_alpha_indices = [
                    i for i in poc_indices if i in get_CA_indices(rec_ref.topology)
                ]

                rec_poc = rec.positions[poc_alpha_indices].value_in_unit(
                    unit.nanometers
                )
                ref_rec_poc = rec_ref.positions[poc_alpha_indices].value_in_unit(
                    unit.nanometers
                )

                rec_origin = rec_poc.mean(axis=0)
                ref_rec_origin = ref_rec_poc.mean(axis=0)

                R, t = find_rigid_alignment(
                    ref_rec_poc - ref_rec_origin, rec_poc - rec_origin
                )

                lig = rdkit_to_modeller(lig_rd)
                lig_pos = lig.positions.value_in_unit(unit.nanometers)
                lig.positions = (
                    R.dot((lig_pos - ref_rec_origin).T).T + rec_origin
                ) * unit.nanometers

                pmf_sim = PMFSim(
                    rec, lig, lig_rd, rec_ref, poc_indices, extra_mols=target.cofactors
                )
                state = pmf_sim.run_alchemical_lr_sim(dt=4.0 * unit.femtoseconds)

                mols.append(lig)

                modeller = modeller_with_state(pmf_sim.mols["lr"], state)
                system = pmf_sim.systems["lr"]
                # system, modeller = make_system(
                #     modeller,
                #     mols=mols,
                #     add_waters=False,
                # )

            else:
                system, modeller = make_system(target.structure, mols=mols)

            db_traj = TargetMDTrajectory(
                target_id=target_id,
                system=system,
                initial_structure=modeller,
                co_structure_id=co_structure_id,
                name=name,
            )
            ses.add(db_traj)
            ses.commit()

            gs_path = f"{db_gs_dir()}/{TargetMDTrajectory.__tablename__}/{db_traj.id}"
            db_traj.path = gs_path
            ses.commit()

        # access these attributes to ensure that they are loaded
        system = db_traj.system
        modeller = db_traj.initial_structure
        gs_path = db_traj.path

    return db_traj


def delete_md_traj_for_target(engine, target_id, co_structure_id=None):
    """Deletes the trajectory for the target if it exists."""

    with Session(engine) as ses:
        db_traj = (
            ses.query(TargetMDTrajectory)
            .filter(TargetMDTrajectory.target_id == target_id)
            .filter(TargetMDTrajectory.co_structure_id == co_structure_id)
            .one_or_none()
        )

        if db_traj is not None:
            print(
                f"Deleting trajectory for target {target_id} and co_structure {co_structure_id}"
            )
            delete_global_gs_path(db_traj.path)
            ses.delete(db_traj)
            ses.commit()


def run_md_sim(
    db_traj,
    tot_time=100.0 * unit.nanoseconds,
    dt=4.0 * unit.femtoseconds,
    report_interval=1.0 * unit.nanoseconds,
):
    """Continues already existing MD trajectory if possible"""

    reporter = GSReporter(db_traj.path, db_traj.initial_structure)

    platform = mm.Platform.getPlatformByName("CUDA")

    N_reports = int(tot_time / report_interval) - len(reporter)
    steps_per_report = int(report_interval / dt)

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, dt
    )
    sim = app.Simulation(
        db_traj.initial_structure.topology, db_traj.system, integrator, platform
    )

    print("Running MD simulation for", N_reports, "reports")
    if len(reporter) > 0:
        print("Starting from existing trajectory with", len(reporter), "frames")
        state = reporter[-1]
        state.apply_to_context(sim.context)
    else:
        print("Starting from scratch")
        sim.context.setPositions(db_traj.initial_structure.positions)
        sim.minimizeEnergy()

    for i in trange(N_reports):
        sim.step(steps_per_report)
        state = states.SamplerState.from_context(sim.context)
        reporter.append(state)


@celery_app.task(
    worker_prefetch_multiplier=1,
    name="target_md_task",
)
def target_md_task(target_id, co_structure_id=None, name=None):
    engine = get_engine()
    db_traj = get_target_md_traj(engine, target_id, co_structure_id, name)
    run_md_sim(db_traj)


def queue_target_md_task(target_name, use_largest_lig, name, force, noqueue):
    engine = get_engine()

    with Session(engine) as ses:
        target = ses.query(Target).filter(Target.name == target_name).one()
        if use_largest_lig:
            co_structures = (
                ses.query(CoStructure.lig_structure, CoStructure.id)
                .filter(CoStructure.target_id == target.id)
                .all()
            )
            assert len(co_structures) > 0, "No co_structures found for target"
            co_structure = max(
                co_structures, key=lambda x: x.lig_structure.GetNumAtoms()
            )
            co_structure_id = co_structure.id
        else:
            co_structure_id = None

    if force:
        delete_md_traj_for_target(engine, target.id, co_structure_id)

    # hacky! This creates the system here b/c the worker doesn't have espaloma installed
    # get_target_md_traj(engine, target.id, co_structure_id)
    if noqueue:
        target_md_task(target.id, co_structure_id, name)
    else:
        target_md_task.delay(target.id, co_structure_id, name)


if __name__ == "__main__":
    target_names = [name for name in sys.argv[1:] if not name.startswith("--")]
    # if this is set, use the largest ligand in the binding site as the co_structure
    # otherwise, use apo conformation
    use_largest_lig = "--use-largest-lig" in sys.argv
    force = "--force" in sys.argv
    noqueue = "--noqueue" in sys.argv
    name = "holo_largest" if use_largest_lig else "apo"

    for target_name in target_names:
        print(f"Queuing MD task for {target_name}")
        queue_target_md_task(target_name, use_largest_lig, name, force, noqueue)
