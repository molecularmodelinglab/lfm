import argparse
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO, TextIOWrapper
import os
import sys
import time
import traceback
from typing import Optional
from uuid import uuid4

from tqdm import tqdm
from common.tanimoto import batch_tanimoto_faster
import gcsfs
import GPUtil
import fsspec
from common.gs import GCLOUD_TOKEN, GS_FS
import torch
import numpy as np
from common.alignment import find_rigid_alignment
from common.md_sim import GSReporter, get_target_md_traj, local_gs_path
from common.unidock import run_unidock
from common.utils import (
    CONFIG,
    FatalError,
    get_CA_indices,
    get_fp,
    get_nonbonded_params,
    get_residue_atom_indices,
    modeller_with_state,
    protonate_smiles,
    rdkit_to_modeller,
    remove_salt_from_smiles,
    save_modeller_pdb,
    save_pointcloud_pdb,
)
from datagen.pmf_sim import PMFSim
from datagen.zinc import get_random_zinc_smiles
from sqlalchemy.orm import Session
from rdkit import Chem
from rdkit.Chem import AllChem
from openmm import unit, app
import zarr
from scipy.spatial.transform import Rotation, Slerp
from common.db import (
    AlternativeStructure,
    PMFDatagen,
    PMFDatagenResult,
    Target,
    TargetMDTrajectory,
    exec_sql,
    get_alt_struct_id,
    get_engine,
    get_target_id_by_name,
    get_target_struct_and_pocket
)

# thread safety
fsspec.asyn.iothread[0] = None
fsspec.asyn.loop[0] = None

# _cur_gsfs = None
# def get_gs_fs():
#     """ Returns the gcsfs file system object. Necessary to be thread-safe """
#     global _cur_gsfs
#     if _cur_gsfs is None:
#         _cur_gsfs = gcsfs.GCSFileSystem(token=GCLOUD_TOKEN)
#     return _cur_gsfs

buc_fs = None

# def get_bucket_fs():
#     global buc_fs
#     if buc_fs is None:
#         buc_fs = zarr.storage.normalize_store_arg(
#             f"gs://{CONFIG.storage.bucket}",
#             storage_options={"token": GCLOUD_TOKEN},
#             mode="w",
#         )
#     return buc_fs

gs_fs_obj = None


def get_gs_fs():
    global gs_fs_obj
    gs_fs_obj = gcsfs.GCSFileSystem(token=GCLOUD_TOKEN)
    return gs_fs_obj


@dataclass
class PMFSimResult:

    forces: np.ndarray
    energies: np.ndarray
    hessians: Optional[np.ndarray]

    lig_positions: np.ndarray
    lig_nbparams: np.ndarray

    # if ligand is unfrozen, this is the trajectory
    lig_md_positions: Optional[np.ndarray]

    # for new COM restraints
    restraint_com: Optional[np.ndarray]

    # lig_insertion_work: float

    # lig2_positions: np.ndarray
    # lig2_insertion_work: float

    lig_rd: Chem.Mol

    def save_gs(self, gs_path, nolig=False):
        """Save all the numpy arrays to a npz file within the path and the ligand to an sdf file"""

        # gs_local = local_gs_path(gs_path)

        np_path = f"{gs_path}/data.npz"
        # stream = BytesIO()
        stream = get_gs_fs().open(np_path, "wb")

        kwargs = {
            "forces": self.forces,
            "energies": self.energies,
            "lig_positions": self.lig_positions,
            "lig_nbparams": self.lig_nbparams,
        }
        if self.hessians is not None:
            kwargs["hessians"] = self.hessians
        if self.lig_md_positions is not None:
            kwargs["lig_md_positions"] = self.lig_md_positions
        if self.restraint_com is not None:
            kwargs["restraint_com_v2"] = self.restraint_com

        np.savez(
            stream,
            **kwargs,
        )
        stream.close()
        if nolig:
            return

        # BUCKET_FS[np_path] = stream.getvalue()

        sdf_path = f"{gs_path}/ligand.sdf"
        # stream = TextIOWrapper(BytesIO(), encoding="utf-8")
        stream = get_gs_fs().open(sdf_path, "w")
        w = Chem.SDWriter(stream)
        w.write(self.lig_rd)
        w.close()
        stream.close()
        # BUCKET_FS[sdf_path] = stream.buffer.getvalue()

    def from_gs(gs_path):
        """Load a result from google cloud"""

        # gs_local = local_gs_path(gs_path)

        np_path = f"{gs_path}/data.npz"
        # stream = BytesIO(get_bucket_fs()[np_path])
        with get_gs_fs().open(np_path, "rb") as f:
            stream = BytesIO(f.read())

        npz = np.load(stream)

        forces = npz["forces"]
        energies = npz["energies"]
        hessians = npz["hessians"] if "hessians" in npz else None
        lig_md_positions = (
            npz["lig_md_positions"] if "lig_md_positions" in npz else None
        )
        restraint_com = npz["restraint_com_v2"] if "restraint_com_v2" in npz else None
        lig_positions = npz["lig_positions"]
        lig_nbparams = npz["lig_nbparams"]
        stream.close()

        sdf_path = f"{gs_path}/ligand.sdf"
        # stream = BytesIO(get_bucket_fs()[sdf_path])
        # stream = get_gs_fs().open(sdf_path, "rb")
        with get_gs_fs().open(sdf_path, "rb") as f:
            stream = BytesIO(f.read())

        lig_rd = next(iter(Chem.ForwardSDMolSupplier(stream, removeHs=False)))
        stream.close()

        return PMFSimResult(
            forces=forces,
            energies=energies,
            hessians=hessians,
            lig_positions=lig_positions,
            lig_nbparams=lig_nbparams,
            lig_md_positions=lig_md_positions,
            restraint_com=restraint_com,
            lig_rd=lig_rd,
        )


def randomize_ligand_pose(
    lig,
    max_trans_mean,
    trans_path=None,
    max_trans=25 * unit.angstrom,
    trans_std=6.0 * unit.angstrom,
    interpolate=False,
):
    """Uniformly choose a mean translation between max_trans_mean and (0,0,0).
    Unless max_trans_path is not None, in which case we randomly sample
    the mean translation from this set of points.
    Then sample a translation from a normal distribution, ensuring its length is less than max_trans.
    Now sample random rotation.
    If interpolate is true, sample a random alpha between 0 and 1. Create interpolated pose.
    Also returns the index of the path point divided by the path length (path_alpha)
    """

    max_trans = max_trans.value_in_unit(unit.nanometers)
    max_trans_mean = max_trans_mean.value_in_unit(unit.nanometers)
    trans_std = trans_std.value_in_unit(unit.nanometers)

    if trans_path is not None:
        trans_path = trans_path.value_in_unit(unit.nanometers)
        idx = np.random.randint(len(trans_path))
        path_alpha = idx / len(trans_path)
        t_mean = trans_path[idx]
    else:
        t_mean = np.random.rand() * max_trans_mean
        path_alpha = None

    while True:
        t_final = np.random.randn(3) * trans_std + t_mean
        if np.linalg.norm(t_final) < max_trans:
            break

    R_final = Rotation.random()

    if interpolate:
        R_initial = Rotation.identity()
        slerp = Slerp([0, 1], Rotation.concatenate([R_initial, R_final]))

        alpha = np.random.rand()

        t = alpha * t_final
        R = slerp(alpha)
    else:
        t = t_final
        R = R_final

    lig_pos = lig.positions.value_in_unit(unit.nanometers)
    lig_center = lig_pos.mean(axis=0)

    lig_pos = R.apply(lig_pos - lig_center) + t + lig_center

    return app.Modeller(lig.topology, lig_pos * unit.nanometers), path_alpha


def compute_exit_point(
    rec_ref,
    poc_alpha_indices,
    min_rec_dist=0.9,
    max_poc_radius=2.5,
    n_query_points=5000000,
):
    """Computes the nearest point to the binding site center within min_rec_dist nm"""
    from pykdtree.kdtree import KDTree

    rec_pos = rec_ref.positions.value_in_unit(unit.nanometer)
    tree = KDTree(rec_pos)

    all_poc_pos = rec_ref.positions[poc_alpha_indices].value_in_unit(unit.nanometer)
    poc_center = all_poc_pos.mean(axis=0)

    query_points = (
        np.random.uniform(-1, 1, size=(n_query_points, 3)) * max_poc_radius + poc_center
    )
    dists, idx = tree.query(query_points)

    candidates = query_points[dists > min_rec_dist]

    best_candidate = candidates[
        np.linalg.norm(candidates - poc_center, axis=1).argmin()
    ]

    return best_candidate

def compute_exit_path(
    rec_ref,
    poc_alpha_indices,
    min_rec_dist=0.9,
    max_poc_radius=2.5,
    clash_dist=0.22,
    n_query_points=500000,
    graph_k=5
):
    """Makes a KNN graph graph of points around the pocket and finds
    a path from the start (anything min_rec_dist away from the rec)
    to the center of the pocket without clashing with the receptor.
    The start of this path is the exit point (this function supercedes
    the above naive function)"""
    from pykdtree.kdtree import KDTree
    import graph_tool.all as gt

    rec_pos = rec_ref.positions.value_in_unit(unit.nanometer)
    tree = KDTree(rec_pos)

    all_poc_pos = rec_ref.positions[poc_alpha_indices].value_in_unit(unit.nanometer)
    poc_center = all_poc_pos.mean(axis=0)
    rec_center = rec_pos.mean(axis=0)

    query_points = (
        np.random.uniform(-1, 1, size=(n_query_points, 3)) * max_poc_radius + poc_center
    )
    dists, idx = tree.query(query_points)

    # first remove all points that are too close to the receptor
    query_points = query_points[dists > clash_dist]
    dists = dists[dists > clash_dist]

    # now the candidate points are those that are further than min_rec_dist
    candidate_mask = dists > min_rec_dist

    # find closest point to poc center

    query_tree = KDTree(query_points)
    _, idx = query_tree.query(poc_center.reshape(1, -1))
    start = idx[0]

    dists, idx = query_tree.query(query_points, k=graph_k)

    edges = []
    weights = []
    for i in range(len(query_points)):
        for j in range(1, graph_k):
            edges.append((i, idx[i, j]))
            weights.append(dists[i, j])

    G = gt.Graph(edges)
    eprop = G.new_edge_property("float")
    G.edge_properties["weight"] = eprop
    for e, w in zip(G.edges(), tqdm(weights)):
        eprop[e] = w

    dist_map, pred_map = gt.dijkstra_search(G, eprop, start)

    search_dists = dist_map.get_array()
    search_dists[~candidate_mask] = np.inf
    exit_point_idx = search_dists.argmin()

    cur_idx = exit_point_idx
    path = []
    while cur_idx != start:
        path.append(query_points[cur_idx])
        cur_idx = pred_map[cur_idx]
    path = np.array(path, dtype=np.float32)

    return path

def get_datagen_by_name(engine, target_id, name):
    """Get a datagen by name"""
    with Session(engine) as session:
        datagen = (
            session.query(PMFDatagen)
            .filter_by(target_id=target_id)
            .filter_by(name=name)
            .one()
        )
    return datagen

def get_datagen_for_target(
    engine,
    name,
    target_id,
    sim_name,
    alt_struct_id=None,
    sim_time=4.0,
    ligand_restraint_k=10.0,
    prod_ligand_restraint_k=None,
    ligand_restraint_version=1,
    remove_existing=False,
):
    """Makes a datagen instance if it doesn't already exist"""
    with Session(engine) as session:
        datagen = (
            session.query(PMFDatagen)
            .filter(PMFDatagen.target_id == target_id)
            .join(TargetMDTrajectory, PMFDatagen.md_traj_id == TargetMDTrajectory.id)
            .filter(TargetMDTrajectory.name == sim_name)
            .filter(TargetMDTrajectory.target_id == target_id)
            .filter(PMFDatagen.sim_time == sim_time)
            .filter(PMFDatagen.ligand_restraint_k == ligand_restraint_k)
            .filter(PMFDatagen.prod_ligand_restraint_k == prod_ligand_restraint_k)
            .filter(PMFDatagen.ligand_restraint_version == ligand_restraint_version)
            .one_or_none()
        )
        if remove_existing and datagen is not None:
            print("Removing existing datagen")
            session.delete(datagen)
            session.commit()
            datagen = None

        if datagen is None:

            md_traj = (
                session.query(TargetMDTrajectory)
                .filter_by(name=sim_name)
                .filter_by(target_id=target_id)
                .one()
            )
            assert (
                md_traj is not None
            ), f"No trajectory found for target with name {sim_name}"

            # poc_res_indices = target.binding_site
            # rec_ref = target.structure
            rec_ref, poc_res_indices = get_target_struct_and_pocket(
                engine, target_id, alt_struct_id
            )
            poc_indices = get_residue_atom_indices(rec_ref.topology, poc_res_indices)
            poc_alpha_indices = [
                i for i in poc_indices if i in get_CA_indices(rec_ref.topology)
            ]
            # exit_point = compute_exit_point(rec_ref, poc_alpha_indices)
            exit_path = compute_exit_path(rec_ref, poc_alpha_indices)
            exit_point = exit_path[0]

            datagen = PMFDatagen(
                name=name,
                target_id=target_id,
                sim_time=sim_time,
                ligand_restraint_k=ligand_restraint_k,
                prod_ligand_restraint_k=prod_ligand_restraint_k,
                ligand_restraint_version=ligand_restraint_version,
                exit_point=exit_point,
                exit_path=exit_path,
                md_traj_id=md_traj.id,
            )
            session.add(datagen)
            session.commit()
        assert (
            datagen.sim_time == sim_time
        ), "Already started datagen with a different sim_time"

        assert (
            datagen.ligand_restraint_k == ligand_restraint_k
        ), "Already started datagen with a different ligand_restraint_k"

        assert datagen.name == name, "Already started datagen with a different name"

    return datagen


def initialize_datagen(
    target_name,
    name,
    sim_name,
    alt_struct_name=None,
    remove_existing=False,
):
    """Create a datagen instance for the target and returns the id"""

    engine = get_engine()
    target_id = get_target_id_by_name(engine, target_name)
    alt_struct_id = get_alt_struct_id(engine, target_id, alt_struct_name)

    datagen = get_datagen_for_target(
        engine,
        name,
        target_id,
        sim_name,
        alt_struct_id=alt_struct_id,
        remove_existing=remove_existing,
    )
    return datagen.id


def make_single_datapoint(
    datagen,
    rec_ref,
    rec,
    rec_nowat,
    # rec_initial,
    poc_indices,
    poc_alpha_indices,
    # reporter,
    act_fps,
    cofactors,
    tanimoto_cutoff=0.3,
    debug=False,
):
    """Makes a single datapoint and uploads to google cloud.
    Will reject any similes with a tanimoto similarity greater
    than tanimoto_cutoff to any known active"""

    print("Making a single datapoint")

    # generate a gs path and save the result
    id = uuid4()

    # now get a random ligand
    smi = get_random_zinc_smiles()
    zinc_fp = get_fp(Chem.MolFromSmiles(smi))
    if len(act_fps) > 0:
        max_tanimoto = batch_tanimoto_faster(zinc_fp, act_fps).max()
        if max_tanimoto > tanimoto_cutoff:
            print(
                f"Rejected ligand with tanimoto similarity {max_tanimoto} to known actives"
            )
            return

    rec_poc = rec.positions[poc_alpha_indices].value_in_unit(unit.nanometers)
    ref_rec_poc = rec_ref.positions[poc_alpha_indices].value_in_unit(unit.nanometers)

    rec_origin = rec_poc.mean(axis=0)
    ref_rec_origin = ref_rec_poc.mean(axis=0)

    R, t = find_rigid_alignment(ref_rec_poc - ref_rec_origin, rec_poc - rec_origin)

    # relative to pocket origin
    exit_vector = R.dot((datagen.exit_point - ref_rec_origin)) * unit.nanometers
    if datagen.exit_path is not None:
        exit_vector_path = (
            np.einsum("ij,bj", R, (datagen.exit_path - ref_rec_origin))
            * unit.nanometers
        )
    else:
        exit_vector_path = None

    if datagen.ligand_restraint_k is None:
        # dock the ligand, then randomize its pose
        lig_rd = run_unidock(rec_nowat, [smi], poc_indices)[0]
        lig = rdkit_to_modeller(lig_rd)

    else:

        smi = remove_salt_from_smiles(smi)
        prot_smi = protonate_smiles(smi)
        lig_rd = Chem.MolFromSmiles(prot_smi)
        lig_rd = Chem.AddHs(lig_rd)

        AllChem.EmbedMolecule(lig_rd)
        AllChem.UFFOptimizeMolecule(lig_rd)
        lig = rdkit_to_modeller(lig_rd)

        # center ligand on pocket origin
        lig_pos = lig.positions.value_in_unit(unit.nanometers)
        lig_pos_centered = lig_pos - lig_pos.mean(axis=0) + rec_origin
        lig = app.Modeller(lig.topology, lig_pos_centered * unit.nanometers)

    # lig = randomize_ligand_pose(lig, exit_vector)
    lig, alpha = randomize_ligand_pose(
        lig,
        exit_vector,
        exit_vector_path,
        datagen.max_trans * unit.nanometers,
        datagen.trans_std * unit.nanometers,
    )

    # now alchemically add ligand to rec and simulate

    pmf_sim = PMFSim(rec, lig, lig_rd, rec_ref, poc_indices, extra_mols=cofactors)

    if datagen.ligand_restraint_k is None:
        state = pmf_sim.run_alchemical_lr_sim()
    else:

        if debug:
            save_modeller_pdb(pmf_sim.mols["lr"], "output/init.pdb")

        lig_restraint_k = (
            datagen.ligand_restraint_k * unit.kilojoules_per_mole / unit.nanometer**2
            if datagen.ligand_restraint_k is not None
            else None
        )
        prod_lig_restraint_k = (
            datagen.prod_ligand_restraint_k
            * unit.kilojoules_per_mole
            / unit.nanometer**2
            if datagen.prod_ligand_restraint_k is not None
            else None
        )

        state = pmf_sim.run_alchemical_lr_sim(
            freeze_ligand=False,
            lig_restraint_k=lig_restraint_k,
            lig_restraint_version=datagen.ligand_restraint_version,
        )

        pmf_sim.set_init_lig_pos_to_state(state)
        state.positions[pmf_sim.lig_ion_indices] = (
            pmf_sim.init_lig_pos * unit.nanometers
        )

        lr = modeller_with_state(pmf_sim.mols["lr"], state)
        if debug:
            save_modeller_pdb(lr, "output/after_alc.pdb")

    dcd_fname = f"output/prod.dcd" if debug else None

    lig_md_positions, energies, forces, hessians = pmf_sim.run_pmf_sim(
        state,
        total_time=datagen.sim_time * unit.nanoseconds,
        freeze_ligand=datagen.prod_ligand_restraint_k is None,
        lig_restraint_k=prod_lig_restraint_k,
        lig_restraint_version=datagen.ligand_restraint_version,
        dcd_fname=dcd_fname,
        # only save hessians if ligand is frozen
        save_hessians=False # prod_lig_restraint_k is None,
    )

    lig_nb_params = get_nonbonded_params(pmf_sim.systems["lig_noion"])

    if (
        datagen.prod_ligand_restraint_k is not None
        and datagen.ligand_restraint_version == 1
    ):
        og_pos = np.array(pmf_sim.mols["lr"].positions.value_in_unit(unit.nanometers))
        restraint_com = og_pos[pmf_sim.lig_noion_indices].mean(axis=0)
        # put back in reference frame
        restraint_com = (
            pmf_sim.ref_R.dot((restraint_com - pmf_sim.sim_origin).T).T
            + pmf_sim.ref_origin
        )
    else:
        restraint_com = None

    result = PMFSimResult(
        forces=forces,
        energies=energies,
        hessians=hessians,
        lig_positions=pmf_sim.ref_lig_pos,
        lig_nbparams=lig_nb_params,
        lig_rd=lig_rd,
        lig_md_positions=lig_md_positions,
        restraint_com=restraint_com,
    )

    gs_path = f"{datagen.get_gs_path()}/{id}"
    result.save_gs(gs_path)

    print(f"Saved to {gs_path}")

    return id


def run(datagen_id, rec_burnin=10, debug=False):  # rec_burnin in ns (1 frame per ns)
    """Make datapoints forever"""

    n_cpu = os.cpu_count()
    n_gpu = len(GPUtil.getGPUs())
    gpu_name = GPUtil.getGPUs()[0].name

    vast_instance_id = None
    if "VAST_CONTAINERLABEL" in os.environ:
        vast_instance_id = os.environ["VAST_CONTAINERLABEL"]

    engine = get_engine()

    with Session(engine) as ses:
        datagen = ses.query(PMFDatagen).filter_by(id=datagen_id).one()
        target = ses.query(Target).filter_by(id=datagen.target_id).one()
        cofactors = target.cofactors
        db_traj = ses.query(TargetMDTrajectory).filter_by(id=datagen.md_traj_id).one()
        target_name = target.name
        target_id = target.id
        sim_name = db_traj.name

    print("=====================================")
    print(f"Running datagen for {target_name} with simulation {sim_name}")
    print("=====================================")

    poc_res_indices = target.binding_site
    rec_ref = target.structure

    poc_indices = get_residue_atom_indices(rec_ref.topology, poc_res_indices)
    poc_alpha_indices = [
        i for i in poc_indices if i in get_CA_indices(rec_ref.topology)
    ]

    # db_traj = get_target_md_traj(engine, target.id)
    reporter = GSReporter(db_traj.get_gs_path(), db_traj.initial_structure, "r", download=True)
    rec_initial = db_traj.initial_structure

    query = f"SELECT mol.mol AS smiles from molecules mol JOIN activities a on a.mol_id = mol.id WHERE a.target_id = {target_id}"
    act_df = exec_sql(engine, query)
    act_fps = np.array(
        [get_fp(Chem.MolFromSmiles(smiles)) for smiles in tqdm(act_df.smiles)],
        dtype=bool,
    )

    while True:
        error_type = None
        error_msg = None
        success = False
        tb = None
        uid = None

        start_time = datetime.now()
        try:

            if not torch.cuda.is_available():
                # don't want to innundate the logs with this
                time.sleep(30)
                raise FatalError("No GPU available")

            # first initialize rec positions randomly according to the MD trajectory
            rand_idx = np.random.randint(rec_burnin, len(reporter))
            state = reporter[rand_idx]

            rec = modeller_with_state(rec_initial, state)

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

            rec_nowat = app.Modeller(
                rec_ref.topology, rec.positions[: len(rec_ref.positions)]
            )

            uid = make_single_datapoint(
                datagen,
                rec_ref,
                rec,
                rec_nowat,
                poc_indices,
                poc_alpha_indices,
                act_fps,
                cofactors,
                debug=debug,
            )
            success = True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("Error in datagen, retrying")
            traceback.print_exc()
            tb = traceback.format_exc()

            error_type = type(e).__name__
            error_msg = str(e)

        total_time = datetime.now() - start_time

        # log result to database
        try:
            with Session(engine) as ses:
                result = PMFDatagenResult(
                    version=1,
                    datagen_id=datagen_id,
                    n_cpu=n_cpu,
                    n_gpu=n_gpu,
                    gpu_name=gpu_name,
                    success=success,
                    start_time=start_time,
                    total_time=total_time,
                    error_type=error_type,
                    error_msg=error_msg,
                    traceback=tb,
                    uid=uid,
                    vastai_instance_id=vast_instance_id,
                )
                ses.add(result)
                ses.commit()
        except KeyboardInterrupt:
            raise
        except:
            print("Error saving result to db")
            traceback.print_exc()


if __name__ == "__main__":

    action = sys.argv[1]

    match action:
        case "init":

            parser = argparse.ArgumentParser()
            parser.add_argument("action", type=str, help="init or run")
            parser.add_argument("target_name", type=str)
            parser.add_argument("name", type=str)
            parser.add_argument("--sim_name", type=str, default="apo")
            parser.add_argument("--alt_struct_name", type=str, default=None)
            parser.add_argument("--remove", action="store_true")

            args = parser.parse_args()

            if args.sim_name == "None":
                args.sim_name = None

            datagen_id = initialize_datagen(
                args.target_name,
                args.name,
                args.sim_name,
                alt_struct_name=args.alt_struct_name,
                remove_existing=args.remove,
            )
            print(datagen_id)

            # save rec and exit point
            engine = get_engine()
            target_id = get_target_id_by_name(engine, args.target_name)
            alt_struct_id = get_alt_struct_id(engine, target_id, args.alt_struct_name, verbose=False)

            rec, binding_site = get_target_struct_and_pocket(engine, target_id, alt_struct_id)
            with Session(engine) as ses:
                datagen = ses.query(PMFDatagen).filter_by(id=datagen_id).one()
                exit_path = datagen.exit_path

            rec_path = f"output/rec.pdb"
            save_modeller_pdb(rec, rec_path)
            print(f"Saved rec to {rec_path}")

            exit_path_path = f"output/exit_path.pdb"
            save_pointcloud_pdb(exit_path, exit_path_path)
            print(f"Saved exit path to {exit_path_path}")

        case "run":
            datagen_id = int(sys.argv[2])
            debug = "--debug" in sys.argv
            run(datagen_id, debug=debug)
