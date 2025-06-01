# utilities for interfacing with cockroachdb

from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
import gzip
import io
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from common.utils import CONFIG, compress_mol, decompress_mol, get_output_dir
from sqlalchemy import create_engine, JSON, delete
from psycopg2.extras import execute_values
import sqlalchemy.types as types
from sqlalchemy import ForeignKey, MetaData, Column, Table
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from sqlalchemy.schema import Index
from rdkit import Chem

try:
    from openmm import app
    import openmm as mm
except ModuleNotFoundError:
    pass

db_params = json.load(open("secrets/cockroachdb.json"))


def db_gs_dir():
    """Returns the directory where we store big binary files associated with the database.
    This changes depending on whether we're using the local or production database."""
    name = "db"
    if "POSTGRES_HOST" in os.environ:
        name = "debug_db"
    return f"gs://{CONFIG.storage.bucket}/{name}"


def get_engine(echo=False):

    # try local connection first
    if "POSTGRES_HOST" in os.environ:
        print("Using local SQL connection")
        host = os.environ["POSTGRES_HOST"]
        port = os.environ["POSTGRES_PORT"]
        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        return create_engine(
            f"postgresql://{user}:{password}@{host}:{port}",
            echo=echo,
        )

    print("Using CockroachDB connection")
    return create_engine(
        f'cockroachdb://{db_params["username"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/defaultdb?sslmode=verify-full',
        echo=echo,
        pool_pre_ping=True,
        connect_args={
            "sslrootcert": "secrets/postgresql_root.crt",
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
    )


def chunker(seq, size):
    # from http://stackoverflow.com/a/434328
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def insert_table(engine, df, table_name, chunksize=100, append=True):
    """Insert a dataframe into the database with a progress bar. By default, appends to the table"""
    # modified from https://stackoverflow.com/questions/39494056/progress-bar-for-pandas-dataframe-to-sql

    conn = engine.raw_connection()

    with tqdm(total=len(df) // chunksize) as pbar:
        for i, cdf in enumerate(chunker(df, chunksize)):
            replace = "replace" if i == 0 and not append else "append"
            cdf.to_sql(con=conn, name=table_name, if_exists=replace, index=False)
            pbar.update(chunksize)


def exec_sql(engine, query, *args, fetch=True, transaction=True, **kwargs):
    """Executes a SQL script, commits, and returns the result as a dataframe"""

    con = engine.raw_connection()
    cursor = con.cursor()
    if not transaction:
        cursor.execute("ROLLBACK")

    cursor.execute(query, *args, **kwargs)
    if transaction:
        con.commit()

    if fetch:
        rows = cursor.fetchall()
        cols = [col.name for col in cursor.description]
        ret = pd.DataFrame(rows, columns=cols)
    else:
        ret = None

    cursor.close()

    return ret


def exec_sql_values(engine, query, *args, fetch=True, **kwargs):
    """Same as above, but uses execute_values"""

    con = engine.raw_connection()
    cursor = con.cursor()

    execute_values(cursor, query, *args, **kwargs)
    con.commit()

    if fetch:
        rows = cursor.fetchall()
        cols = [col.name for col in cursor.description]
        ret = pd.DataFrame(rows, columns=cols)
    else:
        ret = None

    cursor.close()

    return ret


class Mol2D(types.TypeDecorator):
    """Serialize rdkit mols to and from smiles"""

    impl = types.String

    def process_bind_param(self, value, dialect):
        if isinstance(value, str):  # useful for importing from csv
            return value
        return Chem.MolToSmiles(value)

    def process_result_value(self, value, dialect):
        return Chem.MolFromSmiles(value)


class Mol3D(types.TypeDecorator):
    """Serialize mols by gzipp'd mol blocks"""

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        return compress_mol(value)

    def process_result_value(self, value, dialect):
        return decompress_mol(value)


class CifStructure(types.TypeDecorator):
    """Serialize an OpenMM Modeller object to a gzipped CIF file"""

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        buffer = io.StringIO()
        app.PDBxFile.writeFile(value.topology, value.positions, buffer)
        return gzip.compress(buffer.getvalue().encode())

    def process_result_value(self, value, dialect):
        cif_str = gzip.decompress(value).decode()
        buffer = io.StringIO(cif_str)
        cif = app.PDBxFile(buffer)
        modeller = app.Modeller(cif.topology, cif.getPositions(asNumpy=True))
        return modeller


class BindingSite(types.TypeDecorator):
    """Represents an a binding site as a numpy array of residue indices"""

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        if isinstance(value, bytes):
            return value
        return value.tobytes()

    def process_result_value(self, value, dialect):
        return np.frombuffer(value, dtype=np.int64)


class NumpyArray(types.TypeDecorator):
    """Represents an arbitrary numpy array as a binary blob using
    np.save and np.load"""

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        if isinstance(value, bytes):
            return value
        if value is None:
            return None
        buffer = io.BytesIO()
        np.save(buffer, value)
        return buffer.getvalue()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        buffer = io.BytesIO(value)
        return np.load(buffer)


class OMMSystem(types.TypeDecorator):
    """Serialize an OpenMM System object to a gzipped xml file"""

    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        buffer = io.StringIO()
        buffer.write(mm.XmlSerializer.serialize(value))
        return gzip.compress(buffer.getvalue().encode())

    def process_result_value(self, value, dialect):
        xml_str = gzip.decompress(value).decode()
        buffer = io.StringIO(xml_str)
        return mm.XmlSerializer.deserialize(buffer.getvalue())


convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def get_db_gs_path(table, id):
    """Return the gs path for a given table and id"""
    return f"{db_gs_dir()}/{table}/{id}"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=convention)

    def get_gs_path(self):
        """For tables with a single gs path with associated data"""
        return get_db_gs_path(type(self).__tablename__, self.id)


class TargetType(Enum):
    """Type of macromolecular target"""

    PROTEIN = "Protein"
    RNA = "RNA"
    DNA = "DNA"


class ActivityType(Enum):
    """Type of binding activity"""

    IC50 = "IC50"
    Kd = "Kd"
    Ki = "Ki"


class Target(Base):
    """Macromolecular target with a binding site we can screen against"""

    __tablename__ = "targets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)  # human-readable name
    type: Mapped[TargetType] = mapped_column(nullable=False)
    pdb: Mapped[str] = mapped_column(nullable=True)  # PDB ID if we have one
    structure: Mapped[bytes] = mapped_column(
        CifStructure, nullable=False
    )  # 3D structure
    binding_site: Mapped[bytes] = mapped_column(
        BindingSite, nullable=True
    )  # binding site as residue indices
    cofactors: Mapped[bytes] = mapped_column(
        Mol3D, nullable=True
    )  # cofactors as 3D molecules. Needed to parameterize simulations


class ProteinTarget(Base):

    __tablename__ = "protein_targets"

    id: Mapped[int] = mapped_column(ForeignKey("targets.id"), primary_key=True)
    target: Mapped[Target] = relationship("Target")
    uniprot: Mapped[str] = mapped_column(nullable=True)
    sequence: Mapped[str] = mapped_column(nullable=False)  # amino acid sequence


class RNATarget(Base):

    __tablename__ = "rna_targets"

    id: Mapped[int] = mapped_column(ForeignKey("targets.id"), primary_key=True)
    target: Mapped[Target] = relationship("Target")
    sequence: Mapped[str] = mapped_column(nullable=False)  # nucleotide sequence


class DNATarget(Base):

    __tablename__ = "dna_targets"

    id: Mapped[int] = mapped_column(ForeignKey("targets.id"), primary_key=True)
    target: Mapped[Target] = relationship("Target")
    sequence: Mapped[str] = mapped_column(nullable=False)  # nucleotide sequence


class ScreenMol(Base):
    """Molecules that we can screen and buy. Eg Enamine but we also
    want to support other vendors"""

    __tablename__ = "screen_mols"

    id: Mapped[int] = mapped_column(primary_key=True)
    collection_id: Mapped[str] = mapped_column(nullable=False)
    collection: Mapped[str] = mapped_column(nullable=False)
    library: Mapped[str] = mapped_column(nullable=True)
    mol: Mapped[str] = mapped_column(Mol2D, nullable=False)


class PMFScreenResult(Base):
    """Results from screening using PMFNet. Includes the score and the 3D pose"""

    __tablename__ = "pmf_screen_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    mol_id: Mapped[int] = mapped_column(ForeignKey("screen_mols.id"), nullable=False)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )

    # nullable only true because of the old BRD4 results
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=True)

    score: Mapped[float] = mapped_column(nullable=True)
    mol: Mapped[bytes] = mapped_column(Mol3D, nullable=True)

    extra_data: Mapped[JSON] = mapped_column(JSON, nullable=True)
    protocol_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_scoring_protocols.id"), nullable=True
    )

    model: Mapped["Model"] = relationship("Model")
    target: Mapped[Target] = relationship("Target")
    protocol: Mapped["PMFScoringProtocol"] = relationship("PMFScoringProtocol")


class Molecule(Base):
    """Standardized molecules with known activities or structures in the the
    databases we pull from. Use chembl standardization before inserting"""

    __tablename__ = "molecules"

    id: Mapped[int] = mapped_column(primary_key=True)
    mol: Mapped[str] = mapped_column(Mol2D, nullable=False)


class Activity(Base):
    """Activities for molecules in the database."""

    __tablename__ = "activities"

    id: Mapped[int] = mapped_column(primary_key=True)
    mol_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), nullable=False)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )

    type: Mapped[ActivityType] = mapped_column(nullable=False)
    pK: Mapped[float] = mapped_column(nullable=False)

    mol: Mapped[Molecule] = relationship("Molecule")
    target: Mapped[Target] = relationship("Target")


class CoStructure(Base):
    """Co-crystal structures of molecules in the database"""

    __tablename__ = "co_structures"

    id: Mapped[int] = mapped_column(primary_key=True)
    mol_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), nullable=False)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    rec_structure: Mapped[bytes] = mapped_column(CifStructure, nullable=False)
    lig_structure: Mapped[bytes] = mapped_column(Mol3D, nullable=False)

    pdb: Mapped[str] = mapped_column(nullable=True)
    # for now we get pK from plinder which may be different from how we handle chembl actives.
    # in the future we should have this be linked to the activities table
    pK: Mapped[float] = mapped_column(nullable=True)

    target: Mapped[Target] = relationship("Target")
    mol: Mapped[Molecule] = relationship("Molecule")


class AlternativeStructure(Base):
    """Alternative structures for a target that we can use for docking.
    Always aligned to the reference structure"""

    __tablename__ = "alternative_structures"

    id: Mapped[int] = mapped_column(primary_key=True)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    structure: Mapped[bytes] = mapped_column(CifStructure, nullable=False)
    name: Mapped[str] = mapped_column(nullable=True)  # human-readable name

    # if this came from an MD trajectory
    md_traj_id: Mapped[int] = mapped_column(
        ForeignKey("target_md_trajectories.id"), nullable=True
    )
    md_traj_index: Mapped[int] = mapped_column(nullable=True)

    # if this came from the PDB
    pdb: Mapped[str] = mapped_column(nullable=True)


class Decoy(Base):
    """Property-matched decoys for targets"""

    __tablename__ = "decoys"

    id: Mapped[int] = mapped_column(primary_key=True)
    mol_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), nullable=False)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    # which molecule is this decoy coming from? Alas I haven't always saved this info
    parent_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), nullable=True)

    mol: Mapped[Molecule] = relationship("Molecule", foreign_keys=[mol_id])
    target: Mapped[Target] = relationship("Target")
    parent: Mapped[Molecule] = relationship("Molecule", foreign_keys=[parent_id])


class TargetMDTrajectory(Base):
    """MD simulation results for a target"""

    __tablename__ = "target_md_trajectories"

    id: Mapped[int] = mapped_column(primary_key=True)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    system: Mapped[bytes] = mapped_column(OMMSystem, nullable=False)
    initial_structure: Mapped[bytes] = mapped_column(CifStructure, nullable=False)

    co_structure_id: Mapped[int] = mapped_column(
        ForeignKey("co_structures.id"), nullable=True
    )  # co-structure used for the simulation (if null, assume we're using the reference structure)
    path: Mapped[str] = mapped_column(nullable=True)  # path to the gs zarr file
    name: Mapped[str] = mapped_column(nullable=True)  # human-readable name

    target: Mapped[Target] = relationship("Target")


class YankInput(Base):
    """Input structure and system for Yank ABFE simulations"""

    __tablename__ = "yank_inputs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=True)  # human-readable name

    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )

    # TODO: unite screenmol and molecules!
    mol_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), nullable=True)
    screen_mol_id: Mapped[int] = mapped_column(
        ForeignKey("screen_mols.id"), nullable=True
    )

    # null if we used docking to generate the complex structure
    costructure_id: Mapped[int] = mapped_column(
        ForeignKey("co_structures.id"), nullable=True
    )

    complex_structure: Mapped[bytes] = mapped_column(CifStructure, nullable=False)
    ligand_structure: Mapped[bytes] = mapped_column(CifStructure, nullable=False)

    complex_system: Mapped[bytes] = mapped_column(OMMSystem, nullable=False)
    ligand_system: Mapped[bytes] = mapped_column(OMMSystem, nullable=False)

    cfg: Mapped[JSON] = mapped_column(JSON, nullable=False)  # stringified yaml config

    target: Mapped[Target] = relationship("Target")
    mol: Mapped[Molecule] = relationship("Molecule")
    screen_mol: Mapped[ScreenMol] = relationship("ScreenMol")
    costructure: Mapped[CoStructure] = relationship("CoStructure")


class YankResult(Base):

    __tablename__ = "yank_results"

    input_id: Mapped[int] = mapped_column(
        ForeignKey("yank_inputs.id"), primary_key=True
    )

    dG: Mapped[float] = mapped_column(nullable=True)
    dG_uncertainty: Mapped[float] = mapped_column(nullable=True)

    complex_transition_mat: Mapped[bytes] = mapped_column(NumpyArray, nullable=True)
    solvent_transition_mat: Mapped[bytes] = mapped_column(NumpyArray, nullable=True)

    complex_max_transition_mat: Mapped[float] = mapped_column(nullable=True)
    solvent_max_transition_mat: Mapped[float] = mapped_column(nullable=True)

    complex_perron_eigenvalue: Mapped[float] = mapped_column(nullable=True)
    solvent_perron_eigenvalue: Mapped[float] = mapped_column(nullable=True)

    input: Mapped[YankInput] = relationship("YankInput")


class PMFDatagen(Base):
    """The raw simulation files for a dataset"""

    __tablename__ = "pmf_datagens"
    id: Mapped[int] = mapped_column(primary_key=True)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    md_traj_id: Mapped[int] = mapped_column(
        ForeignKey("target_md_trajectories.id"), nullable=True
    )

    sim_time: Mapped[float] = mapped_column(
        nullable=False
    )  # Total simulation time in ns

    ligand_restraint_k: Mapped[float] = mapped_column(
        nullable=True
    )  # k in kJ/mol/nm^2 for the ligand-pocket distance restraint
    # null if ligand atoms are frozen during alchemical simulation

    prod_ligand_restraint_k: Mapped[float] = mapped_column(
        nullable=True,
    )  # k in kJ/mol/nm^2 for the ligand-pocket distance restraint
    # null if ligand atoms are frozen during _production_ simulation

    ligand_restraint_version: Mapped[int] = mapped_column(
        nullable=False,
        server_default="0",
    )  # version of the ligand restraint code
    # 0 for pocket distance COM restraint
    # 1 for COM restraint to fixed point

    max_trans: Mapped[float] = mapped_column(
        nullable=False,
        default=3.0,
    )  # max distance from the pocket center to translate ligand (nm)
    trans_std: Mapped[float] = mapped_column(
        nullable=False,
        default=0.25,
    )  # std deviation of the translation (nm)

    # nearest point to the ligand binding site > 0.9 nm away from the protein
    exit_point: Mapped[bytes] = mapped_column(NumpyArray, nullable=False)

    # path from the exit point to the pocket center. Only used in the new datagen code
    exit_path: Mapped[bytes] = mapped_column(NumpyArray, nullable=True)

    name: Mapped[str] = mapped_column(nullable=True)  # human-readable name

    target: Mapped[Target] = relationship("Target")
    md_traj: Mapped[TargetMDTrajectory] = relationship("TargetMDTrajectory")


class PMFDatagenResult(Base):
    """Stores successful and failed datagen tasks. Useful for removing
    instances that aren't doing anything"""

    __tablename__ = "pmf_datagen_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(nullable=False)
    datagen_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_datagens.id"), nullable=False
    )

    n_cpu: Mapped[int] = mapped_column(nullable=False)
    n_gpu: Mapped[int] = mapped_column(nullable=False)
    gpu_name: Mapped[str] = mapped_column(nullable=True)

    success: Mapped[bool] = mapped_column(nullable=False)
    start_time: Mapped[datetime] = mapped_column(nullable=False)
    total_time: Mapped[timedelta] = mapped_column(nullable=False)

    error_type: Mapped[str] = mapped_column(nullable=True)
    error_msg: Mapped[str] = mapped_column(nullable=True)
    traceback: Mapped[str] = mapped_column(nullable=True)

    uid: Mapped[str] = mapped_column(nullable=True)  # null for failed tasks
    vastai_instance_id: Mapped[str] = mapped_column(nullable=True)


Index(
    "pmf_datagen_results_start_time_target_id_idx",
    PMFDatagenResult.start_time,
    PMFDatagenResult.datagen_id,
)

pmf_datagen_combo_table = Table(
    "pmf_dataset_combos",
    Base.metadata,
    Column("dataset_id", ForeignKey("pmf_datasets.id"), primary_key=True),
    Column("sub_dataset_id", ForeignKey("pmf_datasets.id"), primary_key=True),
)


class PMFDataset(Base):
    """A dataset of PMFNet training data"""

    __tablename__ = "pmf_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    datagen_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_datagens.id"), nullable=True
    )
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )

    size: Mapped[int] = mapped_column(nullable=False)  # Total length of the dataset
    burnin_time: Mapped[float] = mapped_column(nullable=False)  # Burnin time in ns
    production_time: Mapped[float] = mapped_column(
        nullable=True
    )  # Production time in ns. By default this is all the available time

    name: Mapped[str] = mapped_column(nullable=True)  # human-readable name

    datagen: Mapped[PMFDatagen] = relationship("PMFDatagen")

    # If false, we place all the test data into the training set
    has_test_split: Mapped[bool] = mapped_column(nullable=False, default=True)

    # if datagen_id is null, this must be a combo dataset
    sub_datasets = relationship(
        "PMFDataset",
        secondary=pmf_datagen_combo_table,
        primaryjoin=id == pmf_datagen_combo_table.c.dataset_id,
        secondaryjoin=id == pmf_datagen_combo_table.c.sub_dataset_id,
    )

    target: Mapped[Target] = relationship("Target")


class Model(Base):
    """Represents a model that can be used for docking and/or scoring"""

    __tablename__ = "models"
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str]

    __mapper_args__ = {
        "polymorphic_identity": "models",
        "polymorphic_on": "type",
    }


class BaselineModel(Model):
    """Gnina, Vina, or other baselines"""

    __tablename__ = "baseline_models"
    id: Mapped[int] = mapped_column(ForeignKey("models.id"), primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "baseline_models",
    }


class PMFModel(Model):
    """The neural network model"""

    __tablename__ = "pmf_models"
    id: Mapped[int] = mapped_column(ForeignKey("models.id"), primary_key=True)

    git_hash: Mapped[str] = mapped_column(nullable=True)
    config: Mapped[JSON] = mapped_column(JSON, nullable=False)

    wandb_id: Mapped[str] = mapped_column(nullable=True)

    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_datasets.id"), nullable=False
    )

    target: Mapped[Target] = relationship("Target")
    dataset: Mapped[PMFDataset] = relationship("PMFDataset")

    __mapper_args__ = {
        "polymorphic_identity": "pmf_models",
    }


class PMFScoringProtocol(Base):
    """Protocol for scoring PMF models -- if we use the solvation scoring, how many steps if so, etc"""

    __tablename__ = "pmf_scoring_protocols"
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str]
    name: Mapped[str] = mapped_column(nullable=False)

    minimize_mm: Mapped[bool] = mapped_column(
        nullable=False, default=True
    )  # minimize with MM before scoring

    mm_restraint_k: Mapped[float] = mapped_column(
        nullable=True, default=5.0
    )  # kJ/mol/nm^2 for the MM position restraint

    minimize_pmf: Mapped[bool] = mapped_column(
        nullable=False, default=False
    )  # minimize with PMF before scoring

    __mapper_args__ = {
        "polymorphic_identity": "models",
        "polymorphic_on": "type",
    }


class SolvationScoringProtocol(PMFScoringProtocol):
    """Protocol for using the PMF model + solvation simulation"""

    __tablename__ = "solvation_scoring_protocols"
    id: Mapped[int] = mapped_column(
        ForeignKey("pmf_scoring_protocols.id"), primary_key=True
    )

    n_steps: Mapped[int] = mapped_column(nullable=False)
    n_burnin: Mapped[int] = mapped_column(nullable=False)
    report_interval: Mapped[int] = mapped_column(nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "solvation_scoring_protocols",
    }


class DockingResult(Base):
    """Stores results of docking runs on activity data"""

    __tablename__ = "docking_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    mol_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), nullable=False)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    # if using an alternative structure. Null if using the reference structure
    alt_structure_id: Mapped[int] = mapped_column(
        ForeignKey("alternative_structures.id"), nullable=True
    )

    # if we're rescoring an earlier docking run
    docking_result_id: Mapped[int] = mapped_column(
        ForeignKey("docking_results.id"), nullable=True
    )

    protocol_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_scoring_protocols.id"), nullable=True
    )

    # can be null if the docking failed
    score: Mapped[float] = mapped_column(nullable=True)
    pose: Mapped[bytes] = mapped_column(Mol3D, nullable=True)
    # some docking methods (eg GNINA) produce multiple scores
    extra_data: Mapped[JSON] = mapped_column(JSON, nullable=True)

    dt: Mapped[float] = mapped_column(nullable=True)  # docking time in seconds

    model: Mapped[Model] = relationship("Model")
    mol: Mapped[Molecule] = relationship("Molecule")
    target: Mapped[Target] = relationship("Target")
    alt_structure: Mapped[AlternativeStructure] = relationship("AlternativeStructure")
    docking_result: Mapped["DockingResult"] = relationship(
        "DockingResult", foreign_keys=[docking_result_id]
    )
    protocol: Mapped[PMFScoringProtocol] = relationship("PMFScoringProtocol")


class ScoringResult(Base):
    """Stores the results of scoring (minimization) on co-structure data"""

    __tablename__ = "scoring_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    costructure_id: Mapped[int] = mapped_column(
        ForeignKey("co_structures.id"), nullable=False
    )
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    protocol_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_scoring_protocols.id"), nullable=True
    )
    # if using an alternative structure. Null if using the reference structure
    alt_structure_id: Mapped[int] = mapped_column(
        ForeignKey("alternative_structures.id"), nullable=True
    )

    score: Mapped[float] = mapped_column(nullable=True)
    pose: Mapped[bytes] = mapped_column(Mol3D, nullable=True)
    # some docking methods (eg GNINA) produce multiple scores
    extra_data: Mapped[JSON] = mapped_column(JSON, nullable=True)

    dt: Mapped[float] = mapped_column(nullable=True)  # docking time in seconds

    model: Mapped[Model] = relationship("Model", foreign_keys=[model_id])
    costructure: Mapped[CoStructure] = relationship("CoStructure")
    protocol: Mapped[PMFScoringProtocol] = relationship("PMFScoringProtocol")
    alt_strucure: Mapped[AlternativeStructure] = relationship(
        "AlternativeStructure", foreign_keys=[alt_structure_id]
    )

class DatasetMolecule(Base):
    """A 3D molecule that we took from a dataset"""

    __tablename__ = "dataset_molecules"
    id: Mapped[int] = mapped_column(primary_key=True)
    mol_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_datasets.id"), nullable=False, index=True
    )

    lig_structure: Mapped[bytes] = mapped_column(Mol3D, nullable=False)

    split: Mapped[str] = mapped_column(nullable=False)  # train, val, test
    index: Mapped[int] = mapped_column(nullable=False)  # index in the dataset

    mol: Mapped[Molecule] = relationship("Molecule")
    dataset: Mapped[PMFDatagen] = relationship("PMFDataset")


class TruePMFCampaign(Base):
    """For running a bunch of MAF sims to estimate the true
    PMF for a pose"""

    __tablename__ = "true_pmf_campaigns"
    id: Mapped[int] = mapped_column(primary_key=True)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    mol_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), nullable=False)
    # justr need datagen for exit point
    datagen_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_datagens.id"), nullable=False
    )
    md_traj_id: Mapped[int] = mapped_column(
        ForeignKey("target_md_trajectories.id"), nullable=False
    )
    docking_result_id: Mapped[int] = mapped_column(
        ForeignKey("docking_results.id"), nullable=True
    )  # if this came from a docking result

    pose: Mapped[bytes] = mapped_column(Mol3D, nullable=False)

    name: Mapped[str] = mapped_column(nullable=True)  # human-readable namme
    sim_time: Mapped[float] = mapped_column(
        nullable=False
    )  # Total simulation time in ns

    target: Mapped[Target] = relationship("Target")
    mol: Mapped[Molecule] = relationship("Molecule")
    datagen: Mapped[PMFDatagen] = relationship("PMFDatagen")
    md_traj: Mapped[TargetMDTrajectory] = relationship("TargetMDTrajectory")
    docking_result: Mapped[DockingResult] = relationship("DockingResult")


class MAFResult(Base):
    """Stores the results of running MAF simulations
    for the PMF campaign to gcs"""

    __tablename__ = "maf_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    pmf_campaign_id: Mapped[int] = mapped_column(
        ForeignKey("true_pmf_campaigns.id"), nullable=False
    )

    t: Mapped[float] = mapped_column(
        nullable=False
    )  # 0 is the original position, 1 is translated to exit point

    pmf_campaign: Mapped[TruePMFCampaign] = relationship("TruePMFCampaign")


class ScreenResult(Base):
    """Stores results of docking runs on screening data"""

    __tablename__ = "screen_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )
    mol_id: Mapped[int] = mapped_column(ForeignKey("screen_mols.id"), nullable=False)

    protocol_id: Mapped[int] = mapped_column(
        ForeignKey("pmf_scoring_protocols.id"), nullable=True
    )

    score: Mapped[float] = mapped_column(nullable=True)
    pose: Mapped[bytes] = mapped_column(Mol3D, nullable=True)
    # some docking methods (eg GNINA) produce multiple scores
    extra_data: Mapped[JSON] = mapped_column(JSON, nullable=True)

    model: Mapped[Model] = relationship("Model")
    mol: Mapped[Molecule] = relationship("ScreenMol")
    target: Mapped[Target] = relationship("Target")
    protocol: Mapped[PMFScoringProtocol] = relationship("PMFScoringProtocol")


class TaskStatus(Enum):
    """Status of a task"""

    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"


class Task(Base):
    """DIY replacement for celery task queue. Supposed to be simple"""

    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    # args and kwargs, must be json-encoded
    args: Mapped[JSON] = mapped_column(JSON, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(nullable=False)

    # when to rerun the task
    expires_at: Mapped[datetime] = mapped_column(nullable=True)


Index(
    "task_name_status_expires_at_idx",
    Task.name,
    Task.status,
    Task.expires_at,
)


class CMCCampaign(Base):
    """Chemical monte carlo results"""

    __tablename__ = "cmc_campaigns"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    target_id: Mapped[int] = mapped_column(
        ForeignKey("targets.id"), nullable=False, index=True
    )

    # could be co_structures, docking_results, or screen_results
    mol_table: Mapped[str] = mapped_column(nullable=True)

    # if we're using a co-structure, this is null
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=True)

    target: Mapped[Target] = relationship("Target")
    model: Mapped[Model] = relationship("Model")

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    exec_sql(
        engine,
        "ALTER DATABASE defaultdb SET idle_in_transaction_session_timeout = '10min'",
        fetch=False,
    )
    return engine


def get_target_id_by_name(engine, name):
    with Session(engine) as session:
        return session.query(Target.id).filter_by(name=name).one().id


def get_alt_struct_id(engine, target_id, alt_struct_name, verbose=True):
    with Session(engine) as sess:
        if alt_struct_name is None:
            alt_struct_id = None
        else:
            alt_struct_id = (
                sess.query(AlternativeStructure.id)
                .filter_by(target_id=target_id, name=alt_struct_name)
                .one()
                .id
            )
            if verbose:
                print("Using alt struct", alt_struct_name)
    return alt_struct_id


def get_dataset_id(engine, target_id, dataset_name):
    with Session(engine) as session:
        dataset_id = (
            session.query(PMFDataset.id)
            .filter_by(target_id=target_id, name=dataset_name)
            .one()
            .id
        )
    return dataset_id


def delete_duplicate_docking_results(engine):
    """Delete all duplicate docking results"""

    query = """
    SELECT id, target_id, mol_id, model_id FROM docking_results
    """
    docking_results = exec_sql(engine, query)
    # find all ids for each (target_id, mol_id, model_id) combo
    docking_results_dict = defaultdict(list)
    for i, row in tqdm(docking_results.iterrows(), total=len(docking_results)):
        docking_results_dict[(row.target_id, row.mol_id, row.model_id)].append(row.id)

    to_delete = []
    for key, val in docking_results_dict.items():
        if len(val) > 1:
            to_delete.extend([int(i) for i in val[1:]])

    with Session(engine) as sess:
        stmt = delete(DockingResult).where(DockingResult.id.in_(to_delete))
        sess.execute(stmt)
        sess.commit()


def get_target_and_dataset_id_from_config():
    """Queries the database to determine target and dataset ids from config options"""
    target_name = CONFIG.target_name

    engine = get_engine()
    with Session(engine) as session:
        target = session.query(Target).filter(Target.name == target_name).one()
        target_id = target.id
        dq = session.query(PMFDataset).filter(PMFDataset.target_id == target_id)
        if "burnin_time" in CONFIG.dataset:
            dq = dq.filter(PMFDataset.burnin_time == CONFIG.dataset.burnin_time)
        if "production_time" in CONFIG.dataset:
            dq = dq.filter(PMFDataset.production_time == CONFIG.dataset.production_time)
        if "name" in CONFIG.dataset:
            dq = dq.filter(PMFDataset.name == CONFIG.dataset.name)

        dataset = dq.one()
        dataset_id = dataset.id

    return target_id, dataset_id


MODEL_IDS = {}


def get_baseline_model_id(engine, model_name):
    """Makes or gets a model for the given baseline name"""
    if model_name in MODEL_IDS:
        return MODEL_IDS[model_name]
    with Session(engine) as session:
        model = session.query(BaselineModel).filter_by(name=model_name).first()
        if model is None:
            model = BaselineModel(name=model_name)
            session.add(model)
            session.commit()
        MODEL_IDS[model_name] = model.id
        return model.id


TARGET_DATA = {}


def get_target_struct_and_pocket(engine, target_id, alt_struct_id=None):
    """Returns the structure and pocket indices for the target. Gives alt struct if requested"""
    key = (target_id, alt_struct_id)
    if key in TARGET_DATA:
        return TARGET_DATA[key]
    with Session(engine) as session:
        target = session.query(Target).get(target_id)
        if alt_struct_id is None:
            struct = target.structure
        else:
            alt_struct = session.query(AlternativeStructure).get(alt_struct_id)
            assert alt_struct.target_id == target_id
            struct = alt_struct.structure
        TARGET_DATA[key] = (struct, target.binding_site)
        return TARGET_DATA[key]


def add_maybe_null_to_query(query, col, val):
    """Adds a column to a (string) query if the value is not None,
    or specifies that the column is NULL"""
    if val is not None:
        q = query + f"\nAND {col} = {val}"
    else:
        q = query + f"\nAND {col} IS NULL"
    return q


def add_protocols():
    """Add all the protocols to the database"""
    engine = get_engine()
    protocols = {
        "mm_5": PMFScoringProtocol(
            minimize_mm=True, mm_restraint_k=5.0, minimize_pmf=False
        ),
        "no_mm": PMFScoringProtocol(
            minimize_mm=False, mm_restraint_k=None, minimize_pmf=False
        ),
    }
    with Session(engine) as sess:
        for protocol_name, p in protocols.items():
            protocol = (
                sess.query(PMFScoringProtocol).filter_by(name=protocol_name).one_or_none()
            )
            if protocol is None:
                protocol = p
                protocol.name = protocol_name
                sess.add(protocol)
                sess.commit()


if __name__ == "__main__":
    init_db()
    add_protocols()
