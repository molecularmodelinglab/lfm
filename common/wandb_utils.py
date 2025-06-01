
from common.db import PMFModel, get_target_and_dataset_id_from_config
from common.utils import CONFIG, get_output_dir
from sqlalchemy.orm import Session
from omegaconf import OmegaConf
import wandb
import os
import ast

os.environ["WANDB_API_KEY"] = open("./secrets/wandb_key.txt").read()
os.environ["WANDB_DATA_DIR"] = f"{get_output_dir()}/wandb_data"
os.environ["WANDB_ARTIFACT_LOCATION"] = f"{get_output_dir()}/wandb_artifact_location"
os.environ["WANDB_ARTIFACT_DIR"] = f"{get_output_dir()}/wandb_artifact"
os.environ["WANDB_CACHE_DIR"] = f"{get_output_dir()}/wandb_cache"
os.environ["WANDB_CONFIG_DIR"] = f"{get_output_dir()}/wandb_config"

for dir in [
    os.environ["WANDB_DATA_DIR"],
    os.environ["WANDB_ARTIFACT_LOCATION"],
    os.environ["WANDB_ARTIFACT_DIR"],
    os.environ["WANDB_CACHE_DIR"],
    os.environ["WANDB_CONFIG_DIR"],
]:
    os.makedirs(dir, exist_ok=True)

_api = wandb.Api()

def get_weight_artifact(run, tag):
    return _api.artifact(f"{run.project}/model-{run.id}:{tag}", type='model')

def get_old_run(run_name, tag="best"):
    runs = _api.runs(path=CONFIG.wandb.project, filters={"display_name": run_name})
    assert len(runs) == 1
    return runs[0]

def load_old_run_config(run):
    """ Load the config of a run. The way we log hyperparams is pretty sus smh """
    for key, val in run.config.items():
        if key == "storage": continue
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        CONFIG[key] = val

def get_old_checkpoint(run_name, run_id=None, tag="best"):
    """ Get the wandb run with name run_name, and return a model
    loaded from the saved checkpoint with the correct tag.
    WARNING: This will overwrite the config in CONFIG """

    if run_id is not None:
        run = _api.run(f"{CONFIG.wandb.project}/{run_id}")
    else:
        run = get_old_run(run_name, tag)
    load_old_run_config(run)

    artifact = get_weight_artifact(run, tag)
    artifact_dir = f"{os.environ['WANDB_ARTIFACT_DIR']}/model-{run.id}:{artifact.version}"
    if not os.path.exists(artifact_dir):
        artifact.download()
    checkpoint_file = artifact_dir + "/model.ckpt"

    return checkpoint_file

def get_old_pl_model(run_name, run_id=None, tag="best"):
    from pmf_net.train import ForceMatcher
    checkpoint_file = get_old_checkpoint(run_name, run_id, tag)
    pl_model = ForceMatcher.load_from_checkpoint(checkpoint_file)

    return pl_model

def get_old_model(run_name, run_id=None, tag="best"):
    return get_old_pl_model(run_name, run_id, tag).model

DB_MODELS = {}
def get_wandb_model(engine, model_name, target_id):
    """ Maybe add a new model to the db, returns model id """
    key = (model_name, target_id)
    if key in DB_MODELS:
        return DB_MODELS[key]

    run = get_old_run(model_name)
    load_old_run_config(run)
    # we actually store the dataset id in the run config but wandb 
    # is rounding the number... very weird bug lol
    _target_id, dataset_id = get_target_and_dataset_id_from_config()
    assert target_id == _target_id, f"Target id {target_id} does not match config target id {_target_id}"
    with Session(engine) as session:
        # first check if the model is already in the db
        model_db = session.query(PMFModel).filter(PMFModel.wandb_id == run.id).first()
        if model_db is None:
            model_db = PMFModel(
                config=OmegaConf.to_container(CONFIG),
                wandb_id=run.id,
                target_id=target_id,
                dataset_id=dataset_id
            )
            session.add(model_db)
            session.commit()
        DB_MODELS[key] = model_db.id
        return model_db.id