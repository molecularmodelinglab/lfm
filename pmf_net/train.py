from io import BytesIO, TextIOWrapper
import os
from traceback import print_exc

import numpy as np
from common.docking_metrics import run_model_docking_metrics
from common.task_queue import task
from common.wandb_utils import get_old_checkpoint, get_old_model, get_old_pl_model
from sqlalchemy.orm import Session
from omegaconf import OmegaConf
from openmm import unit
from datasets.pmf_dataset import beta

from common.db import PMFDatagen, Target, get_engine, PMFDataset as DBPMFDataset, get_target_and_dataset_id_from_config
from common.scoring_metrics import run_model_scoring_metrics
from datagen.pmf_datagen import initialize_datagen
from common.utils import CONFIG, get_output_dir, get_residue_atom_indices, load_config
from scipy.spatial.transform import Rotation
from torch.autograd.functional import vhp

import sys
from typing import Optional
import warnings

from models.grid_net import GridNet

# so many torchmd warnings smh
warnings.filterwarnings("ignore")

from datasets.pmf_dataset import PMFData, PMFDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
from torchmetrics.regression import R2Score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmdnet.models.output_modules import Scalar, OutputModel
from torch_geometric.loader import DataLoader
from models.pmf_net import PMFNet
from models.torch_md_net import TorchMD_Net
from common.metrics import (
    calc_mean_force,
    calc_mean_torque,
    get_dataset_metrics,
    get_metrics,
)

def resample_positions(batch):
    """ uniformly sample translations accross the grid and rotations are random """

    grid_center = torch.tensor(
        CONFIG.dataset.pocket.center, dtype=batch.pos.dtype, device=batch.pos.device
    )
    grid_radius = torch.tensor(
        CONFIG.dataset.pocket.grid_radius, dtype=batch.pos.dtype, device=batch.pos.device
    )

    grid_min = grid_center - grid_radius
    grid_max = grid_center + grid_radius

    new_pos_batch = torch.zeros_like(batch.pos)
    for b in range(len(batch)):
        mask = batch.batch == b
        pos = batch.pos[mask]

        pos_centered = pos - pos.mean(0)

        R = Rotation.random().as_matrix()
        R = torch.asarray(R, dtype=pos.dtype, device=pos.device)
        pos_rot = torch.matmul(pos_centered, R)

        t = torch.rand(3, device=pos.device, dtype=pos.dtype)
        trans = grid_min + t * (grid_max - grid_min)

        new_pos = pos_rot + trans
        new_pos_batch[mask] = new_pos

    return new_pos_batch

class ForceMatcher(L.LightningModule):

    # target_id = None for backwards compatibility
    def __init__(self, dataset_id, target_id=None, wandb_resume=None):
        super().__init__()

        rep_model = {
            "grid": GridNet,
        }[CONFIG.model.type]()

        out_model = Scalar(CONFIG.model.hidden_dim)

        self.epoch_offset = CONFIG.train.get("epoch_offset", 0)

        if wandb_resume is None:
            model = TorchMD_Net(
                representation_model=rep_model,
                output_model=out_model,
                derivative=True,
            )
            if CONFIG.train.jit:
                self.model = torch.jit.script(model)
            else:
                self.model = model
        else:
            self.model = get_old_model(wandb_resume)
            CONFIG.train["resumed_from"] = wandb_resume

        self.hparams.update(CONFIG)
        self.save_hyperparameters()

        self.target_id = target_id
        self.dataset_id = dataset_id
        
        dataset_metrics = get_dataset_metrics(dataset_id)
        self.train_metrics = get_metrics("train", dataset_metrics)
        self.val_metrics = get_metrics("val", dataset_metrics)

        if CONFIG.get("debug_docking", False):
            run_model_docking_metrics(
                self.model,
                self.target_id,
                CONFIG.eval.docking.baseline,
                None
            )
        if CONFIG.get("debug_scoring", False):
            run_model_scoring_metrics(
                self.model,
                self.target_id,
                CONFIG.eval.scoring.baseline,
                None,
                CONFIG.eval.scoring.decoy_steps,
            )

    def forward(self, data):
        # force loss
        x = PMFData.from_pyg_data(data)
        out1 = self.model(x, data.pos, data.batch)

        # contrastive loss
        pos_rand = resample_positions(data)
        deriv = self.model.derivative
        self.model.derivative = False
        out2 = self.model(x, pos_rand, data.batch)

        # hessian loss
        # todo: this is inefficient, we shouldn't need to call
        # the model again

        if CONFIG.loss.hessian_weight > 0:
            def f(pos):
                # betaW
                return beta*self.model(x, pos.reshape((-1, 3)), data.batch)[0].sum()
            _, pred_vhp = vhp(f, data.pos.reshape(-1), data.rand_vec, create_graph=True)
        else:
            pred_vhp = None

        self.model.derivative = deriv


        return out1, out2, pred_vhp


    def training_step(self, batch, batch_idx):

        loss = self.shared_step("train", batch)

        metrics = self.train_metrics.compute()
        self.log_dict(
            metrics, on_step=True, on_epoch=False, batch_size=CONFIG.train.batch_size
        )
        self.train_metrics.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        return self.shared_step("val", batch)

    # todo: re-add eval later
    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(
            metrics, on_step=False, on_epoch=True, batch_size=CONFIG.train.batch_size
        )
        self.val_metrics.reset()

        cur_epoch = self.epoch_offset + self.current_epoch

        try:
            if CONFIG.eval.scoring.freq is not None and (cur_epoch + 1) % CONFIG.eval.scoring.freq == 0:
                self.log_dict(
                    run_model_scoring_metrics(
                        self.model,
                        self.target_id,
                        CONFIG.eval.scoring.baseline,
                        CONFIG.eval.scoring.decoy_steps,
                    )
                )
        except:
            print_exc()
            

        try:
            if CONFIG.eval.docking.freq is not None and (cur_epoch + 1) % CONFIG.eval.docking.freq == 0:
                self.log_dict(
                    run_model_docking_metrics(
                        self.model,
                        self.target_id,
                        CONFIG.eval.docking.baseline,
                    )
                )
        except:
            print_exc()

    def shared_step(self, stage, batch):

        with torch.enable_grad():
            (U, yp), (U_rand, _), pred_vhp = self(batch)

        loss, loss_dict = self.get_losses(yp, U, U_rand, pred_vhp, batch)

        self.log(
            f"{stage}_loss",
            loss,
            batch_size=CONFIG.train.batch_size,
        )
        for key, value in loss_dict.items():
            self.log(
                f"{stage}_{key}",
                value,
                batch_size=CONFIG.train.batch_size,
            )

        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(yp, batch)

        return loss

    def configure_optimizers(self):

        # scale the GB params maybe
        gb_params = []
        other_params = []
        for key, val in self.named_parameters():
            if "gb_gnn" in key:
                gb_params.append(val)
            else:
                other_params.append(val)
        gb_lr = CONFIG.train.lr * CONFIG.train.gb_lr_scale
        param_groups = [
            {"params": gb_params, "lr": gb_lr},
            {"params": other_params},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=CONFIG.train.lr)
        return optimizer

    def get_losses(self, yp, U, U_rand, pred_vhp, batch):

        yt = batch.forces

        yp_mean = calc_mean_force(yp, batch.batch)
        yt_mean = calc_mean_force(batch.forces, batch.batch)

        yp_torque = calc_mean_torque(batch.pos, yp, batch.batch)
        yt_torque = calc_mean_torque(batch.pos, batch.forces, batch.batch)

        force_mse = F.mse_loss(yp, yt)
        mean_force_mse = F.mse_loss(yp_mean, yt_mean)
        mean_torque_mse = F.mse_loss(yp_torque, yt_torque)

        # we want U to be lower than U_rand, up to a point
        naive_contrastive_loss = U - U_rand
        naive_contrastive_loss = torch.clamp(naive_contrastive_loss, CONFIG.loss.naive_contrastive_min, None)
        naive_contrastive_loss = naive_contrastive_loss.mean()

        loss = (
            CONFIG.loss.force_weight * force_mse
            + CONFIG.loss.mean_force_weight * mean_force_mse
            + CONFIG.loss.mean_torque_weight * mean_torque_mse
            + CONFIG.loss.naive_contrastive_weight * naive_contrastive_loss
        )

        loss_dict = {
            "force_mse": force_mse,
            "mean_force_mse": mean_force_mse,
            "mean_torque_mse": mean_torque_mse,
            "naive_contrastive": naive_contrastive_loss
        }

        if CONFIG.loss.hessian_weight > 0:
            vhp_true = beta*batch.mean_hessian_vp - beta*beta*batch.force_cov_vp
            hessian_loss = F.mse_loss(pred_vhp, vhp_true)
            loss += CONFIG.loss.hessian_weight * hessian_loss
            loss_dict["hessian_mse"] = hessian_loss

        return loss, loss_dict


def add_pocket_info_to_config(dataset_id):
    """Adds CONFIG.pocket.radius and CONFIG.pocket.center to the config
    based on the current target. This is a pretty hacky way to store this info
    -- come up with a better way to do this later"""
    engine = get_engine()

    # if we've manually overridden the pocket info, don't do anything
    if "pocket" in CONFIG.dataset:
        for attr in ["center", "radius", "exit_point"]:
            if attr not in CONFIG.dataset.pocket:
                raise ValueError(
                    f"Pocket info already in config, but {attr} was not found."
                )
        print("Pocket info already in config, skipping")
        return

    with Session(engine) as sess:
        dataset = sess.query(DBPMFDataset).get(dataset_id)
        target = dataset.target
        rec_structure = target.structure
        poc_residues = target.binding_site

        if dataset.datagen is None:
            # just get the first exit point for this target
            datagen = (
                sess.query(PMFDatagen).filter(PMFDatagen.target_id == target.id)
            ).first()
            exit_point = datagen.exit_point * 10  # nm -> angstroms
        else:
            exit_point = dataset.datagen.exit_point * 10 

        poc_indices = get_residue_atom_indices(rec_structure.topology, poc_residues)
        all_poc_pos = rec_structure.positions[poc_indices].value_in_unit(unit.angstroms)
        # add exit point to all_poc_pos
        all_poc_pos = np.concatenate([all_poc_pos, [exit_point]])
        box_center = 0.5 * (all_poc_pos.max(axis=0) + all_poc_pos.min(axis=0))
        box_size = all_poc_pos.max(axis=0) - all_poc_pos.min(axis=0)

        poc_center = np.mean(all_poc_pos, axis=0)

        CONFIG.dataset["pocket"] = {
            "center": poc_center.tolist(),
            "grid_center": box_center.tolist(),
            "grid_radius": (float(box_size.max()) / 2) + 1,
            "exit_point": (exit_point).tolist(),  # nm -> angstroms,
        }


def train_immediate(target_id, dataset_id):

    add_pocket_info_to_config(dataset_id)

    checkpoint_callback = ModelCheckpoint(monitor=CONFIG.train.monitor.metric, mode=CONFIG.train.monitor.mode, save_top_k=CONFIG.train.monitor.save_top_k)

    # define epochs before reloading CONFIG
    run_name = CONFIG.train.run_name
    max_epochs = CONFIG.train.epochs
    wandb_resume = CONFIG.train.get("wandb_resume", None)
    checkpoint_path = None
    if wandb_resume is not None:
        print("Resuming from wandb run", wandb_resume)
        checkpoint_path = get_old_checkpoint(wandb_resume)

    train_dataset = PMFDataset(dataset_id, "train")
    val_dataset = PMFDataset(dataset_id, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.train.batch_size,
        shuffle=True,
        num_workers=CONFIG.dataloader_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.train.batch_size,
        shuffle=False,
        num_workers=CONFIG.dataloader_workers,
    )

    if CONFIG.train.log_wandb:
        wandb.finish()
        logger = WandbLogger(
            project=CONFIG.wandb.project, log_model="all", name=run_name
        )
    else:
        logger = None

    # if wandb_resume is None:
    model = ForceMatcher(dataset_id, target_id)
    # else:
    #     print("Resuming from wandb run", wandb_resume)
    #     model = get_old_pl_model(wandb_resume)

    assert model.target_id is not None

    trainer = L.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        val_check_interval=CONFIG.train.val_check_interval,
        check_val_every_n_epoch=CONFIG.train.check_val_every_n_epoch,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)


# @celery_app.task(
#     worker_prefetch_multiplier=1,
#     name="train_task",
#     max_retries=0,
# )
# we never want to retry these tasks
@task(
    max_runtime_hours=None,
    retry_failed=False,
    prefetch=1,
)
def train(target_id, dataset_id, cfg_str):
    """Train with the yaml-encoded config string"""
    cfg_fname = f"{get_output_dir()}/config.yaml"
    with open(cfg_fname, "w") as f:
        f.write(cfg_str)
    load_config(cfg_fname, False)
    train_immediate(target_id, dataset_id)

if __name__ == "__main__":

    action = sys.argv[1]

    # this is the only way to send command line arguments to runpod
    if "CMD_ARGS" in os.environ:
        args = os.environ["CMD_ARGS"]
        print("Loading additional command line arguments:", args)
        sys.argv += args.split()

        load_config(None, True)

    cfg_fname = None
    if "cfg" in CONFIG:
        print("Loading config from", CONFIG.cfg)
        cfg_fname = CONFIG.cfg

    load_config(cfg_fname, True)

    # queue up the training task
    target_name = CONFIG.target_name

    engine = get_engine()
    # datagen_id = initialize_datagen(target_name)
    with Session(engine) as session:
        target = session.query(Target).filter(Target.name == target_name).one()
        target_id = target.id
        dq = session.query(DBPMFDataset).filter(DBPMFDataset.target_id == target_id)
        if "burnin_time" in CONFIG.dataset:
            dq = dq.filter(DBPMFDataset.burnin_time == CONFIG.dataset.burnin_time)
        if "production_time" in CONFIG.dataset:
            dq = dq.filter(
                DBPMFDataset.production_time == CONFIG.dataset.production_time
            )
        if "name" in CONFIG.dataset:
            dq = dq.filter(DBPMFDataset.name == CONFIG.dataset.name)

        dataset = dq.one()
        dataset_id = dataset.id

    buffer = TextIOWrapper(BytesIO())
    OmegaConf.save(CONFIG, buffer)
    buffer.seek(0)
    cfg_text = buffer.read()

    target_id, dataset_id = get_target_and_dataset_id_from_config()

    if action == "queue":
        train.delay(target_id, dataset_id, cfg_text)
    elif action == "run":
        train(target_id, dataset_id, cfg_text)
    else:
        raise ValueError(f"Unknown action {action}")
