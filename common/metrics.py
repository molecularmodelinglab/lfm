import os

import numpy as np
from common.db import get_db_gs_path
from common.gs import GS_FS
from datagen.pmf_datagen import get_gs_fs
import torch
from torch import nn
from torchmetrics import Metric, R2Score, MetricCollection, MeanSquaredError
from torchmdnet.models.utils import scatter
from common.utils import get_cache_dir

def calc_mean_torque(positions, forces, batch):
    """Calculates the mean torque on the system around the mean position
    (not the center of mass)"""
    mean_pos = scatter(positions, batch, reduce="mean")
    mean_pos = mean_pos[batch]
    torques = torch.linalg.cross(positions - mean_pos, forces)
    return scatter(torques, batch, dim=0, reduce="mean")


def calc_mean_force(forces, batch):
    """Calculates the mean force on the system"""
    return scatter(forces, batch, dim=0, reduce="mean")

def calc_dataset_metrics(dataset_id):
    """Saves the mean and std of forces, mean force, and mean torque for the dataset"""
    from datasets.pmf_dataset import PMFDataset
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm
    import numpy as np

    all_forces = []
    all_mean_forces = []
    all_torques = []

    dataset = PMFDataset(dataset_id, "all", use_metrics=False)
    loader = DataLoader(dataset, batch_size=1)
    for i, batch in enumerate(tqdm(loader)):

        torques = calc_mean_torque(batch.pos, batch.forces, batch.batch)
        mean_forces = calc_mean_force(batch.forces, batch.batch)

        all_forces.append(batch.forces)
        all_mean_forces.append(mean_forces)
        all_torques.append(torques)

    all_forces = torch.cat(all_forces, dim=0)
    all_mean_forces = torch.stack(all_mean_forces, dim=0)
    all_torques = torch.stack(all_torques, dim=0)

    results = {}

    for name, tensor in zip(
        ["forces", "mean_forces", "torques"], [all_forces, all_mean_forces, all_torques]
    ):
        results[name + "_mean"] = tensor.mean(dim=0).cpu().numpy()
        results[name + "_std"] = tensor.std(dim=0).cpu().numpy()

    return results

def get_dataset_metrics(dataset_id):
    """ Downloads the dataset metrics for the PMF dataset """
    gs_folder = get_db_gs_path("pmf_datasets", dataset_id)
    metrics_path = os.path.join(gs_folder, "metrics.npz")
    return dict(np.load(get_gs_fs().open(metrics_path, "rb")))

class ForceR2(Metric):
    """(mean) R2 for force predictions. Uses the cached mean and std 
    of the dataset rather than computing these metrics on the fly """

    def __init__(self, dataset_metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse = MeanSquaredError(num_outputs=3)
        mean = torch.tensor(dataset_metrics["forces_mean"], dtype=torch.float32)
        std = torch.tensor(dataset_metrics["forces_std"], dtype=torch.float32)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def update(self, yp, batch):
        self.mse.update(yp, batch.forces)

    def compute(self):
        mse = self.mse.compute()
        return (1 - mse / self.std.pow(2)).mean()

    def reset(self):
        self.mse.reset()


class MeanForceR2(Metric):
    """ R2 for mean force predictions. Uses the cached mean and std """

    def __init__(self, dataset_metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse = MeanSquaredError(num_outputs=3)
        mean = torch.tensor(dataset_metrics["mean_forces_mean"], dtype=torch.float32)
        std = torch.tensor(dataset_metrics["mean_forces_std"], dtype=torch.float32)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def update(self, yp, batch):
        yp = calc_mean_force(yp, batch.batch)
        yt = calc_mean_force(batch.forces, batch.batch)
        self.mse.update(yp, yt)

    def compute(self):
        mse = self.mse.compute()
        return (1 - mse / self.std.pow(2)).mean()
    
    def reset(self):
        self.mse.reset()

class TorqueR2(Metric):
    """ R2 for mean torque predictions. Uses the cached mean and std """

    def __init__(self, dataset_metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse = MeanSquaredError(num_outputs=3)
        mean = torch.tensor(dataset_metrics["torques_mean"], dtype=torch.float32)
        std = torch.tensor(dataset_metrics["torques_std"], dtype=torch.float32)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def update(self, yp, batch):
        yp = calc_mean_torque(batch.pos, yp, batch.batch)
        yt = calc_mean_torque(batch.pos, batch.forces, batch.batch)
        self.mse.update(yp, yt)

    def compute(self):
        mse = self.mse.compute()
        return (1 - mse / self.std.pow(2)).mean()
    
    def reset(self):
        self.mse.reset()

def get_metrics(stage, dataset_metrics):
    """A moduledict of all the metrics we want to track"""

    return MetricCollection(
        {
            "force_r2": ForceR2(dataset_metrics),
            "mean_force_r2": MeanForceR2(dataset_metrics),
            "mean_torque_r2": TorqueR2(dataset_metrics),
        },
        prefix=f"{stage}_",
    )
