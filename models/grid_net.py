from datasets.pmf_dataset import PMFData
from models.et import TorchMD_ET
from typing import NamedTuple, Optional, List, Tuple, Dict
from models.tensornet import TensorNet
from torch import nn, Tensor
import torch
# from torch_cubic_spline_grids import CubicBSplineGrid3d
from torch_cubic_spline_grids.interpolate_grids import interpolate_grid_3d
from torch_cubic_spline_grids._constants import CUBIC_B_SPLINE_MATRIX

from common.utils import CONFIG
from models.nn_utils import make_linear, create_grid

class LigGridEncoder(nn.Module):
    """ Uses parameter grid to come up with better nonlinear
    function of the nonbonded parameters """

    def __init__(self):
        super().__init__()

        lig_cfg = CONFIG.model.lig_nb_grid
        self.lig_grid = nn.Parameter(
            create_grid(lig_cfg.spacing, lig_cfg.channels, lig_cfg.max_val * 2)
        )

        self.lig_encoder = make_linear(lig_cfg.channels, CONFIG.model.lig_enc_feats)

        nb_param_radius = torch.tensor(lig_cfg.max_val, dtype=torch.float32)
        self.register_buffer("nb_param_radius", nb_param_radius)
        self.register_buffer("interp_matrix", CUBIC_B_SPLINE_MATRIX)

    def reset_parameters(self):
        self = LigGridEncoder()

    def forward(self, nb_params):

        nb_norm = 0.5 + 0.5 * (nb_params / self.nb_param_radius)
        nb_norm = torch.clamp(nb_norm, 0, 1)

        lig_x = interpolate_grid_3d(self.lig_grid, nb_norm, self.interp_matrix)
        lig_x = self.lig_encoder(lig_x)

        return lig_x


class GridNet(nn.Module):
    """Similar to PMFNet, but uses a grid representation of the receptor.
    Should be a lot faster
    """

    def __init__(self):
        super().__init__()

        self.use_atom_types = CONFIG.model.use_atom_types

        if self.use_atom_types:
            self.n_possible_charges = (
                CONFIG.model.encoder.max_formal_charge
                - CONFIG.model.encoder.min_formal_charge
                + 1
            )
            self.n_possible_elements = CONFIG.model.encoder.max_element + 1
            self.min_formal_charge = CONFIG.model.encoder.min_formal_charge

            self.atom_encoder = nn.Embedding(
                self.n_possible_charges * self.n_possible_elements,
                CONFIG.model.lig_enc_feats,
            )
        else:
            if CONFIG.model.get("use_lig_nb_grid", False):
                self.lig_encoder = LigGridEncoder()
            else:
                self.lig_encoder = make_linear(
                    CONFIG.model.in_feats, CONFIG.model.lig_enc_feats
                )

        self.rec_grids = nn.ParameterList(
            [
                nn.Parameter(create_grid(cfg.spacing, cfg.channels, CONFIG.dataset.pocket.grid_radius * 2))
                for cfg in CONFIG.model.rec_grids
            ]
        )
        self.register_buffer("interp_matrix", CUBIC_B_SPLINE_MATRIX)

        tot_rec_channels = sum(cfg.channels for cfg in CONFIG.model.rec_grids)
        self.rec_encoder = make_linear(tot_rec_channels, CONFIG.model.rec_enc_feats)

        self.rec_both_encoder = make_linear(
            CONFIG.model.rec_enc_feats, CONFIG.model.both_enc_feats
        )
        self.lig_both_encoder = make_linear(
            CONFIG.model.lig_enc_feats, CONFIG.model.both_enc_feats
        )

        self.lig_dropout = nn.Dropout(CONFIG.model.dropout)
        self.rec_dropout = nn.Dropout(CONFIG.model.dropout)

        self.final_encoder = make_linear(
            CONFIG.model.lig_enc_feats
            + CONFIG.model.rec_enc_feats
            + CONFIG.model.both_enc_feats,
            CONFIG.model.hidden_dim,
        )

        self.final_dropout = nn.Dropout(CONFIG.model.dropout)

        match CONFIG.model.nn_type:
            case "et":
                self.net = TorchMD_ET(
                    hidden_channels=CONFIG.model.hidden_dim,
                    num_layers=CONFIG.model.num_layers,
                    cutoff_upper=CONFIG.model.edge_cutoff,
                    max_num_neighbors=int(CONFIG.model.max_neighbors),
                )
            case "tensornet":
                self.net = TensorNet(
                    hidden_channels=CONFIG.model.hidden_dim,
                    num_layers=CONFIG.model.num_layers,
                    cutoff_upper=CONFIG.model.edge_cutoff,
                    max_num_neighbors=int(CONFIG.model.max_neighbors),
                )
            case _:
                raise ValueError(f"Invalid model type {CONFIG.model.nn_type}")

        poc_center = torch.tensor(CONFIG.dataset.pocket.grid_center, dtype=torch.float32)
        self.register_buffer("poc_center", poc_center)

        self.pocket_radius = float(CONFIG.dataset.pocket.grid_radius)

    def reset_parameters(self):
        self = GridNet()

    def forward(
        self,
        data: PMFData,
        pos: Tensor,
        # batch: Optional[Tensor] = None,
        batch: Tensor,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ):
        assert box is None
        assert q is None
        assert s is None
        assert extra_args is None

        pos_norm  = 0.5 + 0.5*((pos - self.poc_center) / self.pocket_radius)
        pos_norm = torch.clamp(pos_norm, 0, 1)

        rec_x = torch.cat([ interpolate_grid_3d(grid, pos_norm, self.interp_matrix) for grid in self.rec_grids], dim=-1)
        rec_x = self.rec_dropout(rec_x)
        rec_x = self.rec_encoder(rec_x)


        if self.use_atom_types:
            assert False
            charge_cat = data.formal_charges - self.min_formal_charge
            elem_cat = data.elements * self.n_possible_charges
            lig_x = self.atom_encoder(charge_cat + elem_cat)
        else:
            lig_x = self.lig_encoder(data.nb_params)
            lig_x = self.lig_dropout(lig_x)

        lig_both = self.lig_both_encoder(lig_x)
        rec_both = self.rec_both_encoder(rec_x)

        both_x = lig_both * rec_both

        x = self.final_encoder(torch.cat([lig_x, rec_x, both_x], dim=-1))
        x = self.final_dropout(x)

        # return x, None, z, pos, batch
        return self.net(x, pos, batch)