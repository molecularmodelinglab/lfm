from typing import Optional
from common.utils import add_coords_to_mol
import roma
import torch
from functools import reduce
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch_geometric.data import Data
import spyrmsd
from spyrmsd.rmsd import symmrmsd, rmsd
from torch_geometric.utils import to_dense_adj

from terrace.batch import Batch, Batchable, collate


class Pose(Batchable):
    coord: torch.Tensor

    @staticmethod
    def collate_coord(x):
        return x


class MultiPose(Batchable):
    coord: torch.Tensor

    @staticmethod
    def collate_coord(x):
        return x

    @staticmethod
    def combine(poses):
        return MultiPose(torch.stack([p.coord for p in poses]))

    def get(self, i):
        return Pose(coord=self.coord[i])

    def num_poses(self):
        return len(self.coord)

    def items(self):
        for coord in self.coord:
            yield Pose(coord)

    def batch_get(self, i):
        return collate([mp.get(i) for mp in self])


def add_pose_to_mol(mol, pose):
    add_coords_to_mol(mol, pose.coord.detach().cpu())


def add_multi_pose_to_mol(mol, mp):
    mol.RemoveAllConformers()
    for i in range(mp.num_poses()):
        pose = mp.get(i)
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for i, coord in enumerate(pose.coord.detach().cpu()):
            conformer.SetAtomPosition(
                i, Point3D(float(coord[0]), float(coord[1]), float(coord[2]))
            )

        mol.AddConformer(conformer, True)


# tried to make this class less hacky but only partially succeeded
# a single transform actually can emcompass a batch of transforms
# adding batches of batches to terrace will be complex tho, so rn
# we shall stick with this sadness
class PoseTransform(Batchable):

    # todo: torsional angles!
    rot: torch.Tensor
    trans: torch.Tensor
    tor_angles: torch.Tensor

    @staticmethod
    def collate_tor_angles(x):
        return x

    @staticmethod
    def to_raw(t):
        buf = []
        buf.append(t.trans.reshape(-1))
        buf.append(t.rot.reshape(-1))
        for angle in t.tor_angles:
            buf.append(angle.reshape(-1))
        return torch.cat(buf, 0)

    @staticmethod
    def from_raw(raw, template=None):
        idx = 0

        def eat_arr(temp_arr):
            nonlocal idx
            shape = temp_arr.shape
            size = reduce(lambda a, b: a * b, shape)
            ret = raw[idx : idx + size].reshape(shape)
            idx += size
            return ret

        if template is None:
            trans = raw[:3]
            rot = raw[3:6]
            angles = raw[6:]
        else:
            trans = eat_arr(template.trans)
            rot = eat_arr(template.rot)
            angles = []
            for angle in template.tor_angles:
                angles.append(eat_arr(angle))
            if not isinstance(template, Batch):
                angles = torch.stack(angles)
        if isinstance(template, Batch):
            return Batch(PoseTransform, trans=trans, rot=rot, tor_angles=angles)
        else:
            return PoseTransform(trans=trans, rot=rot, tor_angles=angles)

    @staticmethod
    def identity(td):
        """Return an identity transform given torsion data"""

        rot = torch.tensor([0.0, 0.0, 2.0 * torch.pi])
        trans = torch.zeros(3)
        tor_angles = torch.zeros(len(td.rot_edges))

        return PoseTransform(rot, trans, tor_angles)

    def to(self, device):
        return PoseTransform(
            self.rot.to(device), self.trans.to(device), self.tor_angles.to(device)
        )

    # @torch.compile(dynamic=True, fullgraph=True)
    def apply(self, pose, tor_data, use_tor=True):
        coord = pose.coord
        rot = self.rot + 1e-10  # add epsilon to avoid NaN gradients
        rot_mat = roma.rotvec_to_rotmat(rot)
        if self.tor_angles.size(-1) > 0 and use_tor:
            coord = tor_data.set_all_angles(self.tor_angles, coord)
        centroid = coord.mean(-2, keepdim=True)

        if len(coord.shape) == 3 or len(rot_mat.shape) == 3:
            trans = self.trans.unsqueeze(1)
        else:
            trans = self.trans
        coord = (
            torch.einsum("...ij,...bj->...bi", rot_mat, coord - centroid)
            + trans
            + centroid
        )
        if coord.dim() == 3:
            return MultiPose(coord=coord)
        return Pose(coord=coord)

    def batch_apply(self, lig_poses, lig_tor_data):
        return collate(
            [
                PoseTransform.apply(t, c, d)
                for t, c, d in zip(self, lig_poses, lig_tor_data)
            ]
        )

    def grad(self, U):
        rot_grad, trans_grad = torch.autograd.grad(
            U, [self.rot, self.trans], create_graph=True
        )
        return PoseTransform(rot=rot_grad, trans=trans_grad)

    def batch_grad(self, U):
        # todo: why allow unused?
        rot_grad, trans_grad, *tor_grad = torch.autograd.grad(
            U,
            [self.rot, self.trans, *self.tor_angles],
            create_graph=True,
            allow_unused=True,
        )
        return Batch(PoseTransform, rot=rot_grad, trans=trans_grad, tor_angles=tor_grad)

    def batch_requires_grad(self):
        self.rot.requires_grad_()
        self.trans.requires_grad_()
        for angle in self.tor_angles:
            angle.requires_grad_()

    def requires_grad(self):
        self.rot.requires_grad_()
        self.trans.requires_grad_()
        self.tor_angles.requires_grad_()

    def batch_update_from_grad(self, grad):
        mul = 1.0
        rot = self.rot - mul * grad.rot
        trans = self.trans - mul * grad.trans
        tor = []
        for angle, grad_angle in zip(self.tor_angles, grad.tor_angles):
            if grad_angle is None:
                tor.append(angle)
            else:
                tor.append(angle - mul * grad_angle)
        # print(grad.rot, grad.tor_angles)
        return Batch(PoseTransform, rot=rot, trans=trans, tor_angles=tor)


def sym_rmsd(pose1: Pose, pose2: Pose, data1: Data, data2: Optional[Data] = None):

    if data2 is None:
        data2 = data1

    adj1 = to_dense_adj(data1.edge_index)[0].cpu().numpy().astype(int)
    aprops1 = data1.elements.cpu().numpy()

    adj2 = to_dense_adj(data2.edge_index)[0].cpu().numpy().astype(int)
    aprops2 = data2.elements.cpu().numpy()

    coords1 = pose1.coord.cpu().numpy()
    coords2 = pose2.coord.cpu().numpy()

    # we only want non-H rmsd; else isomorphisms take too long
    mask1 = aprops1 != 1
    aprops1 = aprops1[mask1]
    coords1 = coords1[mask1]
    adj1 = adj1[mask1][:, mask1]

    mask2 = aprops2 != 1
    aprops2 = aprops2[mask2]
    coords2 = coords2[mask2]
    adj2 = adj2[mask2][:, mask2]

    return symmrmsd(
        coords1,
        coords2,
        aprops1,
        aprops2,
        adj1,
        adj2,
    )
