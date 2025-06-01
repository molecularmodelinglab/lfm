import numpy as np
from typing import List, Tuple, Optional
from dataclassy import dataclass
from scipy.spatial.transform import Rotation
from nptyping import NDArray, Shape, Float, Int
from openmm import unit
from .alignment import *
from .utils import with_units

@dataclass
class Transformation:

    R: Rotation
    t: NDArray[Shape["3"], Float]

    @staticmethod
    def from_vec(v: NDArray[Shape["6"], Float]) -> "Transformation":
        return Transformation(Rotation.from_rotvec(v[:3]), v[3:])
    
    def to_vec(self) -> NDArray[Shape["6"], Float]:
        return np.concatenate([self.R.as_rotvec(), self.t])

    def __call__(self, x: NDArray[Shape["N,3"], Float]) -> NDArray[Shape["N,3"], Float]:
        """ Can be called with an batch of point clouds as well (shape B,N,3)"""
        c = x.mean(axis=-2, keepdims=True)
        x_centered = x - c
        x_rot = self.R.apply(x_centered.reshape(-1, 3)).reshape(x.shape)
        return (x_rot + c + self.t).astype(np.float32)
    
    @staticmethod
    def from_alignment(source: NDArray[Shape["N,3"], Float],
                       target: NDArray[Shape["N,3"], Float]) -> "Transformation":
        """ Returns the transformation that aligns the source to the target. """

        source_c = source.mean(axis=0)
        target_c = target.mean(axis=0)
        rotmat, _t = find_rigid_alignment(source - source_c, target - target_c)
        # assert np.allclose(_t, 0, atol=1e-3)
        R = Rotation.from_matrix(rotmat)
        t = target_c - source_c
        return Transformation(R, t)
    
    @staticmethod
    def concatenate(args: List["Transformation"]) -> "MultiTransformation":
        """ Concatenate multiple transformations """
        R = Rotation.concatenate([t.R for t in args])
        t = np.stack([t.t for t in args])
        return MultiTransformation(R, t)
    
@dataclass
class MultiTransformation:
    """ Represents a batch of transformations. Each transformation is applied to each input """

    R: Rotation
    t: NDArray[Shape["B,3"], Float]

    @staticmethod
    def from_vec(v: NDArray[Shape["6"], Float]) -> "Transformation":
        return MultiTransformation(Rotation.from_rotvec(v[:,:3]), v[:,3:])
    
    def to_vec(self) -> NDArray[Shape["6"], Float]:
        return np.concatenate([self.R.as_rotvec(), self.t], axis=-1)

    def __getitem__(self, i: int) -> Transformation:
        if isinstance(i, slice):
            return MultiTransformation(self.R[i], self.t[i])
        else:
            assert isinstance(i, int)
            return Transformation(self.R[i], self.t[i])
    
    def __len__(self) -> int:
        return len(self.t)
    
    def __add__(self, other: "MultiTransformation") -> "MultiTransformation":
        return MultiTransformation.concatenate([self, other])

    @staticmethod
    def concatenate(args: List["MultiTransformation"]) -> "MultiTransformation":
        """ Concatenate multiple transformations """
        R = Rotation.concatenate([t.R for t in args])
        t = np.concatenate([t.t for t in args])
        return MultiTransformation(R, t)

    def apply_single(self, x: NDArray[Shape["N,3"], Float]) -> NDArray[Shape["B,N,3"], Float]:
        """ Apply to a single point cloud """
        R = self.R.as_matrix().astype(np.float32)
        t = self.t.astype(np.float32)
        c = x.mean(axis=-2, keepdims=True)
        x_centered = x - c
        return R.dot(x_centered.T).transpose((0,2,1)) + t[:,None] + c

    def apply_batch(self, x: NDArray[Shape["B1,N,3"], Float]) -> NDArray[Shape["B1,B,N,3"], Float]:
        """ Apply to a batch of point clouds. Currently not vectorized """
        ret = []
        for x in x:
            ret.append(self.apply_single(x))
        return np.stack(ret)

    def __call__(self, x: NDArray[Shape["B,N,3"], Float]) -> NDArray[Shape["B,N,3"], Float]:
        """ Can be called with a single point cloud or a batch """
        if len(x.shape) == 2:
            return self.apply_single(x)
        else:
            return self.apply_batch(x)
