import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from dataclassy import dataclass
from .utils import *
from .alignment import *
from .transformation import *
from openmm import app
from openmm import unit
from openmm.unit import Quantity
import openmm as mm
from pymbar import MBAR
from scipy import stats
from nptyping import NDArray, Shape, Float, Int
from sklearn.cluster import MeanShift, estimate_bandwidth
from .bingham import BinghamDistribution

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

def norm_logpdf_unnorm(sigma, x):
    """Returns the unnormalized log probability of a normal distribution"""
    return -((x**2) / (2 * sigma**2))


def multivariate_logpdf_unnorm(dist, x):
    """Returns the unnormalized log probability of a multivariate normal distribution.
    Modified from https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_multivariate.py#L289-L851
    """

    dev = x - dist.mean
    maha = np.sum(np.square(dist.cov_object.whiten(dev)), axis=-1)
    return -0.5 * maha


@dataclass(frozen=True)
class TransformationalRestraint:
    """Base class for transformational restraints -- this should make it easy
    to define new distributions for translations and rotations of the ligand
    w/r/t a reference conformation."""

    reference: NDArray[Shape["S,3"], Float]  # reference positions
    kT: unit.Quantity  # temperature times Boltzmann constant

    @property
    def lig_indices(self):
        return self.topography.lig_indices

    @property
    def rec_indices(self):
        return self.topography.rec_indices

    @property
    def poc_alpha_indices(self):
        return self.topography.poc_alpha_indices

    @with_units({1: unit.nanometer}, unit.nanometer)
    def align_lig_frames(
        self, lig_frames: NDArray[Shape["F,L,3"], Float]  # ligand frames
    ) -> NDArray[Shape["F,L,3"], Float]:
        """Aligns the ligand frames to the reference frame"""
        reference = self.reference / unit.nanometer
        return rigid_align_batched(
            lig_frames, reference
        )
    
    @with_units({1: unit.nanometer}, unit.nanometer)
    def align_lig_pos(
        self, lig_positions: NDArray[Shape["S,3"], Float]  # ligand positions
    ) -> NDArray[Shape["S,3"], Float]:
        """Aligns the ligand positions to the reference frame"""
        reference = self.reference / unit.nanometer
        return rigid_align(lig_positions, reference)

    @with_units({1: unit.nanometer}, unit.nanometer)
    def sample_lig_frames(
        self,
        frames: NDArray[Shape["F,L,3"], Float],  # ligand frames (MUST BE ALIGNED)
        num_transformations,  # number of transformations to generate
    ) -> NDArray[Shape["F,N,L,3"], Float]:
        """Samples random transformations of the ligand for each (aligned) frame"""

        # want to be able to handle single frames too
        single_frame = False
        if len(frames.shape) == 2:
            frames = frames[None]
            single_frame = True

        mt = self.sample_transformation_batch(num_transformations)
        new_frames = mt(frames)

        if single_frame:
            new_frames = new_frames[0]

        return new_frames

    def sample_transformation(self) -> Transformation:
        """Samples a random transformation  (rotation and translation)
        from the Boltzmann distribution of this restraint"""
        raise NotImplementedError

    def sample_transformation_batch(self, n: int) -> MultiTransformation:
        """Samples a batch of random transformations  (rotation and translation)
        from the Boltzmann distribution of this restraint. Override this method
        if you want to make this more efficient."""
        return Transformation.concatenate(
            [self.sample_transformation() for _ in range(n)]
        )

    def logprob_unnorm(
        self,
        T: Transformation,
    ) -> Float:
        """Returns the unnormalized log probability of a given transformation
        under this restraint -- that is, the energy of the bolztmann distribution
        in units of kT. You cannot just use dist.logpdf because the normalization
        constant must not be included here"""
        raise NotImplementedError
    
    def logprob(self, T: Transformation) -> Float:
        """Returns the log probability of a given transformation under this restraint"""
        raise NotImplementedError

    def logprob_unnorm_batch(self, T: Transformation) -> NDArray[Shape["N"], Float]:
        """Returns the log probability of a batch of transformations under this restraint"""
        return np.array([self.logprob_unnorm(t) for t in T])

    def get_standard_state_dF_MBAR(self, N: int = 1000000) -> unit.Quantity:
        """Free energy of transforming the ligand from uniform box
        at concentration 1M to the energy well of this restraint."""

        # first sample from standard state
        V = 1 * unit.liter / (unit.mole * unit.AVOGADRO_CONSTANT_NA)
        L = (V ** (1 / 3)).value_in_unit(unit.nanometer)

        R0 = Rotation.random(N)
        t0 = np.random.uniform(-L / 2, L / 2, (N, 3))
        T0 = MultiTransformation(R0, t0)

        T1 = self.sample_transformation_batch(N)

        u0 = np.zeros(2 * N)  # constant potential
        u1 = -self.logprob_unnorm_batch(T0 + T1)  # potential of the restraint

        u_nk = np.stack([u0, u1])
        N_k = np.array([N, N])

        mbar = MBAR(u_nk, N_k)
        results = mbar.compute_free_energy_differences()
        if results["dDelta_f"].max() > 0.2:
            print(
                f"Warning: uncertainty in standard state free energy is large {results['dDelta_f'].max()}"
            )

        return results["Delta_f"][0, 1] * self.kT

    def get_standard_state_dF(self, N: int = 1000000) -> unit.Quantity:
        """Free energy of transforming the ligand from uniform box
        at concentration 1M to the energy well of this restraint. This
        is computed analytically for everything now"""

        V_t = 1 * unit.liter / (unit.mole * unit.AVOGADRO_CONSTANT_NA)
        V_t = V_t.value_in_unit(unit.nanometer**3)
        # surface volume of 3-sphere, from which we can uniformly sample quaternions
        V_R = 2*(np.pi**2)
        F_ss = -self.kT*np.log(V_t) - self.kT*np.log(V_R)

        T = self.sample_transformation()
        logZ = self.logprob_unnorm(T) - self.logprob(T)
        F = -self.kT*logZ

        return (F - F_ss)

    def get_lig_transformation(
        self, positions: NDArray[Shape["S,3"], Float]
    ) -> Transformation:
        """Returns the ligand transformation that aligns the ligand
        to the reference from a frame from the ligand-receptor system"""

        if isinstance(positions, unit.Quantity):
            positions = positions.value_in_unit(unit.nanometer)
        ref_positions = self.reference / unit.nanometer

        return Transformation.from_alignment(ref_positions, positions)

    def get_lig_transformation_batch(
        self, frames: NDArray[Shape["F,S,3"], Float]
    ) -> Tuple[Rotation, NDArray[Shape["F,3"], Float]]:
        """Returns the ligand transformations that aligns the ligand
        to the reference from ligand-receptor frames"""

        return Transformation.concatenate(
            [self.get_lig_transformation(f) for f in frames]
        )

    def U(self, positions: NDArray[Shape["S,3"], Float]) -> Float:
        """Energy of the ligand in the harmonic restraint. Positions should be
        the positions of the whole system."""

        T = self.get_lig_transformation(positions)
        return -self.logprob_unnorm(T) * self.kT

class BinghamRestraint(TransformationalRestraint):
    """Defines a Bingham distribution for the rotations and
    a multivariate Gaussian distribution for the translations.

    This can also be fit from data"""

    t_dist: stats._multivariate.multivariate_normal_frozen
    R_dist: BinghamDistribution
    bandwidth: Optional[float]  # bandwidth for mean shift clustering (used for caching)

    def sample_transformation(self) -> Transformation:
        """Samples a random transformation  (rotation and translation)
        from the Boltzmann distribution of this restraint"""
        return self.sample_transformation_batch(1)[0]

    def sample_transformation_batch(self, n: int) -> MultiTransformation:
        ts = self.t_dist.rvs(n).astype(np.float32)
        if n == 1:  # smh
            ts = ts[None]
        quats = self.R_dist.random_samples(n)
        Rs = Rotation.from_quat(quats)
        return MultiTransformation(Rs, ts)

    def logprob_unnorm(
        self,
        T: Transformation,
    ) -> Float:
        """Returns the log probability of a given transformation under this restraint"""
        return self.logprob_unnorm_batch(Transformation.concatenate([T]))[0]

    def logprob_unnorm_batch(
        self, mT: MultiTransformation
    ) -> NDArray[Shape["N"], Float]:
        """Returns the log probability of a batch of transformations under this restraint"""
        ts = mT.t
        Rs = mT.R.as_quat()
        U_t = multivariate_logpdf_unnorm(self.t_dist, ts)
        U_r = self.R_dist.logpdf_unnorm(Rs)
        return U_t + U_r
    
    def logprob(self, T: Transformation) -> float:
        ts = T.t[None]
        Rs = T.R.as_quat()[None]
        U_t = self.t_dist.logpdf(ts)
        U_r = self.R_dist.logpdf(Rs)[0]
        return U_t + U_r

    def get_cache_str(self) -> str:
        return f"bingham_{self.bandwidth}_{super().get_cache_str()}"

    def from_frames(
        reference: NDArray[Shape["S,3"], Float],  # reference positions
        kT: unit.Quantity,  # temperature times Boltzmann constant
        frames: NDArray[Shape["F,S,3"], Float],  # ligand-receptor frames
        bandwidth: Optional[
            float
        ],  # bandwidth for mean shift clustering (if None, estimate)
    ) -> "BinghamRestraint":

        tmp_restraint = TransformationalRestraint(reference, kT)
        Ts = tmp_restraint.get_lig_transformation_batch(frames)
        # use quaternions -- those are smoother
        data = np.concatenate([Ts.R.as_quat(), Ts.t], axis=-1)

        bandwidth_impl = estimate_bandwidth(data) if bandwidth is None else bandwidth

        if bandwidth_impl == 0.0:
            # print(f"Warning: failed to estimate bandwidth for mean shift; using all the data (N={len(data)})")
            largest_mode_samples = data
        else:
            ms = MeanShift(bandwidth=bandwidth_impl).fit(data)
            labels = ms.labels_

            # find the largest mode
            unique, counts = np.unique(labels, return_counts=True)
            largest_mode_label = unique[np.argmax(counts)]

            # Isolate the samples of the largest mode
            largest_mode_samples = data[labels == largest_mode_label]

        R_samples = largest_mode_samples[:, :4]
        t_samples = largest_mode_samples[:, 4:]

        # Fit a normal distribution to the translation samples
        mean = np.mean(t_samples, axis=0)
        cov = np.cov(t_samples, rowvar=False)
        t_dist = stats.multivariate_normal(mean=mean, cov=cov)

        # Fit a Bingham distribution to the rotation samples
        R_dist = BinghamDistribution.fit(R_samples)

        return BinghamRestraint(
            reference,
            kT,
            t_dist,
            R_dist,
            bandwidth
        )