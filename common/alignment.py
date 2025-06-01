
import numpy as np

# Thanks, Guillaume Bouvier!
# source: https://gist.github.com/bougui505/23eb8a39d7a601399edc7534b28de3d4

def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Numpy array of shape (N,D) -- Point Cloud to Align (source)
        -    B: Numpy array of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = np.asarray([[1., 1.], [2., 2.], [1.5, 3.]])
        >>> R0 = np.asarray([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]])
        >>> B = (R0.dot(A.T)).T
        >>> t0 = np.array([3., 3.])
        >>> B += t0
        >>> B.shape
        (3, 2)
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.dot(A.T)).T + t
        >>> rmsd = np.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        2.5639502485114184e-16
        >>> B *= np.array([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.dot(A.T)).T + t
        >>> rmsd = np.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        2.5639502485114184e-16
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.dot(B_c)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Rotation matrix
    R = V.dot(U.T)
    # Translation vector
    t = b_mean - R.dot(a_mean)
    return R, t

def find_rigid_alignment_batched(A, B):
    """ Assumes A has an extra batch dimension at the beginning.
    TODO: actually vectorize this
    """

    Rs = []
    Ts = []
    for frame in A:
        R, T = find_rigid_alignment(frame, B)
        Rs.append(R)
        Ts.append(T)
    Rs = np.array(Rs)
    Ts = np.array(Ts)

    return Rs, Ts

def rigid_align(A, B):
    """ Aligns A to B using Kabsch algorithm """
    R, t = find_rigid_alignment(A, B)
    return R.dot(A.T).T + t

def rigid_align_batched(A, B):
    """ Aligns A to B using Kabsch algorithm """
    Rs, Ts = find_rigid_alignment_batched(A, B)
    aligned = np.zeros_like(A)
    for i, (R, T) in enumerate(zip(Rs, Ts)):
        aligned[i] = R.dot(A[i].T).T + T
    return aligned

def align_subset(to_align, reference, indices):
    """ Aligns the subset of to_align specified by indices to the reference """
    R, t = find_rigid_alignment(to_align[indices], reference)
    return R.dot(to_align.T).T + t

def align_subset_batched(to_align, reference, indices):
    """ Aligns the subset of to_align specified by indices to the reference """
    Rs, Ts = find_rigid_alignment_batched(to_align[:, indices], reference)
    aligned = np.zeros_like(to_align)
    for i, (R, T) in enumerate(zip(Rs, Ts)):
        aligned[i] = R.dot(to_align[i].T).T + T
    return aligned
