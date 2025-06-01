from array import array
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, shared_memory
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from scipy import sparse
import scipy
from tqdm import tqdm

def tanimoto(fp1, fp2):
    inter = np.logical_and(fp1, fp2)
    union = np.logical_or(fp1, fp2)
    return inter.sum()/union.sum()

def batch_tanimoto(fp, fps):
    inter = np.logical_and(fp, fps)
    union = np.logical_or(fp, fps)
    sims = inter.sum(-1)/union.sum(-1)
    return sims

def get_bytes(a):
    return a.data.nbytes + a.row.nbytes + a.col.nbytes

def batch_tanimoto_faster(fp, fps):
    inter = np.logical_and(fp, fps)
    inter_sum = inter.sum(-1)
    sims = inter_sum/(fps.sum(-1) + fp.sum() - inter_sum)
    return sims

def batch_tanimoto_faster_shared(fp_shape, fp_shm_name, fp_sum_shape, fp_sum_shm_name, idx):
    fp_shm = shared_memory.SharedMemory(name=fp_shm_name)
    fps = np.ndarray(fp_shape, dtype=bool, buffer=fp_shm.buf)
    fp_sum_shm = shared_memory.SharedMemory(name=fp_sum_shm_name)
    fp_sum = np.ndarray(fp_sum_shape, dtype=int, buffer=fp_sum_shm.buf)

    fp = fps[idx]
    inter = np.logical_and(fp, fps)
    inter_sum = inter.sum(-1)
    sims = inter_sum/(fp_sum + fp.sum() - inter_sum)
    
    sims[sims < 0.3] = 0.0
    ssim = sparse.coo_matrix(sims)

    return ssim

TANIMOTO_CPUS = 16
def get_tanimoto_matrix(fps):
    try:
        fp_sum = fps.sum(-1)

        fp_shm = shared_memory.SharedMemory(create=True, size=fps.nbytes)
        fps_shared = np.ndarray(fps.shape, dtype=bool, buffer=fp_shm.buf)
        fps_shared[:] = fps[:]
        
        fp_sum_shm = shared_memory.SharedMemory(create=True, size=fp_sum.nbytes)
        fp_sum_shared = np.ndarray(fp_sum.shape, dtype=int, buffer=fp_sum_shm.buf)
        fp_sum_shared[:] = fp_sum[:]

        sim_func = partial(batch_tanimoto_faster_shared, fps.shape, fp_shm.name, fp_sum.shape, fp_sum_shm.name)
        with Pool(TANIMOTO_CPUS) as p:
            cols = list(tqdm(p.imap(sim_func, range(len(fps))), total=len(fps)))
        return sparse.vstack(cols)
    finally:
        fp_shm.close()
        fp_shm.unlink()
        fp_sum_shm.close()
        fp_sum_shm.unlink()

def sample_diverse_set(tan_mat, tan_cutoff=0.3):
    """ Sample the largest diverse set of molecules from a Tanimoto similarity matrix
    with a given tanimoto cutoff """
    import graph_tool as gt
    from graph_tool.topology import max_independent_vertex_set

    tan_mask = tan_mat.data > tan_cutoff
    tan_graph = gt.Graph(
        list(zip(tan_mat.row[tan_mask], tan_mat.col[tan_mask])), directed=False
    )

    diverse_mask = np.array(list(max_independent_vertex_set(tan_graph)), dtype=bool)
    return np.where(diverse_mask)[0]