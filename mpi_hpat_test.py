from mpi4py import MPI

comm = MPI.COMM_WORLD
node_id = comm.Get_rank()
num_pes = comm.Get_size()

import numpy as np
import hpat

@hpat.jit
def f():
    return np.ones(10).sum()

#print(f())
