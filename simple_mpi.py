from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"Rank {comm.rank}/{comm.size}")
