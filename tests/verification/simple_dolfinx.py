
print("Start import")
try:
    from mpi4py import MPI
    print(f"MPI imported. Rank: {MPI.COMM_WORLD.rank}")
    import dolfinx
    print(f"Dolfinx imported version: {dolfinx.__version__}")
    from dolfinx import mesh
    from dolfinx import fem
    print("Dolfinx mesh/fem imported")
    import basix
    print(f"Basix imported: {basix.__version__}")
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 4, 4, 4, mesh.CellType.hexahedron)
    print("Mesh created")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
print("Done")
