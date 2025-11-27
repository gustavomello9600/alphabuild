import dolfinx
from dolfinx import fem, mesh
from mpi4py import MPI

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=((0.0, 0.0), (2.0, 1.0)), n=(10, 5))
W = fem.functionspace(msh, ("DG", 0))
print(f"Type of W.element.interpolation_points: {type(W.element.interpolation_points)}")
try:
    print("Calling it...")
    pts = W.element.interpolation_points()
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    
try:
    print("Accessing as property...")
    pts = W.element.interpolation_points
    print(f"Success, shape: {pts.shape}")
except Exception as e:
    print(f"Failed as property: {e}")
