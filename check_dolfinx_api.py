import dolfinx.fem as fem
import dolfinx.mesh as mesh
from mpi4py import MPI
import numpy as np

domain = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
V = fem.functionspace(domain, ("DG", 0))
print(f"Element type: {type(V.element)}")

try:
    pts = V.element.interpolation_points()
    print("interpolation_points is a method")
    print(pts.shape)
except TypeError as e:
    print(f"interpolation_points call failed: {e}")
    try:
        pts = V.element.interpolation_points
        print("interpolation_points is a property")
        print(type(pts))
        print(pts.shape)
    except Exception as e:
        print(f"Error accessing property: {e}")
except Exception as e:
    print(f"Other error: {e}")
