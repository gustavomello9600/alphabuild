import dolfinx.fem.petsc
import inspect

print("LinearProblem init signature:")
print(inspect.signature(dolfinx.fem.petsc.LinearProblem.__init__))
