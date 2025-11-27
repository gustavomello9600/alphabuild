import dolfinx.fem
import inspect

print("Attributes in dolfinx.fem:")
print(dir(dolfinx.fem))

if hasattr(dolfinx.fem, 'functionspace'):
    print("\nfunctionspace signature:")
    print(inspect.signature(dolfinx.fem.functionspace))
else:
    print("\nfunctionspace NOT found.")

if hasattr(dolfinx.fem, 'FunctionSpace'):
    print("\nFunctionSpace init signature:")
    print(inspect.signature(dolfinx.fem.FunctionSpace.__init__))
