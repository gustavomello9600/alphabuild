import numpy as np
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.core.solver import solve_topology

def main():
    print("=== AlphaBuilder Physics Core Smoke Test ===")
    
    # 1. Configuração
    resolution = (32, 16) 
    
    P = 100.0
    L = 2.0
    E = 1.0
    I = 1.0**3 / 12.0
    
    analytical_disp = (P * L**3) / (3 * E * I)
    print(f"Analytical Parameters: P={P}, L={L}, E={E}, I={I:.4f}")
    print(f"Expected Max Displacement (Analytical): {analytical_disp:.6f}")
    
    props = PhysicalProperties(
        E_solid=E,
        E_void=1e-6,
        nu=0.3,
        disp_limit=5000.0
    )
    
    # 2. Inicializar Contexto
    print("Initializing FEM Context...")
    ctx = initialize_cantilever_context(resolution, props)
    
    # 3. Teste 1: Viga Sólida (Full Material)
    print("\n--- Test 1: Solid Beam (All 1s) ---")
    topology_solid = np.ones((resolution[1], resolution[0]), dtype=np.int32)
    
    result_solid = solve_topology(topology_solid, ctx, props)
    
    fem_disp = result_solid.max_displacement
    error_pct = abs(fem_disp - analytical_disp) / analytical_disp * 100.0
    
    print(f"FEM Max Displacement: {fem_disp:.6f}")
    print(f"Error: {error_pct:.2f}%")
    
    if error_pct < 15.0:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL (Error too high)")

    # 4. Teste 2: Viga Vazia (All 0s)
    print("\n--- Test 2: Void Beam (All 0s) ---")
    topology_void = np.zeros((resolution[1], resolution[0]), dtype=np.int32)
    
    result_void = solve_topology(topology_void, ctx, props)
    
    fem_disp_void = result_void.max_displacement
    expected_void_ratio = props.E_solid / props.E_void
    actual_ratio = fem_disp_void / fem_disp
    
    print(f"FEM Max Displacement (Void): {fem_disp_void:.6f}")
    print(f"Ratio (Void/Solid): {actual_ratio:.2e}")
    print(f"Expected Ratio (~E_solid/E_void): {expected_void_ratio:.2e}")
    
    if 0.5 * expected_void_ratio < actual_ratio < 2.0 * expected_void_ratio:
        print("RESULT: PASS (Linear behavior confirmed)")
    else:
        print("RESULT: WARNING (Ratio mismatch, check E_void scaling)")

if __name__ == "__main__":
    main()
