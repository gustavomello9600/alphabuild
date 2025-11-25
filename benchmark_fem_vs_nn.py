#!/usr/bin/env python3
"""
Benchmark script to compare execution times of FEM Solver vs Neural Network Inference.
"""

import time
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.core.solver import solve_topology
from alphabuilder.src.neural.model_arch import build_3d_vit
from alphabuilder.src.neural.trainer import predict_fitness

def benchmark():
    print("=== Benchmarking FEM vs Neural Network ===")
    
    # 1. Setup Environment
    resolution = (16, 32)
    props = PhysicalProperties()
    ctx = initialize_cantilever_context(resolution, props)
    
    # Create a random topology (50% fill)
    topology = np.random.randint(0, 2, (resolution[1], resolution[0])).astype(np.int32)
    # Ensure support
    topology[0, :] = 1
    
    # 2. Benchmark FEM Solver
    print("\n--- Benchmarking FEM Solver ---")
    fem_times = []
    # Warmup
    solve_topology(topology, ctx, props)
    
    for _ in range(20):
        start = time.time()
        solve_topology(topology, ctx, props)
        end = time.time()
        fem_times.append(end - start)
    
    avg_fem = np.mean(fem_times)
    print(f"FEM Solver (1 call): {avg_fem*1000:.2f} ms")
    
    # 3. Benchmark Neural Network
    print("\n--- Benchmarking Neural Network ---")
    model = build_3d_vit()
    
    # Create dummy batch of candidates (simulating a greedy search step)
    # A 16x32 grid has 512 cells. 
    # We use a smaller batch for benchmarking to avoid OOM on CPU
    num_candidates = 32
    candidates = [topology.copy() for _ in range(num_candidates)]
    thicknesses = [1] * num_candidates
    
    # Warmup
    predict_fitness(model, candidates[:10], thicknesses[:10])
    
    # Measure Single Inference
    single_times = []
    for _ in range(20):
        start = time.time()
        predict_fitness(model, candidates[:1], thicknesses[:1])
        end = time.time()
        single_times.append(end - start)
    
    avg_nn_single = np.mean(single_times)
    print(f"NN Inference (1 candidate): {avg_nn_single*1000:.2f} ms")
    
    # Measure Batch Inference (Full Grid Scan)
    batch_times = []
    for _ in range(10):
        start = time.time()
        predict_fitness(model, candidates, thicknesses)
        end = time.time()
        batch_times.append(end - start)
        
    avg_nn_batch = np.mean(batch_times)
    print(f"NN Inference (Batch of {num_candidates}): {avg_nn_batch*1000:.2f} ms")
    
    # 4. Comparison
    print("\n--- Comparison ---")
    print(f"Ratio (1 FEM / 1 NN): {avg_fem / avg_nn_single:.1f}x faster")
    print(f"Ratio (1 FEM / Batch NN): {avg_fem / avg_nn_batch:.2f}x")
    
    if avg_nn_batch < avg_fem:
        print("\nResult: Neural Network Batch Scan is FASTER than a single FEM solve.")
    else:
        print("\nResult: Neural Network Batch Scan is SLOWER than a single FEM solve.")
        
    print(f"Note: To replace FEM completely for search, we need to evaluate {num_candidates} candidates.")
    print(f"Time to evaluate {num_candidates} candidates with FEM: {avg_fem * num_candidates:.2f} s")
    print(f"Time to evaluate {num_candidates} candidates with NN:   {avg_nn_batch:.4f} s")
    print(f"Speedup Factor for Search: {(avg_fem * num_candidates) / avg_nn_batch:.1f}x")

if __name__ == "__main__":
    benchmark()
