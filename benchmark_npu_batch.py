
import logging
import time
import numpy as np
import openvino.runtime as ov
from alphabuilder.src.neural.openvino_inference import AlphaBuilderOpenVINO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

def benchmark():
    checkpoint_path = "checkpoints/warmup_simple_backbone/best_model.pt"
    
    modes = [
        ("LATENCY", [1, 2, 4, 8, 16]),
        ("THROUGHPUT", [4, 8, 16, 32, 64])
    ]

    for mode_name, batch_sizes in modes:
        print(f"\n========================================================")
        print(f"Testing Mode: {mode_name}")
        print(f"========================================================")
        
        # Initialize Backend with specific hint
        backend = AlphaBuilderOpenVINO(
            checkpoint_path, 
            device='AUTO', # Will likely pick GPU
            model_config={"performance_hint": mode_name}
        )
        print(f"Device: {backend.device} | Hint: {mode_name}")

        results = []
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking Batch Size: {batch_size}...")
            
            # Create dummy inputs
            # Input shape: (B, 7, 64, 32, 8)
            dummy_states = [np.random.randn(7, 64, 32, 8).astype(np.float32) for _ in range(batch_size)]
            
            # Warmup
            try:
                for _ in range(5):
                    backend.predict_batch(dummy_states)
                    
                # Benchmark
                start_time = time.time()
                n_iters = 50
                if batch_size >= 32: n_iters = 25 
                
                for _ in range(n_iters):
                    backend.predict_batch(dummy_states)
                    
                end_time = time.time()
                total_time = end_time - start_time
                avg_latency = (total_time / n_iters) * 1000 # ms
                ips = (n_iters * batch_size) / total_time
                
                print(f"  > Latency: {avg_latency:.2f} ms")
                print(f"  > Throughput: {ips:.2f} IPS")
                
                results.append((batch_size, ips, avg_latency))
            except Exception as e:
                print(f"Failed at batch size {batch_size}: {e}")

        print(f"\nSummary for {mode_name}:")
        print(f"{'Batch':<10} | {'IPS':<10} | {'Latency (ms)':<15}")
        print("-" * 40)
        for res in results:
             print(f"{res[0]:<10} | {res[1]:<10.1f} | {res[2]:<15.2f}")

if __name__ == "__main__":
    benchmark()
