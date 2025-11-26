import os
import subprocess
import sys

def install_dependencies():
    print("Installing dependencies for AlphaBuilder v1.1...")
    packages = [
        "torch",
        "monai",
        "einops",
        "numpy",
        "tqdm"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def main():
    print("=== AlphaBuilder Colab Entrypoint ===")
    
    # 1. Install
    install_dependencies()
    
    # 2. Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. Training will be slow.")
        
    # 3. Run Smoke Test
    print("\n--- Running Architecture Verification ---")
    from alphabuilder.src.neural.model_arch import build_model
    try:
        model = build_model()
        x = torch.randn(1, 5, 64, 32, 32)
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
        y = model(x)
        print("Smoke Test Passed! Model is ready.")
    except Exception as e:
        print(f"Smoke Test Failed: {e}")
        return

    print("\nReady to train. Run 'python -m alphabuilder.src.neural.train' to start.")

if __name__ == "__main__":
    main()
