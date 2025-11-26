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
    
    print("Installing FEniCSx via fem-on-colab...")
    try:
        import fem_on_colab
        print("fem-on-colab already installed.")
    except ImportError:
        # We are likely in a script, not a notebook, so we can't use %pip directly easily without IPython.
        # But usually on Colab we run this script !python ...
        # The official way is:
        # try:
        #     import google.colab
        #     import fem_on_colab
        # except ImportError:
        #     !wget "https://fem-on-colab.github.io/releases/fenicsx-install-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
        # Since this is a python script, we use subprocess.
        subprocess.run('wget "https://fem-on-colab.github.io/releases/fenicsx-install-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"', shell=True, check=True)
        import dolfinx
        print(f"FEniCSx installed: {dolfinx.__version__}")

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
