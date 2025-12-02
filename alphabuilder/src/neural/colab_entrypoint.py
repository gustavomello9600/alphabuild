"""
AlphaBuilder v3.1 - Colab/Kaggle Entrypoint

Script para configuração e validação do ambiente em notebooks.
Instala dependências, verifica GPU e roda smoke tests.

Uso no Colab/Kaggle:
    !git clone https://github.com/user/alphabuild.git
    %cd alphabuild
    !python -m alphabuilder.src.neural.colab_entrypoint
"""
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
# File: .../alphabuilder/src/neural/colab_entrypoint.py
# Project root: 4 levels up
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def install_dependencies():
    """Instala dependências necessárias para treino."""
    print("=" * 60)
    print("📦 Instalando dependências para AlphaBuilder v3.1...")
    print("=" * 60)
    
    # Dependências Python básicas
    packages = [
        "torch",
        "monai>=1.0.0",
        "einops",
        "numpy",
        "scipy",
        "tqdm",
    ]
    
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + packages
    )
    print("✓ Dependências Python instaladas")
    
    # Instalação do FEniCSx
    print("\n📦 Verificando FEniCSx...")
    try:
        import dolfinx
        print(f"✓ FEniCSx já instalado: {dolfinx.__version__}")
    except ImportError:
        print("⚠️  FEniCSx não encontrado.")
        print("\n   Para instalar FEniCSx:")
        print("   ─────────────────────────────────────────────")
        print("   📌 Google Colab:")
        print("      Execute em uma célula antes deste script:")
        print("      ```")
        print("      try:")
        print("          import dolfinx")
        print("      except ImportError:")
        print("          !wget 'https://fem-on-colab.github.io/releases/fenicsx-install-real.sh' -O /tmp/fenicsx-install.sh")
        print("          !bash /tmp/fenicsx-install.sh")
        print("      ```")
        print("")
        print("   📌 Kaggle:")
        print("      FEniCSx não é suportado nativamente no Kaggle.")
        print("      Use apenas para treino (dados pré-gerados).")
        print("")
        print("   📌 Local (Conda):")
        print("      conda install -c conda-forge fenics-dolfinx mpich mpi4py")
        print("   ─────────────────────────────────────────────")
        print("")
        print("   ℹ️  FEniCSx é necessário APENAS para geração de dados.")
        print("      O treino da rede neural funciona sem ele.")


def check_gpu():
    """Verifica disponibilidade de GPU."""
    import torch
    
    print("\n" + "=" * 60)
    print("🖥️  Verificando Hardware...")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Detectada: {gpu_name}")
        print(f"  Memória: {gpu_mem:.1f} GB")
        return True
    else:
        print("⚠️  WARNING: Nenhuma GPU detectada!")
        print("  O treino será MUITO lento em CPU.")
        return False


def run_smoke_tests():
    """Executa testes de validação da arquitetura."""
    import torch
    
    print("\n" + "=" * 60)
    print("🧪 Executando Smoke Tests...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test 1: Model instantiation
    print("\n[1/4] Instanciando modelo v3.1...")
    try:
        from alphabuilder.src.neural.model import AlphaBuilderV31
        
        model = AlphaBuilderV31(
            in_channels=7,
            out_channels=2,
            feature_size=24,
            use_swin=True  # Tenta usar SwinUNETR
        ).to(device)
        print("✓ Modelo instanciado com sucesso")
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parâmetros totais: {total_params:,}")
        print(f"  Parâmetros treináveis: {trainable_params:,}")
        
    except Exception as e:
        print(f"✗ Falha na instanciação: {e}")
        return False
    
    # Test 2: Forward pass (standard resolution)
    print("\n[2/4] Forward pass (64x32x8)...")
    try:
        x = torch.randn(1, 7, 64, 32, 8).to(device)
        with torch.no_grad():
            policy, value = model(x)
        
        assert policy.shape == (1, 2, 64, 32, 8), f"Policy shape errado: {policy.shape}"
        assert value.shape == (1, 1), f"Value shape errado: {value.shape}"
        print(f"✓ Policy shape: {policy.shape}")
        print(f"✓ Value shape: {value.shape}, valor: {value.item():.4f}")
        
    except Exception as e:
        print(f"✗ Falha no forward pass: {e}")
        return False
    
    # Test 3: Dynamic resolution (non-standard but valid for Swin)
    print("\n[3/4] Dynamic padding (48x24x16)...")
    try:
        # Swin requires reasonable spatial dimensions (bottleneck must be > 1x1x1)
        x_small = torch.randn(1, 7, 48, 24, 16).to(device)
        with torch.no_grad():
            policy_s, value_s = model(x_small)
        
        assert policy_s.shape == (1, 2, 48, 24, 16), f"Policy shape errado: {policy_s.shape}"
        print(f"✓ Dynamic resolution OK: {policy_s.shape}")
        
    except Exception as e:
        print(f"✗ Falha no dynamic padding: {e}")
        return False
    
    # Test 4: Backward pass (gradient check)
    print("\n[4/4] Backward pass (gradient flow)...")
    try:
        model.train()
        x_grad = torch.randn(1, 7, 64, 32, 8, requires_grad=True).to(device)
        policy_g, value_g = model(x_grad)
        
        loss = policy_g.sum() + value_g.sum()
        loss.backward()
        
        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.parameters()
        )
        assert has_grad, "Nenhum gradiente encontrado!"
        print("✓ Gradientes fluindo corretamente")
        
    except Exception as e:
        print(f"✗ Falha no backward pass: {e}")
        return False
    
    return True


def run_data_augmentation_tests():
    """Testa funções de data augmentation."""
    import numpy as np
    
    print("\n" + "=" * 60)
    print("🔄 Testando Data Augmentation...")
    print("=" * 60)
    
    try:
        from alphabuilder.src.neural.augmentation import (
            rotate_90_z, flip_y, erosion_attack, load_multiplier, sabotage, saboteur
        )
        
        # Create sample data
        state = np.random.rand(7, 32, 16, 4).astype(np.float32)
        state[0] = (state[0] > 0.5).astype(np.float32)  # Binary density
        policy = np.zeros((2, 32, 16, 4), dtype=np.float32)
        
        # Test each augmentation
        augmentations = [
            ("rotate_90_z", lambda: rotate_90_z(state, policy)),
            ("flip_y", lambda: flip_y(state, policy)),
            ("erosion_attack", lambda: erosion_attack(state, policy, 0.5)),
            ("load_multiplier", lambda: load_multiplier(state, policy, 0.5)),
            ("sabotage", lambda: sabotage(state, policy, 0.5)),
            ("saboteur", lambda: saboteur(state, policy, 0.5)),
        ]
        
        for name, func in augmentations:
            try:
                result = func()
                print(f"✓ {name}")
            except Exception as e:
                print(f"✗ {name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Falha ao importar augmentations: {e}")
        return False


def main():
    """Entrypoint principal."""
    print("\n" + "=" * 60)
    print("🏗️  AlphaBuilder v3.1 - Setup Colab/Kaggle")
    print("=" * 60)
    
    # 1. Install dependencies
    install_dependencies()
    
    # 2. Check GPU
    has_gpu = check_gpu()
    
    # 3. Run smoke tests
    model_ok = run_smoke_tests()
    
    # 4. Test augmentations
    aug_ok = run_data_augmentation_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Resumo")
    print("=" * 60)
    
    print(f"  GPU disponível: {'✓ Sim' if has_gpu else '✗ Não'}")
    print(f"  Modelo v3.1:    {'✓ OK' if model_ok else '✗ Falhou'}")
    print(f"  Augmentation:   {'✓ OK' if aug_ok else '✗ Falhou'}")
    
    if model_ok and aug_ok:
        print("\n🎉 Ambiente pronto para treino!")
        print("\nPróximos passos:")
        print("  1. Monte o Google Drive: drive.mount('/content/drive')")
        print("  2. Copie os dados: cp /content/drive/MyDrive/data/*.db ./data/")
        print("  3. Inicie o treino:")
        print("     from alphabuilder.src.neural.trainer import train_one_epoch")
        print("     from alphabuilder.src.neural.model import AlphaBuilderV31")
        print("     from alphabuilder.src.neural.dataset import TopologyDatasetV31")
    else:
        print("\n⚠️  Alguns testes falharam. Verifique os erros acima.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

