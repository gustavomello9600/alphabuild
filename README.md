# AlphaBuilder

Projeto de otimização topológica usando Vision Transformers e MCTS.

## Estrutura do Projeto

```
alphabuilder/
├── agents_docs/          # Documentação das Missões e especificações
├── src/
│   ├── core/             # Agente 01 (Física/FEM)
│   ├── logic/            # Agente 02 (MCTS/Game)
│   ├── neural/           # Agente 03 (ViT)
│   ├── api/              # Agente 04 (Backend)
│   ├── web/              # Agente 04 (Frontend)
│   └── analysis/         # Agente 05 (Scripts/Análise)
├── data/                 # training_data.db
└── thesis/               # LaTeX e figuras
```

## Setup do Ambiente

Este projeto usa **miniforge** (conda) para gerenciamento de dependências.

### Instalação do Miniforge (Arch Linux)

```bash
yay -S miniforge
echo '[ -f /opt/miniforge/etc/profile.d/conda.sh ] && source /opt/miniforge/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

### Criar e Ativar o Ambiente

```bash
# Criar ambiente
mamba env create -f environment.yml

# Ativar ambiente
mamba activate alphabuilder
```

### Ou criar manualmente:

```bash
# 1. Criar ambiente com Python 3.10
mamba create -n alphabuilder python=3.10

# 2. Ativar
mamba activate alphabuilder

# 3. Instalar Core Científico (Via Conda-Forge)
mamba install -c conda-forge fenics-dolfinx mpich pyvista matplotlib scipy numpy pandas

# 4. Instalar IA e Web (Via Pip)
pip install tensorflow keras fastapi uvicorn[standard] plotly streamlit-drawable-canvas
```

## Verificar Instalação

```bash
mamba activate alphabuilder
python -c "import dolfinx; import tensorflow as tf; import fastapi; print('✓ FEniCSx:', dolfinx.__version__); print('✓ TensorFlow:', tf.__version__); print('✓ FastAPI:', fastapi.__version__)"
```

## Dependências Principais

- **FEniCSx 0.10.0**: Solver FEM
- **TensorFlow 2.20.0**: Deep Learning
- **FastAPI 0.122.0**: Backend API
- **Python 3.10**: Versão estável para compatibilidade