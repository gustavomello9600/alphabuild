# AlphaBuilder

Projeto de otimização topológica usando redes neurais e SIMP/FEniTop.

**📋 Especificação Atual:** [specs/CURRENT.md](specs/CURRENT.md) → [v3.1](specs/v3.1_spec.md)

## Estrutura do Projeto

```
alphabuilder/
├── src/
│   ├── core/             # Física e tensores
│   ├── logic/            # FEniTop, storage, geração de estruturas
│   ├── neural/           # Arquitetura neural (PyTorch/MONAI)
│   └── utils/            # Utilitários (logger)
└── web/                  # Frontend React

scripts raiz:
├── run_data_harvest.py   # Geração de dados de treino
└── extract_mock_episode.py # Exportação para frontend
```

## Setup do Ambiente

### Instalação do Miniforge (Arch Linux)

```bash
yay -S miniforge
echo '[ -f /opt/miniforge/etc/profile.d/conda.sh ] && source /opt/miniforge/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

### Criar e Ativar o Ambiente

```bash
mamba env create -f environment.yml
mamba activate alphabuilder
```

## Geração de Dados

```bash
# Gerar episódio com estratégia Bezier
mpirun -np 4 python run_data_harvest.py --strategy BEZIER --episodes 1

# Gerar episódio Full Domain
mpirun -np 4 python run_data_harvest.py --strategy FULL_DOMAIN --episodes 1

# Exportar para frontend
python extract_mock_episode.py --output alphabuilder/web/public/data/mock_episode_bezier.json
```

## Dependências Principais

- **FEniCSx/DOLFINx**: Solver FEM
- **PyTorch + MONAI**: Deep Learning (Swin-UNETR)
- **MPI (mpi4py)**: Paralelismo
- **React + Vite**: Frontend
