# 🤖 MISSÃO: AGENTE 01 (PHYSICS_CORE)

**Função:** Especialista em Simulação Física (FEM) e Oráculo de Ground Truth.
**Paradigma:** Funcional Prático (Stateless).
**Stack:** Python 3.10, FEniCSx (dolfinx), UFL, MPI, NumPy.

---

## 1. CONTEXTO E AMBIENTE
Você é o **Oráculo de Verdade** do AlphaBuilder.
Sua missão não é mais apenas "simular", mas fornecer o sinal de recompensa exato (Dense Reward) que treinará a rede neural.
O Agente 02 (Architect) chamará você a cada passo da Fase 2 para perguntar: "Quão boa é esta estrutura?".

**Mudança de Paradigma:**
*   **Antes:** Simulação passiva.
*   **Agora:** Simulação Instrucional. Você deve calcular não apenas o deslocamento, mas a **Energia de Deformação (Compliance)** precisa, pois é isso que a rede tentará minimizar.

---

## 2. ESTRUTURAS DE DADOS (INTERFACE)

```python
from dataclasses import dataclass
import numpy as np
import dolfinx

@dataclass(frozen=True)
class PhysicalProperties:
    """Constantes Físicas e Hiperparâmetros."""
    E_solid: float = 1.0          
    E_void: float = 1e-6          
    nu: float = 0.3               
    # Limites de Projeto
    max_volume_fraction: float = 0.3
    max_displacement_limit: float = 2.0

@dataclass(frozen=True)
class FEMContext:
    """Contexto pré-compilado do FEniCSx."""
    mesh: dolfinx.mesh.Mesh
    V: dolfinx.fem.FunctionSpace        
    D: dolfinx.fem.FunctionSpace        
    u_sol: dolfinx.fem.Function         
    material_field: dolfinx.fem.Function 
    problem: Any                        
    dof_map: np.ndarray                 

@dataclass(frozen=True)
class SimulationResult:
    """Output Rico para Treinamento."""
    compliance: float      # Energia total (Objetivo de Minimização)
    max_u: float           # Deslocamento de pico (Restrição)
    volume: float          # Fração de volume atual
    is_valid: bool         # Se convergiu e não explodiu
    reward_signal: float   # Valor calculado para o RL (ex: -Compliance - lambda*Vol)
```

---

## 3. TAREFAS DE IMPLEMENTAÇÃO

### 3.1. Normalização de Inputs
O sistema agora opera com forças normalizadas.
*   **`apply_normalized_load(ctx: FEMContext, force_vector: Tuple[float, float, float], load_coords: List[Coord])`**
    *   O vetor de força `(Fx, Fy, Fz)` vem no intervalo $[-1, 1]$.
    *   Você deve escalar isso para magnitudes físicas reais se necessário para estabilidade numérica do solver, ou manter adimensional se consistência for mantida.
    *   Recomendação: Mantenha adimensional ($F=1.0$ é a carga unitária padrão).

### 3.2. Solver Otimizado (Reutilização)
Mantenha a estratégia de compilação única (JIT).
*   **`solve_topology(...)`**
    *   Atualize o campo de material.
    *   Resolva o sistema linear $Ku = f$.
    *   Calcule a Compliance: $C = f^T u$. Esta é a métrica mais robusta para otimização topológica.

---

## 4. DICAS TÉCNICAS

### Estabilidade em 3D
Simulações 3D de topologia esparsa são propensas a matrizes singulares.
*   **Solver:** Use `PETSc` com solver direto `MUMPS` se disponível, ou `CG` + `AMG` (Algebraic Multigrid) para grandes volumes.
*   **Material Fraco:** Certifique-se de que $E_{void}$ seja alto o suficiente para evitar singularidade, mas baixo o suficiente para não afetar a física ($10^{-6}$ é usual).

---

## 5. VALIDAÇÃO

1.  **Teste de Compliance:** Para uma viga em balanço cheia, calcule $C_{analytical}$ e compare com $C_{fem}$.
2.  **Teste de Sensibilidade:** Remova um voxel na base da viga (crítico) e verifique se a Compliance aumenta drasticamente. Remova um voxel na ponta (não crítico) e verifique se o aumento é marginal. Isso confirma que o "Oráculo" está dando os sinais corretos para a rede aprender.