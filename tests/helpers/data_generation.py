"""
Helper para geração de dados de teste.
Executa run_data_harvest.py como subprocess para uso em testes.
"""
import subprocess
import sys
import sqlite3
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def generate_test_episode(
    db_path: Path,
    resolution: tuple,
    strategy: str,
    seed: int,
    max_iter: int = 60  # Aumentado para ter dados de REFINEMENT suficientes
) -> str:
    """
    Gera um episódio de teste e salva no banco de dados.
    
    Executa run_data_harvest.py com mpirun para gerar um episódio.
    
    Args:
        db_path: Caminho para o banco de dados
        resolution: Tupla (nx, ny, nz) com a resolução
        strategy: 'BEZIER' ou 'FULL_DOMAIN'
        seed: Seed para reprodutibilidade
        max_iter: Número máximo de iterações SIMP (default 30 para testes)
    
    Returns:
        episode_id: UUID do episódio gerado
    """
    # Garante que o diretório do DB existe
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Conta episódios antes
    episodes_before = _count_episodes(db_path)
    
    # Monta comando
    resolution_str = f"{resolution[0]}x{resolution[1]}x{resolution[2]}"
    
    cmd = [
        "mpirun", "-np", "2",  # 2 processos para testes
        sys.executable,
        str(PROJECT_ROOT / "run_data_harvest.py"),
        "--episodes", "1",
        "--db-path", str(db_path),
        "--resolution", resolution_str,
        "--strategy", strategy,
        "--seed-offset", str(seed),
        "--max-iter", str(max_iter)  # Limita iterações para teste rápido
    ]
    
    print(f"Executando: {' '.join(cmd)}")
    
    # Executa
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300  # 5 minutos max para testes rápidos
    )
    
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"run_data_harvest falhou com código {result.returncode}")
    
    # Pega o episode_id do novo episódio
    episode_id = _get_latest_episode_id(db_path, episodes_before)
    
    if episode_id is None:
        print(f"STDOUT:\n{result.stdout}")
        raise RuntimeError("Nenhum novo episódio foi gerado")
    
    return episode_id


def _count_episodes(db_path: Path) -> int:
    """Conta episódios únicos no DB."""
    if not db_path.exists():
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT episode_id) FROM training_data")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def _get_latest_episode_id(db_path: Path, episodes_before: int) -> str:
    """Retorna o ID do episódio mais recente."""
    if not db_path.exists():
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Pega o episódio mais recente
    cursor.execute("""
        SELECT DISTINCT episode_id 
        FROM training_data 
        ORDER BY id DESC 
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0]
    return None
