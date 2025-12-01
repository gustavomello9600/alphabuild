"""
Helper para geração de dados de teste.
Wrapper simplificado do run_data_harvest.py para uso em testes.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_test_episode(
    db_path: Path,
    resolution: tuple,
    strategy: str,
    seed: int
) -> str:
    """
    Gera um episódio de teste e salva no banco de dados.
    
    Args:
        db_path: Caminho para o banco de dados
        resolution: Tupla (nx, ny, nz) com a resolução
        strategy: 'BEZIER' ou 'FULL_DOMAIN'
        seed: Seed para reprodutibilidade
    
    Returns:
        episode_id: UUID do episódio gerado
    """
    # TODO: Implementar geração de dados v3.1 com 7 canais
    raise NotImplementedError(
        "Geração de dados v3.1 ainda não implementada. "
        "Próximo passo: atualizar storage.py e tensor_utils.py para 7 canais."
    )

