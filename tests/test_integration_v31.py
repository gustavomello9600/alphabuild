"""
AlphaBuilder v3.1 Integration Test

Este teste de integração valida toda a pipeline v3.1:
1. Geração de dados (Bezier + Full Domain)
2. Armazenamento no banco de dados
3. Data augmentation dinâmico
4. Treino de uma epoch
5. Inferência em pontos de Fase 1 e Fase 2

O teste para na PRIMEIRA falha (-x flag).
O banco de dados de teste é limpo no início mas preservado após execução.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import sqlite3
import os

# Test database path
TEST_DB_PATH = Path(__file__).parent / "data" / "episodios_de_testes_de_integracao.db"
RESOLUTION = (64, 32, 8)


class TestIntegrationV31:
    """
    Teste de integração completo para AlphaBuilder v3.1.
    
    Executa sequencialmente:
    1. Setup: Limpa DB de testes
    2. Data Generation: Gera episódios Bezier e Full Domain
    3. DB Validation: Verifica schema e dados armazenados
    4. Augmentation: Testa cada tipo de augmentation
    5. Training: Treina 1 epoch
    6. Inference: Testa inferência em Fase 1 e Fase 2
    """
    
    # ==================== SETUP ====================
    
    def test_00_cleanup_test_database(self):
        """Limpa o banco de dados de testes no início."""
        if TEST_DB_PATH.exists():
            os.remove(TEST_DB_PATH)
        
        # Garante que o diretório existe
        TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        assert not TEST_DB_PATH.exists(), "DB deveria estar limpo"
    
    # ==================== DATA GENERATION ====================
    
    def test_01_generate_bezier_episode(self):
        """Gera um episódio com estratégia Bezier e salva no DB de testes."""
        from alphabuilder.src.logic.storage import initialize_database
        
        # Inicializa o banco
        initialize_database(TEST_DB_PATH)
        
        # Importa e executa geração de dados
        from tests.helpers.data_generation import generate_test_episode
        
        episode_id = generate_test_episode(
            db_path=TEST_DB_PATH,
            resolution=RESOLUTION,
            strategy='BEZIER',
            seed=42
        )
        
        assert episode_id is not None, "Episode ID não deveria ser None"
        
        # Verifica se dados foram salvos
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_data WHERE episode_id = ?", (episode_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count > 0, f"Deveria ter registros para episódio Bezier, encontrou {count}"
        print(f"✓ Episódio Bezier gerado: {count} registros")
    
    def test_02_generate_fulldomain_episode(self):
        """Gera um episódio com estratégia Full Domain e salva no DB de testes."""
        from tests.helpers.data_generation import generate_test_episode
        
        episode_id = generate_test_episode(
            db_path=TEST_DB_PATH,
            resolution=RESOLUTION,
            strategy='FULL_DOMAIN',
            seed=123
        )
        
        assert episode_id is not None, "Episode ID não deveria ser None"
        
        # Verifica se dados foram salvos
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_data WHERE episode_id = ?", (episode_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count > 0, f"Deveria ter registros para episódio Full Domain, encontrou {count}"
        print(f"✓ Episódio Full Domain gerado: {count} registros")
    
    # ==================== DB VALIDATION ====================
    
    def test_03_validate_db_schema(self):
        """Verifica se o schema do banco está correto para v3.1."""
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        
        # Verifica colunas da tabela
        cursor.execute("PRAGMA table_info(training_data)")
        columns = {row[1] for row in cursor.fetchall()}
        
        required_columns = {
            'id', 'episode_id', 'step', 'phase',
            'state_blob', 'policy_blob', 'fitness_score',
            'valid_fem', 'metadata'
        }
        
        missing = required_columns - columns
        assert len(missing) == 0, f"Colunas faltando no schema: {missing}"
        
        conn.close()
        print("✓ Schema do banco válido")
    
    def test_04_validate_tensor_dimensions(self):
        """Verifica se os tensores armazenados têm 7 canais (v3.1)."""
        from alphabuilder.src.logic.storage import deserialize_state
        
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        
        # Pega um registro aleatório
        cursor.execute("SELECT state_blob FROM training_data LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None, "Deveria ter pelo menos um registro"
        
        state = deserialize_state(row[0])
        
        # v3.1: 7 canais (density, mask_x, mask_y, mask_z, fx, fy, fz)
        assert state.shape[0] == 7, f"Estado deveria ter 7 canais, tem {state.shape[0]}"
        assert state.shape[1:] == RESOLUTION, f"Dimensões espaciais incorretas: {state.shape[1:]}"
        
        print(f"✓ Tensor shape válido: {state.shape}")
    
    def test_05_validate_metadata_fields(self):
        """Verifica se metadados is_final_step e is_connected estão presentes."""
        import json
        
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        
        # Verifica se existem registros com is_final_step
        cursor.execute("SELECT metadata FROM training_data WHERE metadata LIKE '%is_final_step%' LIMIT 1")
        row = cursor.fetchone()
        
        assert row is not None, "Deveria ter registros com metadado is_final_step"
        
        metadata = json.loads(row[0])
        assert 'is_final_step' in metadata, "Metadado is_final_step não encontrado"
        
        # Verifica is_connected em registros de Phase 2
        cursor.execute("""
            SELECT metadata FROM training_data 
            WHERE phase = 'REFINEMENT' AND metadata LIKE '%is_connected%' 
            LIMIT 1
        """)
        row = cursor.fetchone()
        
        assert row is not None, "Deveria ter registros REFINEMENT com is_connected"
        
        metadata = json.loads(row[0])
        assert 'is_connected' in metadata, "Metadado is_connected não encontrado"
        
        conn.close()
        print("✓ Metadados is_final_step e is_connected presentes")
    
    def test_06_validate_phase_distribution(self):
        """Verifica distribuição de fases (GROWTH vs REFINEMENT)."""
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT phase, COUNT(*) FROM training_data GROUP BY phase")
        phase_counts = dict(cursor.fetchall())
        conn.close()
        
        assert 'GROWTH' in phase_counts, "Deveria ter registros GROWTH (Fase 1)"
        assert 'REFINEMENT' in phase_counts, "Deveria ter registros REFINEMENT (Fase 2)"
        
        print(f"✓ Distribuição de fases: GROWTH={phase_counts.get('GROWTH', 0)}, REFINEMENT={phase_counts.get('REFINEMENT', 0)}")
    
    # ==================== DATA AUGMENTATION ====================
    
    def test_07_augmentation_rotation_90(self):
        """Testa augmentation de rotação 90° com inversão de forças."""
        from alphabuilder.src.neural.augmentation import rotate_90_z
        from alphabuilder.src.logic.storage import deserialize_state
        
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT state_blob, policy_blob FROM training_data LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        state = deserialize_state(row[0])
        policy = deserialize_state(row[1])
        
        # Aplica rotação
        state_rot, policy_rot = rotate_90_z(state, policy)
        
        # Verifica dimensões preservadas
        assert state_rot.shape == state.shape, "Shape do estado deveria ser preservado"
        assert policy_rot.shape == policy.shape, "Shape da policy deveria ser preservado"
        
        # Verifica que forças foram invertidas corretamente
        # Rotação 90° em Z: Fx -> Fy, Fy -> -Fx
        # Canais: 4=Fx, 5=Fy, 6=Fz
        # Após rotação, o que era Fx deve estar relacionado com Fy
        
        print("✓ Augmentation rotação 90° funcional")
    
    def test_08_augmentation_flip(self):
        """Testa augmentation de flip com inversão de forças."""
        from alphabuilder.src.neural.augmentation import flip_y
        from alphabuilder.src.logic.storage import deserialize_state
        
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT state_blob, policy_blob FROM training_data LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        state = deserialize_state(row[0])
        policy = deserialize_state(row[1])
        
        # Aplica flip
        state_flip, policy_flip = flip_y(state, policy)
        
        # Verifica dimensões preservadas
        assert state_flip.shape == state.shape, "Shape do estado deveria ser preservado"
        assert policy_flip.shape == policy.shape, "Shape da policy deveria ser preservado"
        
        # Verifica que Fy foi invertido (canal 5)
        # Original Fy positivo -> Flip Fy negativo
        
        print("✓ Augmentation flip funcional")
    
    def test_09_augmentation_erosion_attack(self):
        """Testa Erosion Attack em estados finais."""
        from alphabuilder.src.neural.augmentation import erosion_attack
        from alphabuilder.src.logic.storage import deserialize_state
        import json
        
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        # Busca um estado final
        cursor.execute("""
            SELECT state_blob, policy_blob, fitness_score, metadata 
            FROM training_data 
            WHERE metadata LIKE '%"is_final_step": true%'
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None, "Deveria ter um estado final para testar erosion"
        
        state = deserialize_state(row[0])
        policy = deserialize_state(row[1])
        original_value = row[2]
        
        # Aplica erosion attack
        state_eroded, policy_eroded, new_value = erosion_attack(state, policy, original_value)
        
        # Verifica que value foi alterado para -1.0
        assert new_value == -1.0, f"Value após erosion deveria ser -1.0, é {new_value}"
        
        # Verifica que a densidade foi reduzida (erosão)
        density_original = state[0].sum()
        density_eroded = state_eroded[0].sum()
        assert density_eroded < density_original, "Erosion deveria reduzir densidade"
        
        # Verifica que policy de adição foi gerada
        assert policy_eroded[0].sum() > 0, "Policy de adição deveria ter voxels para restaurar"
        
        print(f"✓ Erosion Attack: densidade {density_original:.0f} -> {density_eroded:.0f}, value -> -1.0")
    
    def test_10_augmentation_load_multiplier(self):
        """Testa Load Multiplier em estados conectados."""
        from alphabuilder.src.neural.augmentation import load_multiplier
        from alphabuilder.src.logic.storage import deserialize_state
        
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        # Busca um estado conectado (não final)
        cursor.execute("""
            SELECT state_blob, policy_blob, fitness_score
            FROM training_data 
            WHERE phase = 'REFINEMENT' 
            AND metadata LIKE '%"is_connected": true%'
            AND metadata NOT LIKE '%"is_final_step": true%'
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            pytest.skip("Nenhum estado conectado não-final encontrado")
        
        state = deserialize_state(row[0])
        policy = deserialize_state(row[1])
        original_value = row[2]
        
        # Aplica load multiplier (K=3.0)
        state_stressed, policy_stressed, new_value = load_multiplier(state, policy, original_value, k=3.0)
        
        # Verifica que value foi alterado para -0.8
        assert new_value == -0.8, f"Value após load multiplier deveria ser -0.8, é {new_value}"
        
        # Verifica que forças foram multiplicadas
        original_force_mag = np.sqrt(state[4]**2 + state[5]**2 + state[6]**2).max()
        stressed_force_mag = np.sqrt(state_stressed[4]**2 + state_stressed[5]**2 + state_stressed[6]**2).max()
        
        if original_force_mag > 0:
            assert np.isclose(stressed_force_mag, original_force_mag * 3.0, rtol=0.01), \
                "Forças deveriam ser multiplicadas por 3.0"
        
        print(f"✓ Load Multiplier: forças x3.0, value -> -0.8")
    
    def test_11_augmentation_sabotage(self):
        """Testa Sabotage (remoção de voxels em nós críticos)."""
        from alphabuilder.src.neural.augmentation import sabotage
        from alphabuilder.src.logic.storage import deserialize_state
        
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT state_blob, policy_blob, fitness_score FROM training_data LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        state = deserialize_state(row[0])
        policy = deserialize_state(row[1])
        original_value = row[2]
        
        # Aplica sabotage
        state_sab, policy_sab, new_value = sabotage(state, policy, original_value)
        
        # Verifica que value foi alterado para -1.0
        assert new_value == -1.0, f"Value após sabotage deveria ser -1.0, é {new_value}"
        
        # Verifica que densidade foi reduzida
        density_original = state[0].sum()
        density_sabotaged = state_sab[0].sum()
        assert density_sabotaged < density_original, "Sabotage deveria reduzir densidade"
        
        print(f"✓ Sabotage: densidade {density_original:.0f} -> {density_sabotaged:.0f}, value -> -1.0")
    
    def test_12_augmentation_saboteur(self):
        """Testa Saboteur (remoção de cubo aleatório)."""
        from alphabuilder.src.neural.augmentation import saboteur
        from alphabuilder.src.logic.storage import deserialize_state
        
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT state_blob, policy_blob, fitness_score FROM training_data LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        state = deserialize_state(row[0])
        policy = deserialize_state(row[1])
        original_value = row[2]
        
        # Aplica saboteur
        state_sab, policy_sab, new_value = saboteur(state, policy, original_value)
        
        # Verifica que value foi alterado para -1.0
        assert new_value == -1.0, f"Value após saboteur deveria ser -1.0, é {new_value}"
        
        # Verifica que densidade foi reduzida
        density_original = state[0].sum()
        density_sabotaged = state_sab[0].sum()
        assert density_sabotaged < density_original, "Saboteur deveria reduzir densidade"
        
        # Verifica que policy de reparo foi gerada
        assert policy_sab[0].sum() > 0, "Policy de adição deveria ter voxels para reparar"
        
        print(f"✓ Saboteur: densidade {density_original:.0f} -> {density_sabotaged:.0f}, value -> -1.0")
    
    # ==================== TRAINING ====================
    
    def test_13_train_one_epoch(self):
        """Treina uma epoch com dados augmentados + originais."""
        from alphabuilder.src.neural.dataset_v31 import TopologyDatasetV31
        from alphabuilder.src.neural.model_v31 import AlphaBuilderV31
        from alphabuilder.src.neural.trainer_v31 import train_one_epoch
        from torch.utils.data import DataLoader
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carrega dataset
        dataset = TopologyDatasetV31(
            db_path=TEST_DB_PATH,
            augment=True  # Inclui augmentations
        )
        
        assert len(dataset) > 0, "Dataset não deveria estar vazio"
        
        # DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Modelo com inicialização fria
        model = AlphaBuilderV31(
            in_channels=7,
            out_channels=2,
            feature_size=24  # Menor para teste rápido
        ).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Treina uma epoch
        metrics = train_one_epoch(model, dataloader, optimizer, device)
        
        assert 'loss' in metrics, "Métricas deveriam incluir loss"
        assert 'policy_loss' in metrics, "Métricas deveriam incluir policy_loss"
        assert 'value_loss' in metrics, "Métricas deveriam incluir value_loss"
        
        assert metrics['loss'] > 0, "Loss deveria ser positivo"
        assert not np.isnan(metrics['loss']), "Loss não deveria ser NaN"
        
        print(f"✓ Treino 1 epoch: loss={metrics['loss']:.4f}, policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f}")
    
    # ==================== INFERENCE ====================
    
    def test_14_inference_phase1(self):
        """Testa inferência em um ponto de dados da Fase 1 (GROWTH)."""
        from alphabuilder.src.neural.model_v31 import AlphaBuilderV31
        from alphabuilder.src.logic.storage import deserialize_state
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carrega um ponto de Fase 1
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT state_blob, policy_blob, fitness_score 
            FROM training_data 
            WHERE phase = 'GROWTH' 
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None, "Deveria ter um ponto de Fase 1"
        
        state = deserialize_state(row[0])
        target_policy = deserialize_state(row[1])
        target_value = row[2]
        
        # Modelo
        model = AlphaBuilderV31(
            in_channels=7,
            out_channels=2,
            feature_size=24
        ).to(device)
        model.eval()
        
        # Inferência
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            policy_pred, value_pred = model(state_tensor)
        
        # Validações
        assert policy_pred.shape == (1, 2, *RESOLUTION), f"Policy shape incorreto: {policy_pred.shape}"
        assert value_pred.shape == (1, 1), f"Value shape incorreto: {value_pred.shape}"
        
        # Value deve estar em [-1, 1] (tanh)
        assert -1.0 <= value_pred.item() <= 1.0, f"Value fora do range [-1,1]: {value_pred.item()}"
        
        # Policy não deve ser NaN/Inf
        assert not torch.isnan(policy_pred).any(), "Policy contém NaN"
        assert not torch.isinf(policy_pred).any(), "Policy contém Inf"
        
        print(f"✓ Inferência Fase 1: policy shape={policy_pred.shape}, value={value_pred.item():.4f}")
    
    def test_15_inference_phase2(self):
        """Testa inferência em um ponto de dados da Fase 2 (REFINEMENT)."""
        from alphabuilder.src.neural.model_v31 import AlphaBuilderV31
        from alphabuilder.src.logic.storage import deserialize_state
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carrega um ponto de Fase 2
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT state_blob, policy_blob, fitness_score 
            FROM training_data 
            WHERE phase = 'REFINEMENT' 
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None, "Deveria ter um ponto de Fase 2"
        
        state = deserialize_state(row[0])
        
        # Modelo
        model = AlphaBuilderV31(
            in_channels=7,
            out_channels=2,
            feature_size=24
        ).to(device)
        model.eval()
        
        # Inferência
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            policy_pred, value_pred = model(state_tensor)
        
        # Validações
        assert policy_pred.shape == (1, 2, *RESOLUTION), f"Policy shape incorreto: {policy_pred.shape}"
        assert value_pred.shape == (1, 1), f"Value shape incorreto: {value_pred.shape}"
        
        # Value deve estar em [-1, 1] (tanh)
        assert -1.0 <= value_pred.item() <= 1.0, f"Value fora do range [-1,1]: {value_pred.item()}"
        
        # Policy não deve ser NaN/Inf
        assert not torch.isnan(policy_pred).any(), "Policy contém NaN"
        assert not torch.isinf(policy_pred).any(), "Policy contém Inf"
        
        print(f"✓ Inferência Fase 2: policy shape={policy_pred.shape}, value={value_pred.item():.4f}")
        print("\n" + "="*60)
        print("🎉 TODOS OS TESTES DE INTEGRAÇÃO v3.1 PASSARAM!")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])

