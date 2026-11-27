"""
Gerador de Relatório de Qualidade de Dados de Treino.

Gera um relatório .md completo com análise dos dados gerados
durante o teste de integração.
"""
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

from alphabuilder.src.logic.storage import (
    deserialize_array,
    deserialize_sparse,
    deserialize_state,
    has_new_schema,
    sparse_decode,
)


def generate_quality_report(db_path: Path, output_path: Path) -> str:
    """
    Gera um relatório completo de qualidade dos dados de treino.
    
    Args:
        db_path: Caminho para o banco de dados SQLite.
        output_path: Caminho para salvar o relatório .md.
        
    Returns:
        Caminho do relatório gerado.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    is_v2 = has_new_schema(db_path)
    table_name = "records" if is_v2 else "training_data"

    report_lines = []

    # ==================== HEADER ====================
    report_lines.append("# 📊 Relatório de Qualidade de Dados de Treino")
    report_lines.append("")
    report_lines.append(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Banco de Dados:** `{db_path.name}`")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ==================== RESUMO GERAL ====================
    report_lines.append("## 1. Resumo Geral")
    report_lines.append("")

    # Total de registros
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_records = cursor.fetchone()[0]

    # Episódios únicos
    cursor.execute(f"SELECT COUNT(DISTINCT episode_id) FROM {table_name}")
    total_episodes = cursor.fetchone()[0]

    # Por fase
    cursor.execute(f"SELECT phase, COUNT(*) FROM {table_name} GROUP BY phase")
    phase_counts = dict(cursor.fetchall())

    report_lines.append(f"| Métrica | Valor |")
    report_lines.append(f"|---------|-------|")
    report_lines.append(f"| Total de Registros | **{total_records}** |")
    report_lines.append(f"| Episódios Únicos | **{total_episodes}** |")
    report_lines.append(f"| Registros GROWTH (Fase 1) | {phase_counts.get('GROWTH', 0)} |")
    report_lines.append(f"| Registros REFINEMENT (Fase 2) | {phase_counts.get('REFINEMENT', 0)} |")
    report_lines.append("")

    # ==================== ANÁLISE POR EPISÓDIO ====================
    report_lines.append("## 2. Análise por Episódio")
    report_lines.append("")

    cursor.execute(f"""
        SELECT episode_id, 
               COUNT(*) as records,
               SUM(CASE WHEN phase = 'GROWTH' THEN 1 ELSE 0 END) as growth,
               SUM(CASE WHEN phase = 'REFINEMENT' THEN 1 ELSE 0 END) as refinement,
               AVG(fitness_score) as avg_fitness,
               MIN(fitness_score) as min_fitness,
               MAX(fitness_score) as max_fitness
        FROM {table_name}
        GROUP BY episode_id
    """)
    episodes = cursor.fetchall()

    report_lines.append("| Episode ID | Records | GROWTH | REFINEMENT | Avg Fitness | Min | Max |")
    report_lines.append("|------------|---------|--------|------------|-------------|-----|-----|")

    for ep in episodes:
        ep_id_short = ep[0][:8] + "..." if len(ep[0]) > 8 else ep[0]
        report_lines.append(
            f"| `{ep_id_short}` | {ep[1]} | {ep[2]} | {ep[3]} | "
            f"{ep[4]:.4f} | {ep[5]:.4f} | {ep[6]:.4f} |"
        )
    report_lines.append("")

    # ==================== ANÁLISE DE TENSORES ====================
    report_lines.append("## 3. Análise de Tensores")
    report_lines.append("")

    state_sample = None
    policy_sample = None
    if is_v2:
        cursor.execute("""
            SELECT r.density_blob, r.policy_add_blob, r.policy_remove_blob, r.phase,
                   e.bc_masks_blob, e.forces_blob, e.resolution
            FROM records r
            JOIN episodes e ON r.episode_id = e.episode_id
            LIMIT 10
        """)
        samples = cursor.fetchall()
        if samples:
            d_b, pa_b, pr_b, _ph, bc_b, f_b, res_j = samples[0]
            res = tuple(json.loads(res_j))
            dens = deserialize_array(d_b).astype(np.float32)
            bc_m = deserialize_array(bc_b).astype(np.float32)
            f_arr = deserialize_array(f_b).astype(np.float32)
            state_sample = np.concatenate([dens[None], bc_m, f_arr], axis=0).astype(np.float32)
            pa_arr = sparse_decode(*deserialize_sparse(pa_b), res) if pa_b else np.zeros(res, dtype=np.float32)
            pr_arr = sparse_decode(*deserialize_sparse(pr_b), res) if pr_b else np.zeros(res, dtype=np.float32)
            policy_sample = np.stack([pa_arr, pr_arr], axis=0).astype(np.float32)
    else:
        cursor.execute("SELECT state_blob, policy_blob, phase FROM training_data LIMIT 10")
        samples = cursor.fetchall()
        if samples:
            state_sample = deserialize_state(samples[0][0])
            policy_sample = deserialize_state(samples[0][1])

    if state_sample is not None and policy_sample is not None:
        
        report_lines.append(f"### 3.1 Dimensões dos Tensores")
        report_lines.append("")
        report_lines.append(f"| Tensor | Shape | Dtype |")
        report_lines.append(f"|--------|-------|-------|")
        report_lines.append(f"| State | `{state_sample.shape}` | `{state_sample.dtype}` |")
        report_lines.append(f"| Policy | `{policy_sample.shape}` | `{policy_sample.dtype}` |")
        report_lines.append("")
        
        # Análise dos canais do state
        report_lines.append(f"### 3.2 Análise dos Canais do State (7 canais v3.1)")
        report_lines.append("")
        channel_names = [
            "Density (ρ)", "Mask X (ux)", "Mask Y (uy)", "Mask Z (uz)",
            "Force X (Fx)", "Force Y (Fy)", "Force Z (Fz)"
        ]
        
        report_lines.append(f"| Canal | Nome | Min | Max | Mean | Non-zero % |")
        report_lines.append(f"|-------|------|-----|-----|------|------------|")
        
        for i, name in enumerate(channel_names):
            ch = state_sample[i]
            nonzero_pct = (ch != 0).sum() / ch.size * 100
            report_lines.append(
                f"| {i} | {name} | {ch.min():.4f} | {ch.max():.4f} | "
                f"{ch.mean():.4f} | {nonzero_pct:.2f}% |"
            )
        report_lines.append("")
        
        # Análise dos canais da policy
        report_lines.append(f"### 3.3 Análise dos Canais da Policy (2 canais)")
        report_lines.append("")
        policy_names = ["Add (Adição)", "Remove (Remoção)"]
        
        report_lines.append(f"| Canal | Nome | Min | Max | Mean | Non-zero % |")
        report_lines.append(f"|-------|------|-----|-----|------|------------|")
        
        for i, name in enumerate(policy_names):
            ch = policy_sample[i]
            nonzero_pct = (ch != 0).sum() / ch.size * 100
            report_lines.append(
                f"| {i} | {name} | {ch.min():.4f} | {ch.max():.4f} | "
                f"{ch.mean():.4f} | {nonzero_pct:.2f}% |"
            )
        report_lines.append("")
    
    # ==================== ANÁLISE DE FITNESS SCORE (VALUE TARGET) ====================
    report_lines.append("## 4. Análise de Fitness Score (Value Target)")
    report_lines.append("")
    
    # Detecta schema e busca scores
    if is_v2:
        cursor.execute("SELECT fitness_score FROM records")
    else:
        cursor.execute("SELECT fitness_score FROM training_data")
    
    all_scores = [row[0] for row in cursor.fetchall()]
    scores_array = np.array(all_scores) if all_scores else np.array([])
    
    if len(scores_array) == 0:
        report_lines.append("⚠️ Nenhum score encontrado no banco de dados.")
        report_lines.append("")
        # Salva e retorna
        report_content = "\n".join(report_lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_content, encoding='utf-8')
        return str(output_path)
    
    report_lines.append(f"### 4.1 Estatísticas Gerais")
    report_lines.append("")
    report_lines.append(f"| Estatística | Valor |")
    report_lines.append(f"|-------------|-------|")
    report_lines.append(f"| Mínimo | {scores_array.min():.6f} |")
    report_lines.append(f"| Máximo | {scores_array.max():.6f} |")
    report_lines.append(f"| Média | {scores_array.mean():.6f} |")
    report_lines.append(f"| Mediana | {np.median(scores_array):.6f} |")
    report_lines.append(f"| Desvio Padrão | {scores_array.std():.6f} |")
    report_lines.append(f"| Percentil 25 | {np.percentile(scores_array, 25):.6f} |")
    report_lines.append(f"| Percentil 75 | {np.percentile(scores_array, 75):.6f} |")
    report_lines.append("")
    
    # Distribuição por faixa
    report_lines.append(f"### 4.2 Distribuição por Faixa")
    report_lines.append("")
    bins = [(-1.1, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.1)]
    
    report_lines.append(f"| Faixa | Contagem | Percentual |")
    report_lines.append(f"|-------|----------|------------|")
    
    for low, high in bins:
        count = ((scores_array >= low) & (scores_array < high)).sum()
        pct = count / len(scores_array) * 100
        report_lines.append(f"| [{low:.1f}, {high:.1f}) | {count} | {pct:.1f}% |")
    report_lines.append("")
    
    # Por fase
    report_lines.append(f"### 4.3 Fitness por Fase")
    report_lines.append("")
    
    cursor.execute(f"""
        SELECT phase, 
               AVG(fitness_score), 
               MIN(fitness_score), 
               MAX(fitness_score),
               COUNT(*)
        FROM {table_name} 
        GROUP BY phase
    """)
    phase_stats = cursor.fetchall()
    
    report_lines.append(f"| Fase | Count | Avg | Min | Max |")
    report_lines.append(f"|------|-------|-----|-----|-----|")
    
    for ps in phase_stats:
        report_lines.append(f"| {ps[0]} | {ps[4]} | {ps[1]:.4f} | {ps[2]:.4f} | {ps[3]:.4f} |")
    report_lines.append("")
    
    # ==================== ANÁLISE DE POLICY TARGETS ====================
    report_lines.append("## 5. Análise de Policy Targets")
    report_lines.append("")
    
    # Estatísticas de balanceamento de classes
    report_lines.append(f"### 5.1 Balanceamento de Classes")
    report_lines.append("")
    
    # Amostra mais registros (suporta schema v1 e v2)
    if is_v2:
        cursor.execute("""
            SELECT r.policy_add_blob, r.policy_remove_blob, r.phase, e.resolution
            FROM records r
            JOIN episodes e ON r.episode_id = e.episode_id
            LIMIT 1000
        """)
        policy_samples = cursor.fetchall()
        
        add_ratios = []
        rem_ratios = []
        growth_add_ratios = []
        refinement_add_ratios = []
        
        for policy_add_blob, policy_remove_blob, phase, resolution_json in policy_samples:
            if policy_add_blob or policy_remove_blob:
                resolution = tuple(json.loads(resolution_json))
                add_ch = sparse_decode(*deserialize_sparse(policy_add_blob), resolution) if policy_add_blob else np.zeros(resolution, dtype=np.float32)
                rem_ch = sparse_decode(*deserialize_sparse(policy_remove_blob), resolution) if policy_remove_blob else np.zeros(resolution, dtype=np.float32)
                
                add_ratio = (add_ch > 0.5).sum() / add_ch.size
                rem_ratio = (rem_ch > 0.5).sum() / rem_ch.size
                
                add_ratios.append(add_ratio)
                rem_ratios.append(rem_ratio)
                
                if phase == 'GROWTH':
                    growth_add_ratios.append(add_ratio)
                else:
                    refinement_add_ratios.append(add_ratio)
    else:
        cursor.execute("SELECT policy_blob, phase FROM training_data LIMIT 1000")
        policy_samples = cursor.fetchall()
        
        add_ratios = []
        rem_ratios = []
        growth_add_ratios = []
        refinement_add_ratios = []
        
        for policy_blob, phase in policy_samples:
            policy = deserialize_state(policy_blob)
            add_ch = policy[0]
            rem_ch = policy[1]
            
            add_ratio = (add_ch > 0.5).sum() / add_ch.size
            rem_ratio = (rem_ch > 0.5).sum() / rem_ch.size
            
            add_ratios.append(add_ratio)
            rem_ratios.append(rem_ratio)
            
            if phase == 'GROWTH':
                growth_add_ratios.append(add_ratio)
            else:
                refinement_add_ratios.append(add_ratio)
    
    add_ratios = np.array(add_ratios) if add_ratios else np.array([])
    rem_ratios = np.array(rem_ratios) if rem_ratios else np.array([])
    
    if len(add_ratios) == 0:
        report_lines.append("⚠️ Nenhuma policy encontrada no banco de dados.")
        report_lines.append("")
        # Salva e retorna
        report_content = "\n".join(report_lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_content, encoding='utf-8')
        return str(output_path)
    
    report_lines.append(f"| Métrica | Canal ADD | Canal REMOVE |")
    report_lines.append(f"|---------|-----------|--------------|")
    report_lines.append(f"| Média % Positivos | {add_ratios.mean()*100:.2f}% | {rem_ratios.mean()*100:.2f}% |")
    report_lines.append(f"| Mediana % Positivos | {np.median(add_ratios)*100:.2f}% | {np.median(rem_ratios)*100:.2f}% |")
    report_lines.append(f"| Max % Positivos | {add_ratios.max()*100:.2f}% | {rem_ratios.max()*100:.2f}% |")
    report_lines.append("")
    
    # Recomendação de pos_weight
    avg_add = add_ratios.mean()
    avg_rem = rem_ratios.mean()
    
    if avg_add > 0:
        recommended_add_weight = min(15, max(1, (1 - avg_add) / avg_add))
    else:
        recommended_add_weight = 10
        
    if avg_rem > 0:
        recommended_rem_weight = min(15, max(1, (1 - avg_rem) / avg_rem))
    else:
        recommended_rem_weight = 5
    
    report_lines.append(f"### 5.2 Recomendação de pos_weight (BCEWithLogitsLoss)")
    report_lines.append("")
    report_lines.append(f"| Canal | pos_weight Recomendado |")
    report_lines.append(f"|-------|------------------------|")
    report_lines.append(f"| ADD | **{recommended_add_weight:.1f}** |")
    report_lines.append(f"| REMOVE | **{recommended_rem_weight:.1f}** |")
    report_lines.append("")
    
    # Por fase
    report_lines.append(f"### 5.3 Balanceamento por Fase")
    report_lines.append("")
    
    if growth_add_ratios:
        growth_add = np.array(growth_add_ratios)
        report_lines.append(f"**GROWTH (Fase 1):** ADD médio = {growth_add.mean()*100:.2f}%")
    
    if refinement_add_ratios:
        ref_add = np.array(refinement_add_ratios)
        report_lines.append(f"**REFINEMENT (Fase 2):** ADD médio = {ref_add.mean()*100:.2f}%")
    report_lines.append("")
    
    # ==================== ANÁLISE DE METADADOS ====================
    report_lines.append("## 6. Análise de Metadados")
    report_lines.append("")
    
    # BC Types & Estratégias
    bc_types = {}
    strategies = {}
    if is_v2:
        cursor.execute("""
            SELECT e.bc_type, e.strategy
            FROM records r
            JOIN episodes e ON r.episode_id = e.episode_id
        """)
        for bc_raw, strat_raw in cursor.fetchall():
            bc = bc_raw or 'unknown'
            bc_types[bc] = bc_types.get(bc, 0) + 1
            strat = strat_raw or 'unknown'
            strategies[strat] = strategies.get(strat, 0) + 1
    else:
        cursor.execute("SELECT metadata FROM training_data WHERE metadata IS NOT NULL")
        for row in cursor.fetchall():
            try:
                meta = json.loads(row[0])
                bc = meta.get('bc_type', 'unknown')
                bc_types[bc] = bc_types.get(bc, 0) + 1

                strategy = meta.get('strategy', 'unknown')
                strategies[strategy] = strategies.get(strategy, 0) + 1
            except Exception:
                pass
    
    report_lines.append(f"### 6.1 Boundary Conditions (BC Types)")
    report_lines.append("")
    report_lines.append(f"| BC Type | Contagem | Percentual |")
    report_lines.append(f"|---------|----------|------------|")
    
    for bc, count in bc_types.items():
        pct = count / total_records * 100
        report_lines.append(f"| {bc} | {count} | {pct:.1f}% |")
    report_lines.append("")
    
    report_lines.append(f"### 6.2 Estratégias de Geração")
    report_lines.append("")
    report_lines.append(f"| Estratégia | Contagem | Percentual |")
    report_lines.append(f"|------------|----------|------------|")
    
    for strat, count in strategies.items():
        pct = count / total_records * 100
        report_lines.append(f"| {strat} | {count} | {pct:.1f}% |")
    report_lines.append("")
    
    # ==================== AMOSTRAS ALEATÓRIAS ====================
    report_lines.append("## 7. Amostras Aleatórias")
    report_lines.append("")

    report_lines.append(f"| # | Episode | Step | Phase | Fitness | BC Type |")
    report_lines.append(f"|---|---------|------|-------|---------|---------|")

    if is_v2:
        cursor.execute("""
            SELECT r.episode_id, r.step, r.phase, r.fitness_score, e.bc_type
            FROM records r
            JOIN episodes e ON r.episode_id = e.episode_id
            ORDER BY RANDOM()
            LIMIT 10
        """)
        for i, sample in enumerate(cursor.fetchall(), 1):
            ep_short = sample[0][:8] + "..."
            report_lines.append(
                f"| {i} | `{ep_short}` | {sample[1]} | {sample[2]} | "
                f"{sample[3]:.4f} | {sample[4] or 'N/A'} |"
            )
    else:
        cursor.execute("""
            SELECT episode_id, step, phase, fitness_score, metadata 
            FROM training_data 
            ORDER BY RANDOM() 
            LIMIT 10
        """)
        for i, sample in enumerate(cursor.fetchall(), 1):
            ep_short = sample[0][:8] + "..."
            try:
                meta = json.loads(sample[4]) if sample[4] else {}
                bc = meta.get('bc_type', 'N/A')
            except Exception:
                bc = 'N/A'

            report_lines.append(
                f"| {i} | `{ep_short}` | {sample[1]} | {sample[2]} | "
                f"{sample[3]:.4f} | {bc} |"
            )
    report_lines.append("")

    # ==================== VALIDAÇÃO v3.1 ====================
    report_lines.append("## 8. Validação de Conformidade v3.1")
    report_lines.append("")

    checks = []

    # Check 1: 7 canais
    if state_sample is not None:
        checks.append(("State tem 7 canais", state_sample.shape[0] == 7))

    # Check 2: Policy tem 2 canais
    if policy_sample is not None:
        checks.append(("Policy tem 2 canais", policy_sample.shape[0] == 2))

    # Check 3: Ambas as fases presentes
    checks.append(("Fase GROWTH presente", phase_counts.get('GROWTH', 0) > 0))
    checks.append(("Fase REFINEMENT presente", phase_counts.get('REFINEMENT', 0) > 0))

    # Check 4: Fitness score em [-1, 1]
    checks.append(("Fitness score em [-1, 1]", scores_array.min() >= -1 and scores_array.max() <= 1))

    # Check 5: Metadados com bc_type
    checks.append(("Metadados bc_type presentes", len(bc_types) > 0))

    # Check 6: dtype float32 para policy
    if policy_sample is not None:
        checks.append(("Policy dtype é float32", policy_sample.dtype == np.float32))
    
    report_lines.append(f"| Check | Status |")
    report_lines.append(f"|-------|--------|")
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        report_lines.append(f"| {check_name} | {status} |")
    
    report_lines.append("")
    
    if all_passed:
        report_lines.append("### ✅ Todos os checks de conformidade v3.1 passaram!")
    else:
        report_lines.append("### ⚠️ Alguns checks falharam. Verificar implementação.")
    report_lines.append("")
    
    # ==================== FOOTER ====================
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*Relatório gerado automaticamente pelo teste de integração AlphaBuilder v3.1*")
    
    conn.close()
    
    # Salva o relatório
    report_content = "\n".join(report_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding='utf-8')
    
    return str(output_path)

