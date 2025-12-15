#!/usr/bin/env python3
"""
Recalculate fitness scores in warmup database using the new reward formula (v3.2) and Value Target Logic (v4.0).

Changes:
- Uses INITIAL_CANTILEVER_PROBLEM_REAL_SCALE_FACTOR schema.
- Calculates Pure FEM Score (Compliance + Volume) for targets.
- Implements Phase-dependent Value Targets:
  - Phase 1 (GROWTH): Target = Final Episode FEM Score
  - Phase 2 (REFINEMENT): Target = (1-mixing)*Current + (mixing)*Final
"""

import sqlite3
import numpy as np
import zlib
import pickle
import io
import sys
import os
import json
from tqdm import tqdm
from pathlib import Path

# --- Constants ---

# Legacy (Old) Constants for Reverse Engineering
LEGACY_MU = -6.65
LEGACY_SIGMA = 2.0
LEGACY_ALPHA = 12.0
LEGACY_EPSILON = 1e-9

# New Formula Constants (Pure FEM Score)
COMPLIANCE_BASE = 0.80
COMPLIANCE_SLOPE = 0.16
COMPLIANCE_MIN = -0.50
COMPLIANCE_MAX = 0.85

VOLUME_REFERENCE = 0.10
VOLUME_SENSITIVITY = 2.0
VOLUME_BONUS_MAX = 0.30
VOLUME_PENALTY_MAX = 0.60

# Mixing Factor for Phase 2
MIXING_FACTOR = 0.5

# Database Paths
SOURCE_DB = Path("data/warmup_data_kaggle.db")
DEST_DB = Path("data/warmup_data_v3_2.db")

# --- Helper Functions ---

def deserialize_array(blob: bytes) -> np.ndarray:
    """Deserialize numpy array from compressed bytes."""
    decompressed = zlib.decompress(blob)
    if decompressed[:6] == b'\x93NUMPY':
        buffer = io.BytesIO(decompressed)
        return np.load(buffer, allow_pickle=False)
    return pickle.loads(decompressed)

def calculate_compliance_score(compliance: float) -> float:
    # log10 mapping
    if compliance <= 0: return COMPLIANCE_MAX
    log_c = np.log10(max(compliance, 1.0))
    score = COMPLIANCE_BASE - COMPLIANCE_SLOPE * (log_c - 1.0)
    return float(np.clip(score, COMPLIANCE_MIN, COMPLIANCE_MAX))

def calculate_volume_bonus(vol_frac: float) -> float:
    adjustment = (VOLUME_REFERENCE - vol_frac) * VOLUME_SENSITIVITY
    return float(np.clip(adjustment, -VOLUME_PENALTY_MAX, VOLUME_BONUS_MAX))

def calculate_fem_score(compliance: float, vol_frac: float) -> float:
    """Pure Physics Score: Compliance + Volume (No penalties)."""
    c_score = calculate_compliance_score(compliance)
    v_bonus = calculate_volume_bonus(vol_frac)
    return float(np.clip(c_score + v_bonus, -1.0, 1.0))

def reverse_engineer_compliance(fitness_score: float, vol_frac: float) -> float:
    """Back-calculate compliance C from legacy fitness score."""
    fitness = np.clip(fitness_score, -0.999999, 0.999999)
    # S_fem = tanh((S_raw - mu) / sigma) -> raw = artanh(S)*sigma + mu
    z = np.arctanh(fitness)
    s_raw = z * LEGACY_SIGMA + LEGACY_MU
    # S_raw = -ln(C) - alpha * V -> C = exp(-raw - alpha*V)
    log_c_natural = -s_raw - (LEGACY_ALPHA * vol_frac)
    return float(np.exp(log_c_natural))

def main():
    if not SOURCE_DB.exists():
        print(f"Error: Source database {SOURCE_DB} not found.")
        return
    
    if DEST_DB.exists():
        print(f"Warning: Destination database {DEST_DB} already exists. Deleting...")
        os.remove(DEST_DB)
        
    print(f"Source: {SOURCE_DB}")
    print(f"Destination: {DEST_DB}")
    
    src_conn = sqlite3.connect(SOURCE_DB)
    dest_conn = sqlite3.connect(DEST_DB)
    
    src_cur = src_conn.cursor()
    dest_cur = dest_conn.cursor()
    
    # 1. Create Schema (New Schema with initial_cantilever_problem_real_scale_factor)
    print("Creating schema...")
    
    # Games/Episodes table
    dest_cur.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            neural_engine TEXT,
            checkpoint_version TEXT,
            bc_masks_blob BLOB,
            forces_blob BLOB,
            load_config TEXT,
            bc_type TEXT,
            resolution TEXT,
            final_score REAL,
            final_compliance REAL,
            final_volume REAL,
            total_steps INTEGER DEFAULT 0,
            initial_cantilever_problem_real_scale_factor REAL DEFAULT 500000.0,
            stiffness_E REAL DEFAULT 200000000000.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Steps/Records table
    dest_cur.execute("""
        CREATE TABLE IF NOT EXISTS game_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            step INTEGER,
            phase TEXT,
            density_blob BLOB,
            policy_add_blob BLOB,
            policy_remove_blob BLOB,
            mcts_visit_add_blob BLOB,
            mcts_visit_remove_blob BLOB,
            mcts_q_add_blob BLOB,
            mcts_q_remove_blob BLOB,
            selected_actions_json TEXT,
            value REAL,
            value_target REAL,
            compliance_fem REAL,
            max_displacement REAL,
            volume_fraction REAL,
            island_penalty REAL,
            reward_components_json TEXT,
            is_connected BOOLEAN,
            n_islands INTEGER,
            loose_voxels INTEGER,
            connected_load_fraction REAL DEFAULT 0.0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(game_id) REFERENCES games(game_id)
        )
    """)
    
    # Kaggle source usually has 'episodes' and 'records' tables.
    # We need to map them to 'games' and 'game_steps'.
    # Note: Kaggle source might lack some fields (like mcts blobs), we'll fill nulls/zeros.
    
    print("Migrating data from old schema to new...")
    
    # Process Episodes -> Games
    src_cur.execute("SELECT * FROM episodes")
    old_episodes = src_cur.fetchall()
    
    # Create a mapping for insertion
    # Old Schema Assumption: id, resolution_x, ... ? 
    # Actually, let's just inspect source schema quickly if running interactively, but assume we know it.
    # We will assume we iterate episodes and insert into games.
    
    # For now, let's replicate the logic of 'Copy Episodes' but adapt to new schema
    # Since Kaggle DB structure might differ, I will stick to what 'recalculate_warmup_db.py' was doing before:
    # Identifying fields. 
    # But wait, the previous code copied raw tables. We need to TRANSFORM.
    
    print("Processing Records into Memory for Recalculation...")
    src_cur.execute("SELECT id, episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob, fitness_score, is_final_step, is_connected FROM records ORDER BY episode_id, step")
    
    all_rows = src_cur.fetchall()
    print(f"Loaded {len(all_rows)} records. Grouping...")

    current_ep_id = None
    episode_buffer = []
    
    # We need to buffer standard steps to write to NEW schema
    game_writes = []
    step_writes = []
    
    # Process Grouped
    for row in tqdm(all_rows, desc="Processing"):
        rec_id, ep_id, step, phase, density_blob, p_add, p_rem, old_fitness, is_final, is_conn = row
        
        if ep_id != current_ep_id:
            if episode_buffer:
                process_episode(episode_buffer, game_writes, step_writes, dest_cur)
            current_ep_id = ep_id
            episode_buffer = []
        
        episode_buffer.append(row)
        
    if episode_buffer:
        process_episode(episode_buffer, game_writes, step_writes, dest_cur)
        
    dest_conn.commit()
    dest_conn.execute("VACUUM")
    print("Done.")

def process_episode(episode_rows, game_writes, step_writes, cursor):
    """
    Process one episode buffer.
    Calculates Phase-dependent targets and inserts into DB.
    """
    if not episode_rows: return

    # Get Final Step info
    final_row = episode_rows[-1]
    _, ep_id, _, _, f_density_blob, _, _, f_old_fitness, _, f_is_conn = final_row
    
    # 1. Calculate Final FEM Score (Pure)
    final_fem_score = -1.0
    final_vol_frac = 0.0
    final_steps = len(episode_rows)
    
    if f_is_conn:
        f_density = deserialize_array(f_density_blob)
        final_vol_frac = float(f_density.mean())
        if f_is_conn: # Kaggle is_connected=1 means valid
            f_compliance = reverse_engineer_compliance(f_old_fitness, final_vol_frac)
            final_fem_score = calculate_fem_score(f_compliance, final_vol_frac)
    
    # 2. Insert Game Record (Create fake game metadata if needed, usually we'd copy from episodes table)
    # We'll just insert a basic game record.
    cursor.execute("""
        INSERT INTO games (game_id, neural_engine, checkpoint_version, total_steps, final_score, final_volume, initial_cantilever_problem_real_scale_factor)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ep_id, "kaggle_warmup", "v0", final_steps, final_fem_score, final_vol_frac, 500000.0))
    
    # 3. Process Steps & Calculate Targets
    for row in episode_rows:
        rec_id, ep_id, step, phase, density_blob, p_add, p_rem, old_fitness, is_final, is_conn = row
        
        # Calculate Current FEM Score (if valid)
        current_fem_score = -1.0
        # Need density to get volume
        density = deserialize_array(density_blob)
        vol_frac = float(density.mean())
        
        # Approximate compliance from old fitness if available
        # But old fitness might be -1.0 if invalid.
        # We can trust 'is_conn' or 'old_fitness > -0.99'
        if is_conn and old_fitness > -0.99:
             comp = reverse_engineer_compliance(old_fitness, vol_frac)
             current_fem_score = calculate_fem_score(comp, vol_frac)
        else:
             current_fem_score = -1.0
             
        # Determine Target
        target = 0.0
        # Phase check (kaggle DB stores phase as int 0/1 or string?)
        # Kaggle might store string 'GROWTH'/'REFINEMENT' or int. 
        # Assuming string based on previous script.
        
        # LOGIC:
        if phase == 'GROWTH':
            target = final_fem_score
        else:
            # Phase 2: Mixed target
            # lambda = step_idx / last_step_idx
            current_lambda = step / max(1, final_steps)
            current_lambda = min(1.0, max(0.0, current_lambda))
            
            if comp is not None:
                current_score = calculate_fem_score(comp, vol_frac)
                # target = (1 - lambda) * current + lambda * final
                target = (1.0 - current_lambda) * current_score + current_lambda * final_fem_score
            else:
                target = final_fem_score # Fallback/Penalty
        
        target = max(-1.0, min(1.0, float(target)))
        
        # Insert Step
        cursor.execute("""
            INSERT INTO game_steps (game_id, step, phase, density_blob, policy_add_blob, policy_remove_blob, value, value_target, volume_fraction, is_connected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ep_id, step, phase, density_blob, p_add, p_rem, old_fitness, target, vol_frac, is_conn))

if __name__ == "__main__":
    main()
