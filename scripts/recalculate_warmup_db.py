#!/usr/bin/env python3
"""
Recalculate fitness scores in warmup database using the new reward formula (v3.2).

Since the original database (v2 schema) in warmup_data_kaggle.db does not store
'compliance' or 'volume_fraction' metadata explicitly in the records table,
we must reverse-engineer the compliance from the stored 'fitness_score' using
the legacy formula constants.

Legacy Formula:
    S_raw = -ln(C + eps) - alpha * V
    S_fem = tanh((S_raw - mu) / sigma)

    Where:
    mu = -6.65
    sigma = 2.0
    alpha = 12.0

Reverse Engineering:
    z = artanh(S_fem)  (clamped to avoid infinity)
    S_raw = z * sigma + mu
    ln(C) = -S_raw - alpha * V
    C = exp(ln(C))

New Formula (v3.2):
    compliance_score = 0.80 - 0.16 * (log10(C) - 1)
    volume_bonus = (0.10 - V) * 2.0
    reward = compliance_score + volume_bonus
"""

import sqlite3
import numpy as np
import zlib
import pickle
import io
import sys
import os
from tqdm import tqdm
from pathlib import Path

# Add project root to path to import dataset utils if needed
# But we'll implement minimal deserialization here to be standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Constants ---

# Legacy (Old) Constants
LEGACY_MU = -6.65
LEGACY_SIGMA = 2.0
LEGACY_ALPHA = 12.0
LEGACY_EPSILON = 1e-9

# New (v3.2) Constants
COMPLIANCE_BASE = 0.80
COMPLIANCE_SLOPE = 0.16
COMPLIANCE_MIN = -0.50
COMPLIANCE_MAX = 0.85

VOLUME_REFERENCE = 0.10
VOLUME_SENSITIVITY = 2.0
VOLUME_BONUS_MAX = 0.30
VOLUME_PENALTY_MAX = 0.60

# Database Paths
SOURCE_DB = Path("/home/usuario/alphabuild/data/warmup_data_kaggle.db")
DEST_DB = Path("/home/usuario/alphabuild/data/warmup_data_v3_2.db")

# --- Helper Functions ---

def deserialize_array(blob: bytes) -> np.ndarray:
    """Deserialize numpy array from compressed bytes."""
    decompressed = zlib.decompress(blob)
    
    # Try np.load format first (starts with numpy magic bytes)
    if decompressed[:6] == b'\x93NUMPY':
        buffer = io.BytesIO(decompressed)
        return np.load(buffer, allow_pickle=False)
    
    # Fall back to pickle format (legacy)
    return pickle.loads(decompressed)

def calculate_new_reward(compliance: float, vol_frac: float) -> float:
    """Calculate reward using the NEW additive formula."""
    # Compliance score
    if compliance <= 0:
        compliance = 1.0 # Protect against invalid
    
    log_c = np.log10(max(compliance, 1.0))
    c_score = COMPLIANCE_BASE - COMPLIANCE_SLOPE * (log_c - 1.0)
    c_score = float(np.clip(c_score, COMPLIANCE_MIN, COMPLIANCE_MAX))
    
    # Volume bonus
    v_bonus = (VOLUME_REFERENCE - vol_frac) * VOLUME_SENSITIVITY
    v_bonus = float(np.clip(v_bonus, -VOLUME_PENALTY_MAX, VOLUME_BONUS_MAX))
    
    return float(np.clip(c_score + v_bonus, -1.0, 1.0))

def reverse_engineer_compliance(fitness_score: float, vol_frac: float) -> float:
    """Back-calculate compliance C from legacy fitness score and volume."""
    # Clamp fitness to avoid artanh singularity
    fitness = np.clip(fitness_score, -0.999999, 0.999999)
    
    # S_fem = tanh((S_raw - mu) / sigma)
    # z = (S_raw - mu) / sigma = artanh(S_fem)
    z = np.arctanh(fitness)
    
    # S_raw = z * sigma + mu
    s_raw = z * LEGACY_SIGMA + LEGACY_MU
    
    # S_raw = -ln(C) - alpha * V
    # ln(C) = -S_raw - alpha * V
    log_c_natural = -s_raw - (LEGACY_ALPHA * vol_frac)
    
    # C = exp(ln(C))
    c = np.exp(log_c_natural)
    
    return float(c)

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
    
    # 1. Copy Schema
    print("Creating schema...")
    src_cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='episodes'")
    schema_episodes = src_cur.fetchone()[0]
    dest_cur.execute(schema_episodes)
    
    src_cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='records'")
    schema_records = src_cur.fetchone()[0]
    dest_cur.execute(schema_records)
    
    # 2. Copy Episodes (No change)
    print("Copying episodes...")
    src_cur.execute("SELECT * FROM episodes")
    episodes = src_cur.fetchall()
    dest_cur.executemany("INSERT INTO episodes VALUES (?, ?, ?, ?)", episodes)
    dest_conn.commit()
    print(f"Copied {len(episodes)} episodes.")
    
    # 3. Process records
    print("Processing records...")
    src_cur.execute("SELECT id, episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob, fitness_score, is_final_step, is_connected FROM records")
    
    # Using fetchmany to batch process if needed, but simple loop is fine for script
    # For progress bar, we need total count
    src_cur.execute("SELECT COUNT(*) FROM records")
    total_records = src_cur.fetchone()[0]
    src_cur.execute("SELECT id, episode_id, step, phase, density_blob, policy_add_blob, policy_remove_blob, fitness_score, is_final_step, is_connected FROM records")

    buffer = []
    batch_size = 1000
    
    for row in tqdm(src_cur, total=total_records, desc="Recalculating"):
        rec_id, ep_id, step, phase, density_blob, p_add, p_rem, old_fitness, is_final, is_conn = row
        
        # Deserialize density to get volume
        density = deserialize_array(density_blob)
        vol_frac = float(density.mean())
        
        # Reverse engineer compliance
        compliance = reverse_engineer_compliance(old_fitness, vol_frac)
        
        # Calculate new reward
        new_reward = calculate_new_reward(compliance, vol_frac)
        
        buffer.append((rec_id, ep_id, step, phase, density_blob, p_add, p_rem, new_reward, is_final, is_conn))
        
        if len(buffer) >= batch_size:
            dest_cur.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", buffer)
            buffer = []
            
    if buffer:
        dest_cur.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", buffer)
    
    dest_conn.commit()
    
    print("Optimizing database...")
    dest_conn.execute("VACUUM")
    
    src_conn.close()
    dest_conn.close()
    
    print("Done! verification:")
    
    # Verification sample
    check_conn = sqlite3.connect(DEST_DB)
    c = check_conn.cursor()
    c.execute("SELECT fitness_score FROM records LIMIT 5")
    print("First 5 new scores:", [r[0] for r in c.fetchall()])
    
    # Check if we have negative scores (should be possible now)
    c.execute("SELECT COUNT(*) FROM records WHERE fitness_score < 0")
    neg_count = c.fetchone()[0]
    print(f"Negative scores count: {neg_count}")
    
    check_conn.close()

if __name__ == "__main__":
    main()
