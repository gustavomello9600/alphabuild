import sqlite3
import pickle
import zlib
import numpy as np

db_path = '/home/usuario/alphabuild/data/training_data.db'
episode_id = '0220f3d9-daa4-49d1-9d90-23430006d08a'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT step, displacement_blob FROM records WHERE episode_id = ? AND phase = 'REFINEMENT' LIMIT 5", (episode_id,))
rows = cursor.fetchall()

print(f"Found {len(rows)} rows for REFINEMENT phase.")

for step, blob in rows:
    if blob is None:
        print(f"Step {step}: Blob is None")
        continue

    try:
        # Try expected format: zlib -> pickle -> numpy
        decoded = pickle.loads(zlib.decompress(blob))
        
        if isinstance(decoded, np.ndarray):
            print(f"Step {step}: Decoded as Numpy Array. Shape: {decoded.shape}, Dtype: {decoded.dtype}")
            print(f"  Min: {decoded.min()}, Max: {decoded.max()}, Mean: {decoded.mean()}")
        else:
            print(f"Step {step}: Decoded as {type(decoded)}")

    except Exception as e:
        print(f"Step {step}: Error unpickling: {e}")
        # Try raw zlib
        try:
           raw = zlib.decompress(blob)
           print(f"Step {step}: Raw zlib decompressed size: {len(raw)}")
        except:
           pass

conn.close()
