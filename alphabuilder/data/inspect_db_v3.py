import sqlite3
import numpy as np
import io
import os

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

# Correct path: project root data directory
base_dir = '/home/Gustavo/projects/alphabuild/data'
db_files = [
    os.path.join(base_dir, 'training_data.db'),
    os.path.join(base_dir, 'harvest_part1.db'),
    os.path.join(base_dir, 'harvest_part2.db'),
    os.path.join(base_dir, 'local_test_harvest.db')
]

for db_path in db_files:
    print(f"\n\n==========================================")
    print(f"Inspecting: {db_path}")
    if not os.path.exists(db_path):
        print("File does not exist.")
        continue
        
    print(f"File size: {os.path.getsize(db_path)} bytes")

    try:
        conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables found: {tables}")

        for table_name in tables:
            table = table_name[0]
            print(f"\n--- Table: {table} ---")
            
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            print("Schema:")
            for col in columns:
                print(col)

            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Total rows: {count}")

            if count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                row = cursor.fetchone()
                print("Sample row (first 1):")
                for i, val in enumerate(row):
                    col_name = columns[i][1]
                    val_type = type(val)
                    
                    if isinstance(val, bytes):
                        try:
                            arr = convert_array(val)
                            print(f"  {col_name}: {val_type} -> Shape: {arr.shape}, Dtype: {arr.dtype}")
                            if np.isnan(arr).any() or np.isinf(arr).any():
                                 print(f"    WARNING: Contains NaN or Inf!")
                            print(f"    Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}")
                        except:
                            print(f"  {col_name}: {val_type} (Binary, size={len(val)})")
                    else:
                        print(f"  {col_name}: {val_type} -> {val}")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")
