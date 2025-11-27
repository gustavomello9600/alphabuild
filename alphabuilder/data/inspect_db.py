import sqlite3
import numpy as np
import io

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Register numpy adaptors just in case, though we are reading
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

db_path = '/home/Gustavo/projects/alphabuild/alphabuilder/data/training_data.db'

try:
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables found: {tables}")

    for table_name in tables:
        table = table_name[0]
        print(f"\n--- Table: {table} ---")
        
        # Get schema
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        print("Schema:")
        for col in columns:
            print(col)

        # Count rows
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"Total rows: {count}")

        if count > 0:
            # Fetch one row
            cursor.execute(f"SELECT * FROM {table} LIMIT 1")
            row = cursor.fetchone()
            print("Sample row (first 1):")
            for i, val in enumerate(row):
                col_name = columns[i][1]
                val_type = type(val)
                val_repr = str(val)
                if len(val_repr) > 100:
                    val_repr = val_repr[:100] + "..."
                
                # Try to interpret blobs as numpy arrays if possible
                if isinstance(val, bytes):
                    try:
                        arr = convert_array(val)
                        print(f"  {col_name}: {val_type} -> Shape: {arr.shape}, Dtype: {arr.dtype}")
                        # Check for NaN or Inf
                        if np.isnan(arr).any() or np.isinf(arr).any():
                             print(f"    WARNING: Contains NaN or Inf!")
                        # Basic stats
                        print(f"    Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}")
                    except Exception as e:
                        print(f"  {col_name}: {val_type} (Binary, not a numpy array or load failed: {e})")
                else:
                    print(f"  {col_name}: {val_type} -> {val_repr}")

    conn.close()

except Exception as e:
    print(f"Error: {e}")
