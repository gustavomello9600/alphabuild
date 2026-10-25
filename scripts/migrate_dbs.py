
import sys
from pathlib import Path
import sqlite3

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from alphabuilder.src.logic.selfplay.storage import migrate_db

DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    print(f"Searching for databases in {DATA_DIR}...")
    db_files = list(DATA_DIR.glob("*.db"))
    
    if not db_files:
        print("No databases found.")
        return

    for db_path in db_files:
        print(f"Migrating {db_path.name}...")
        try:
            conn = sqlite3.connect(db_path)
            migrate_db(conn)
            conn.close()
        except Exception as e:
            print(f"Error migrating {db_path.name}: {e}")

if __name__ == "__main__":
    main()
