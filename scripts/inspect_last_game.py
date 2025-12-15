
import sqlite3
import sys
import os

DB_PATH = "data/selfplay_games.db"

def inspect_last_game():
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get last game
    cursor.execute("SELECT game_id, created_at, total_steps FROM games ORDER BY created_at DESC LIMIT 1")
    row = cursor.fetchone()
    
    if not row:
        print("No games found in database.")
        return

    game_id, created_at, total_steps = row
    print(f"Last Game ID: {game_id}")
    print(f"Created At: {created_at}")
    print(f"Total Steps: {total_steps}")
    
    # Get max displacement from the last recorded step (which might be in refinement phase)
    # We look for the step with the highest step number
    cursor.execute("""
        SELECT step, phase, max_displacement, compliance_fem, value, is_connected
        FROM game_steps 
        WHERE game_id = ? 
        ORDER BY step DESC 
        LIMIT 1
    """, (game_id,))
    
    step_row = cursor.fetchone()
    
    if step_row:
        step, phase, max_disp, comp, val, connected = step_row
        print("-" * 40)
        print(f"Last Step: {step} ({phase})")
        print(f"Is Connected: {bool(connected)}")
        print(f"Value (Reward): {val}")
        print(f"Compliance FEM: {comp}")
        print(f"Max Displacement: {max_disp}")
        
        if max_disp is not None:
             print(f"\nScaled (Scale=64.0): {max_disp/64.0:.4f} meters")
    else:
        print("No steps found for this game.")

    conn.close()

if __name__ == "__main__":
    inspect_last_game()
