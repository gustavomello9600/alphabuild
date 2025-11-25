import os
import tensorflow as tf
import numpy as np
from alphabuilder.src.neural.data_loader import load_training_data
from alphabuilder.src.neural.model_arch import build_3d_vit
from alphabuilder.src.neural.trainer import train_step, save_model

# Configuration
DB_PATH = "data/training_data.db"
MODEL_PATH = "models/vit_latest.keras"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

def main():
    print(f"Starting AlphaBuilder Neural Vision Training...")
    print(f"Database: {DB_PATH}")
    
    # 1. Load Data
    print("Loading training data logic initialized (generator).")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return
        
    # batches = load_training_data(DB_PATH, batch_size=BATCH_SIZE)
    # if not batches: ...
    # We defer loading to the loop

    
    # 2. Build Model
    print("Building 3D ViT Model...")
    model = build_3d_vit()
    
    # 3. Setup Training
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # 4. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        epoch_loss = []
        epoch_mae = []
        
        # Re-create generator for each epoch since it's exhausted
        batches = load_training_data(DB_PATH, batch_size=BATCH_SIZE)
        
        for batch_idx, batch in enumerate(batches):
            metrics = train_step(model, batch.inputs.tensor, batch.targets, optimizer, loss_fn)
            epoch_loss.append(metrics["loss"])
            epoch_mae.append(metrics["mae"])
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {metrics['loss']:.4f} | MAE: {metrics['mae']:.4f}", end="\r")
                
        if not epoch_loss:
            print(f"Epoch {epoch+1}/{EPOCHS} | No data found.")
            continue

        avg_loss = np.mean(epoch_loss)
        avg_mae = np.mean(epoch_mae)
        print(f"\nEpoch {epoch+1}/{EPOCHS} Completed | Avg Loss: {avg_loss:.4f} | Avg MAE: {avg_mae:.4f}")
        
    # 5. Save Model
    print("Saving model...")
    save_model(model, MODEL_PATH)
    print("Training Complete.")

if __name__ == "__main__":
    main()
