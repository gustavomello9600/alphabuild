import tensorflow as tf
import argparse
import time
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alphabuilder.src.neural.model import create_vit_regressor
from alphabuilder.src.neural.dataset import create_dataset, get_dataset_shape
from alphabuilder.src.neural.logger import TrainingLogger

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaBuilder ViT Training")
    parser.add_argument("--db-path", type=str, default="data/training_data.db")
    parser.add_argument("--log-dir", type=str, default="logs/vit_run_1")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    # Resolution argument is now optional/fallback
    parser.add_argument("--resolution", type=str, default=None, help="Force resolution HxW (e.g. 32x64). If None, auto-detects.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Auto-detect shape
    detected_shape = get_dataset_shape(args.db_path)
    
    if args.resolution:
        # User passes WxH (e.g. 64x32)
        # Model expects HxW (e.g. 32, 64)
        w, h = map(int, args.resolution.split('x'))
        input_shape = (h, w, 3)
        print(f"Forcing resolution: {input_shape} (H={h}, W={w})")
    elif detected_shape:
        input_shape = detected_shape
        print(f"Auto-detected input shape from DB: {input_shape}")
    else:
        # Fallback default
        print("Warning: Could not detect shape from DB (empty?). Using default 32x64.")
        input_shape = (32, 64, 3)
    
    # Create Model
    # Use flexible input shape to allow variable resolutions/padding
    # The model will adapt to whatever shape padded_batch produces
    model = create_vit_regressor(input_shape=(None, None, 3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Logger
    logger = TrainingLogger(args.log_dir)
    
    # Checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "vit_model.keras"
    
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}")
        model = tf.keras.models.load_model(ckpt_path)
    
    print(f"Starting training on {args.db_path}")
    print(f"Input shape: {input_shape}")
    
    step = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Re-create dataset each epoch to pick up new data from DB
        dataset = create_dataset(args.db_path, batch_size=args.batch_size)
        
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        # Training Loop
        for batch, (x, y) in enumerate(dataset):
            loss = model.train_on_batch(x, y)
            # train_on_batch returns scalar loss if no metrics, or list [loss, metric...]
            if isinstance(loss, list):
                loss_val = loss[0]
            else:
                loss_val = loss
                
            epoch_loss_avg.update_state(loss_val)
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss_val:.4f}", end='\r')
                logger.log(step, epoch, loss_val)
                
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss_avg.result():.4f}")
        
        # Save checkpoint and plot
        model.save(ckpt_path)
        logger.plot()
        
        # Wait a bit if data generation is slow? 
        # No, just proceed. If DB is small, epochs are fast.

if __name__ == "__main__":
    main()
