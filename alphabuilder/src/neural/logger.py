import csv
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "training_log.csv"
        self.plot_path = self.log_dir / "loss_plot.png"
        self.start_time = time.time()
        
        # Initialize CSV
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'step', 'epoch', 'loss', 'val_loss', 'elapsed_time'])
                
    def log(self, step, epoch, loss, val_loss=None):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, step, epoch, loss, val_loss if val_loss else '', elapsed])
            
    def plot(self):
        """Generate and save loss plots."""
        steps = []
        losses = []
        times = []
        
        if not self.csv_path.exists():
            return
            
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['step']))
                losses.append(float(row['loss']))
                times.append(float(row['elapsed_time']))
                
        if not steps:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss vs Step
        ax1.plot(steps, losses, label='Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Loss vs Training Steps')
        ax1.grid(True)
        
        # Loss vs Time
        ax2.plot(times, losses, label='Training Loss', color='orange')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Loss vs Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()
