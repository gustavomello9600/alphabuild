"""
Training Logger for AlphaBuilder v3.1.

Provides CSV logging and optional TensorBoard integration.
Designed for minimal overhead during GPU training.
"""
import csv
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class TrainingLogger:
    """
    Efficient training logger with CSV and optional TensorBoard support.
    
    Designed to minimize CPU overhead during training:
    - Buffered CSV writes
    - Lazy TensorBoard initialization
    - Batch metric accumulation
    """
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        buffer_size: int = 10
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
            use_tensorboard: Whether to use TensorBoard logging
            buffer_size: Number of log entries to buffer before writing
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / "training_log.csv"
        self.plot_path = self.log_dir / "loss_plot.png"
        self.start_time = time.time()
        
        # CSV buffer for efficient writes
        self.buffer = []
        self.buffer_size = buffer_size
        
        # TensorBoard writer (lazy initialization)
        self.writer: Optional[SummaryWriter] = None
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        
        # Initialize CSV
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'step', 'epoch', 
                    'train_loss', 'val_loss',
                    'policy_loss', 'value_loss',
                    'learning_rate', 'elapsed_time', 'samples_per_sec'
                ])
    
    def _get_writer(self) -> Optional[SummaryWriter]:
        """Lazy initialization of TensorBoard writer."""
        if self.use_tensorboard and self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        return self.writer
    
    def log(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        samples_per_sec: Optional[float] = None
    ):
        """
        Log a training step to CSV (buffered).
        
        Args:
            step: Global step number
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Optional validation loss
            policy_loss: Optional policy loss component
            value_loss: Optional value loss component
            learning_rate: Current learning rate
            samples_per_sec: Training throughput
        """
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.buffer.append([
            timestamp, step, epoch, 
            train_loss, val_loss or '',
            policy_loss or '', value_loss or '',
            learning_rate or '', elapsed, samples_per_sec or ''
        ])
        
        # Flush buffer if full
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffered entries to CSV."""
        if not self.buffer:
            return
            
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)
        self.buffer.clear()
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "train"
    ):
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Global step number
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        writer = self._get_writer()
        if writer is None:
            return
            
        for key, value in metrics.items():
            writer.add_scalar(f"{prefix}/{key}", value, step)
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        epoch_time: float,
        samples_per_sec: float
    ):
        """
        Log complete epoch summary.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
            learning_rate: Current learning rate
            epoch_time: Time for this epoch
            samples_per_sec: Training throughput
        """
        # Log to CSV
        self.log(
            step=epoch,
            epoch=epoch,
            train_loss=train_metrics.get('loss', 0),
            val_loss=val_metrics.get('loss'),
            policy_loss=train_metrics.get('policy_loss'),
            value_loss=train_metrics.get('value_loss'),
            learning_rate=learning_rate,
            samples_per_sec=samples_per_sec
        )
        
        # Log to TensorBoard
        writer = self._get_writer()
        if writer:
            writer.add_scalar('epoch/train_loss', train_metrics.get('loss', 0), epoch)
            writer.add_scalar('epoch/val_loss', val_metrics.get('loss', 0), epoch)
            writer.add_scalar('epoch/policy_loss', train_metrics.get('policy_loss', 0), epoch)
            writer.add_scalar('epoch/value_loss', train_metrics.get('value_loss', 0), epoch)
            writer.add_scalar('epoch/learning_rate', learning_rate, epoch)
            writer.add_scalar('epoch/time_seconds', epoch_time, epoch)
            writer.add_scalar('epoch/samples_per_sec', samples_per_sec, epoch)
    
    def log_histogram(self, name: str, values: Any, step: int):
        """Log histogram to TensorBoard (for weight/gradient analysis)."""
        writer = self._get_writer()
        if writer:
            writer.add_histogram(name, values, step)
    
    def plot(self):
        """Generate and save loss plots."""
        # Flush any remaining buffer
        self._flush_buffer()
        
        steps = []
        train_losses = []
        val_losses = []
        times = []
        
        if not self.csv_path.exists():
            return
            
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['step']))
                train_losses.append(float(row['train_loss']))
                if row['val_loss']:
                    val_losses.append(float(row['val_loss']))
                times.append(float(row['elapsed_time']))
                
        if not steps:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Training loss vs Step
        axes[0].plot(steps, train_losses, label='Training Loss', color='blue')
        if val_losses and len(val_losses) == len(steps):
            axes[0].plot(steps, val_losses, label='Validation Loss', color='orange')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss vs Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss vs Time
        axes[1].plot(times, train_losses, label='Training Loss', color='blue')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss vs Time')
        axes[1].grid(True, alpha=0.3)
        
        # Log-scale loss
        axes[2].semilogy(steps, train_losses, label='Training Loss', color='blue')
        if val_losses and len(val_losses) == len(steps):
            axes[2].semilogy(steps, val_losses, label='Validation Loss', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss (log scale)')
        axes[2].set_title('Loss (Log Scale)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()
    
    def close(self):
        """Clean up resources."""
        self._flush_buffer()
        if self.writer:
            self.writer.close()
            self.writer = None
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.close()
        except:
            pass

