import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

class TrainingLogger:
    def __init__(self, log_dir: str, filename: str, headers: List[str]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        self.headers = ["timestamp"] + headers
        
        # Initialize CSV if not exists
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                
    def log(self, metrics: Dict[str, Any]):
        """Log metrics to CSV and Console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Console Output
        log_str = f"[{timestamp}] "
        for k, v in metrics.items():
            if isinstance(v, float):
                log_str += f"{k}={v:.4f} "
            else:
                log_str += f"{k}={v} "
        print(log_str)
        
        # CSV Output
        row = [timestamp]
        for h in self.headers[1:]:
            row.append(metrics.get(h, ""))
            
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
