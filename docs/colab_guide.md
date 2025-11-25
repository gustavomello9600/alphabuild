# AlphaBuilder on Google Colab Guide

This guide explains how to run the AlphaBuilder project on Google Colab, utilizing free GPUs for training and CPUs for data generation in parallel.

## 1. Setup Strategy

We will use **Google Drive** as the persistent storage layer.
- **Notebook A (CPU):** Runs `run_data_harvest.py` to generate FEM data and save to `drive/MyDrive/AlphaBuilder/data/training_data.db`.
- **Notebook B (GPU):** Runs `src/neural/train.py` to read from the same database and train the ViT model, saving logs to `drive/MyDrive/AlphaBuilder/logs`.

## 2. Environment Setup (Common)

In BOTH notebooks, you need to install the dependencies. Since FEniCSx requires complex C++ libraries, we use **fem-on-colab** or **Conda**.

### Recommended Setup Cell
Run this at the top of your Colab notebooks:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone Repository (if not already present)
import os
if not os.path.exists('/content/drive/MyDrive/AlphaBuilder'):
    !git clone https://github.com/gustavomello9600/alphabuild.git /content/drive/MyDrive/AlphaBuilder

# Install FEniCSx (Crucial for Data Harvest, Optional for Training if using pre-generated data)
try:
    import dolfinx
except ImportError:
    !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f250b7/releases/fenicsx-install-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
    import dolfinx

# Install Project Dependencies
!pip install tensorflow matplotlib
```

## 3. Notebook A: Data Harvester (CPU Runtime)

This notebook focuses on solving physics. GPU is not needed (and FEniCSx on Colab usually runs on CPU).

**Execution Command:**
```python
%cd /content/drive/MyDrive/AlphaBuilder

# Run 500 episodes with 64x32 resolution
!python run_data_harvest.py \
    --episodes 500 \
    --resolution 64x32 \
    --db-path data/training_data.db \
    --steps 100
```

*Tip: Keep this tab open to ensure continuous execution.*

## 4. Notebook B: Neural Trainer (GPU Runtime)

This notebook trains the Vision Transformer. Select **Runtime > Change runtime type > T4 GPU**.

**Execution Command:**
```python
%cd /content/drive/MyDrive/AlphaBuilder

# Train continuously
!python alphabuilder/src/neural/train.py \
    --db-path data/training_data.db \
    --log-dir logs/vit_run_colab \
    --checkpoint-dir checkpoints \
    --epochs 1000 \
    --resolution 64x32
```

## 5. Monitoring

The training script generates a plot `logs/vit_run_colab/loss_plot.png`. You can display it in the notebook periodically:

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from IPython.display import display, clear_output

while True:
    try:
        img = mpimg.imread('logs/vit_run_colab/loss_plot.png')
        clear_output(wait=True)
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except:
        pass
    time.sleep(30) # Refresh every 30s
```
