# Scaling Data Generation Strategy

## Goal
Reach 1000 episodes in 10 hours.
Current Status: ~340 episodes (170 done + 170 in progress).
Gap: ~660 episodes.
Required Rate: **66 episodes/hour**.

## Resource Analysis & Strategy

### 1. Local Arch Linux (Current)
- Rate: ~11.5 eps/hr.
- Optimization: Ensure it's running max parallel instances. If CPU usage < 100%, run another instance.

### 2. Windows Machines (Ryzen 7, Core Ultra 7)
- Potential: These are powerful CPUs.
- **Parallelism is Key**: SIMP optimization is often single-core dominant (Python overhead + serial PETSc parts) unless configured otherwise.
- **Action**: Run **4 parallel instances** on each Windows machine.
- Estimated Rate per Machine: 10 eps/hr * 4 = 40 eps/hr.
- Total for 2 Windows Machines: **80 eps/hr**.

### 3. Cloud (Colab/Kaggle)
- Colab Free: ~5.5 eps/hr.
- Use as backup or supplement.

## The Winning Combination
If you deploy the Windows machines effectively with parallel instances, you can easily exceed the target.

**Projected Rate:**
- Arch Linux: 11.5 eps/hr
- Windows 1 (Ryzen 7) x 4 instances: ~40 eps/hr
- Windows 2 (Core Ultra 7) x 4 instances: ~40 eps/hr
- Colab (Current): 5.5 eps/hr
- **Total: ~97 eps/hr** (Well above the required 66 eps/hr).

## Execution Plan

### A. Windows Setup (WSL2)
1.  **Install WSL2**: Open PowerShell as Admin -> `wsl --install`. Reboot.
2.  **Setup Environment**:
    - Open Ubuntu terminal.
    - Clone repo: `git clone https://github.com/gustavomello9600/alphabuild.git`
    - Run setup: `bash alphabuilder/setup_wsl.sh`
3.  **Run Parallel Harvest**:
    - Open 4 separate terminal tabs in WSL.
    - In each tab, run:
      ```bash
      source venv/bin/activate
      python run_data_harvest.py --episodes 50 --seed <UNIQUE_SEED>
      ```
    - *Tip*: Use different seeds (e.g., 1000, 2000, 3000, 4000) to avoid duplicates.

### B. Colab/Kaggle
- Continue running the existing Colab notebook.
- If needed, open a Kaggle Notebook, clone the repo, and run the same script.

### C. Monitoring
- Check the `training_data.db` size.
- Merge databases later using a simple script (sqlite3 or python).

## Scripts Provided
- `run_data_harvest.py`: The main worker script.
- `setup_wsl.sh`: Automated setup for Windows (WSL).
