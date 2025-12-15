# GEMINI.md

## Project Overview

This project, "AlphaBuilder", is a tool for topological optimization using a combination of traditional methods (SIMP/FEniTop) and deep learning. The goal is to generate optimal structures based on given constraints.

The project is composed of:
- A Python backend that uses FEniCSx for finite element analysis and PyTorch for the neural network.
- A data generation pipeline that uses MPI for parallel processing to create training data for the neural network.
- A React frontend for visualizing the optimization process and results.
- A FastAPI backend to serve data to the frontend.

## Building and Running

### Backend

The backend is a Python project. Dependencies are managed with Conda and defined in `environment.yml`.

**Setup:**
```bash
mamba env create -f environment.yml
mamba activate alphabuilder
```

**Running Data Harvest:**
```bash
# Generate episode with Bezier strategy
mpirun -np 4 python run_data_harvest.py --strategy BEZIER --episodes 1

# Generate episode with Full Domain strategy
mpirun -np 4 python run_data_harvest.py --strategy FULL_DOMAIN --episodes 1
```

**Running the web server:**
The backend web server is a FastAPI application. To run it:
```bash
python alphabuilder/web/backend/main.py
```

### Frontend

The frontend is a React application built with Vite.

**Setup:**
```bash
cd alphabuilder/web
npm install
```

**Running:**
```bash
npm run dev
```
The application will be available at `http://localhost:5173`.

## Development Conventions

- The Python code is formatted with `ruff` and type-checked with `mypy`.
- Tests are written with `pytest` and located in the `tests/` directory.
- The frontend code is formatted with `eslint`.
- The project uses a custom logger for training, and the logs are stored in `data/logs/`.
- The main development branch is likely `main` or `master`. Commits should be descriptive and follow conventional commit message formats if any are established.

## Key Files

- `README.md`: Project overview and setup instructions.
- `pyproject.toml`: Python project configuration, including dependencies and scripts.
- `environment.yml`: Conda environment definition.
- `alphabuilder/web/package.json`: Frontend project configuration and dependencies.
- `alphabuilder/web/backend/main.py`: FastAPI backend entrypoint.
- `run_data_harvest.py`: Main script for generating training data.
- `alphabuilder/src/neural/model.py`: Neural network architecture definition.
- `alphabuilder/src/logic/harvest/optimization.py`: FEniTop optimization logic.
- `alphabuilder/web/src/App.tsx`: Main React component for the frontend.
