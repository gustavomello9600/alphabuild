"""
FastAPI Backend for AlphaBuilder Training Data Viewer.

Serves data from SQLite databases for visualization in the frontend.
Supports both legacy schema (v1) and optimized schema (v2).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
try:
    from .routers import training_data, selfplay
except ImportError:
    from routers import training_data, selfplay

app = FastAPI(title="AlphaBuilder Training Data API")

# CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(training_data.router)
app.include_router(selfplay.router, prefix="/selfplay", tags=["selfplay"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
