import os
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import signal
import sys

# Import Modules
from modules.translate_video.endpoints import router as video_router
from modules.transcribe.endpoints import router as transcribe_router
from modules.analysis.endpoints import router as analysis_router
from modules.recorder.endpoints import router as recorder_router
from modules.tts.endpoints import router as tts_router

# Import task manager for startup/shutdown handling
from modules.translate_video.task_manager import task_manager
from core.database import db

def setup_signal_handlers():
    """Setup handlers for graceful shutdown on Windows and Unix."""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        # Note: We can't directly await here, but we set a flag that
        # the lifespan context manager will handle
        global _shutdown_requested
        _shutdown_requested = True
    
    _shutdown_requested = False
    
    # Handle common termination signals
    try:
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        
        # Windows specific
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)  # Ctrl+Break
    except ValueError:
        # May fail if not in main thread
        pass
    
    return lambda: _shutdown_requested

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown.
    This is the modern replacement for on_event("startup")/("shutdown").
    """
    # STARTUP
    print("=" * 50)
    print("VoiceDub Pro - Starting up...")
    print("=" * 50)
    
    # Recover interrupted tasks from previous run
    print("Checking for interrupted tasks...")
    try:
        await task_manager.recover_interrupted_tasks()
    except Exception as e:
        print(f"Warning: Task recovery failed: {e}")
        # Continue startup even if recovery fails
    
    # Get shutdown checker
    shutdown_checker = setup_signal_handlers()
    
    yield  # Server runs here
    
    # SHUTDOWN
    print("=" * 50)
    print("VoiceDub Pro - Shutting down gracefully...")
    print("=" * 50)
    
    # Perform graceful shutdown
    try:
        await task_manager.shutdown()
    except Exception as e:
        print(f"Warning: Graceful shutdown failed: {e}")
    
    # Clear running flags for clean restart
    try:
        db.clear_running_flags()
    except Exception as e:
        print(f"Warning: Failed to clear running flags: {e}")
    
    print("Shutdown complete")

app = FastAPI(
    title="VoiceDub Pro Modular",
    lifespan=lifespan
)

# 1. CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Directory Setup
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
for d in ["uploads", "outputs", "temp_chunks", "static", "templates", "projects_db"]:
    Path(d).mkdir(exist_ok=True, parents=True)

# 3. Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=Path("outputs")), name="outputs")
app.mount("/uploads", StaticFiles(directory=Path("uploads")), name="uploads")

# NEW: Mount temp_chunks for audio samples
app.mount("/temp_chunks", StaticFiles(directory=Path("temp_chunks")), name="temp_chunks")

# 4. Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 5. Routers
app.include_router(video_router)
app.include_router(transcribe_router)
app.include_router(analysis_router)
app.include_router(recorder_router)
app.include_router(tts_router)

# 6. Root
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
