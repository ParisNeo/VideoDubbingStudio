"""
Legacy Phase 1 and Phase 2 logic - REFACTORED to use pipeline modules.

This file now serves as a compatibility layer that imports from the
new pipeline modules. The actual implementation has been moved to:
- modules/translate_video/pipeline/phase1_diarization.py
- modules/translate_video/pipeline/phase2_translation.py

This file is kept for backward compatibility with existing imports.
"""

import asyncio
import json
import soundfile as sf
import subprocess
import traceback
import logging
from pathlib import Path
from typing import Dict, Any
import concurrent.futures
import numpy as np
import torch
import tempfile
import warnings

# Import new modular pipeline
from .pipeline.phase1_diarization import run_phase1_diarization, Phase1Diarizer
from .pipeline.phase2_translation import run_phase2_translation

# Keep existing state imports for compatibility
from .state import active_connections
from .project_manager import ProjectManager, append_log

# Setup logger for this module
logger = logging.getLogger("translate_video.logic")

# Thread pool for CPU-bound diarization (kept for compatibility)
_diarization_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# -------------------------------------------------------------------------
# LEGACY COMPATIBILITY FUNCTIONS
# -------------------------------------------------------------------------

async def start_identification_task(task_id: str, video_path: str):
    """
    Phase 1: Extract audio, diarize, AND transcribe for immediate UI display.
    
    LEGACY: Now delegates to pipeline.phase1_diarization module.
    """
    await log_ws(task_id, f"Phase 1: Analyzing {Path(video_path).name}", "step")
    
    # Create progress callback that updates WebSocket
    async def progress_callback(phase: str, percent: int, message: str):
        await progress_ws(task_id, phase, percent, message)
        await update_status(task_id, {
            "progress": percent,
            "message": message,
            "phase": phase if phase != "complete" else "awaiting_validation",
            "status": "processing" if phase != "complete" else "awaiting_validation"
        })
    
    # Create transcription callback to broadcast to UI
    async def transcription_callback(segments: list):
        await broadcast_transcription_update(task_id, segments)
    
    try:
        # Run new Phase 1 pipeline
        result = await run_phase1_diarization(
            task_id=task_id,
            video_path=video_path,
            progress_callback=progress_callback,
            transcription_callback=transcription_callback
        )
        
        # Broadcast speaker samples to trigger validation UI
        if task_id in active_connections and result.speaker_config:
            speaker_data = {}
            for spk_id, info in result.speaker_config.items():
                # Read audio file and encode as base64 for direct playback
                try:
                    import base64
                    sample_path = Path(result.speaker_samples.get(int(spk_id), ""))
                    if sample_path.exists():
                        with open(sample_path, 'rb') as f:
                            audio_bytes = f.read()
                            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        speaker_data[spk_id] = {
                            "audio_base64": audio_b64,
                            "default_name": info["name"],
                            "sample_rate": 16000,
                            "sample_path": info["sample_path"]
                        }
                except Exception as e:
                    logger.warning(f"Failed to encode speaker sample {spk_id}: {e}")
            
            if speaker_data:
                try:
                    dead_sockets = set()
                    for ws in active_connections[task_id]:
                        try:
                            await ws.send_text(json.dumps({
                                "type": "speaker_samples",
                                "data": speaker_data
                            }))
                        except Exception:
                            dead_sockets.add(ws)
                    for ws in dead_sockets:
                        active_connections[task_id].discard(ws)
                except Exception as e:
                    logger.warning(f"Failed to send speaker_samples: {e}")
        
        # Status already updated by pipeline, but ensure correct state
        await update_status(task_id, {
            "status": "awaiting_validation",
            "phase": "awaiting_validation",
            "progress": 35,
            "speaker_config": result.speaker_config,
            "transcribed_segments": result.transcribed_segments
        })
        
    except Exception as e:
        # FULL TRACEBACK LOGGING
        tb_str = traceback.format_exc()
        logger.error(f"Phase 1 failed with full traceback:\n{tb_str}")
        
        await log_ws(task_id, f"Error: {str(e)}", "error")
        await update_status(task_id, {
            "status": "failed",
            "message": str(e),
            "error_traceback": tb_str
        })


async def start_dubbing_task(task_id: str, user_config: Dict[str, Any], is_resume: bool = False, resume_from_idx: int = -1):
    """
    Phase 2: Translation and TTS generation using modular pipeline.
    
    LEGACY: Now delegates to pipeline.phase2_translation module.
    """
    await log_ws(task_id, "Phase 2: Starting Translation & Dubbing", "step")
    
    # Get task info for language and TTS engine settings
    from core.database import db
    task = db.get_task(task_id)
    
    target_language = task.get('tgt_lang', 'en') if task else 'en'
    tts_engine = task.get('tts_engine', 'f5') if task else 'f5'
    
    # Get pre-transcribed segments if available
    transcribed_segments = task.get('transcribed_segments', []) if task else []
    
    # Log GPU info
    if torch.cuda.is_available():
        await log_ws(task_id, f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)", "info")
    else:
        await log_ws(task_id, "WARNING: Running on CPU, dubbing will be slow", "warning")
    
    # Update status to processing
    await update_status(task_id, {
        "status": "processing",
        "phase": "translating",
        "progress": 35,
        "message": "Starting translation pipeline..."
    })
    
    # Create progress callback that updates WebSocket AND broadcasts translations
    async def progress_callback(phase: str, percent: int, message: str):
        await log_ws(task_id, message, "info")
        await update_status(task_id, {
            "progress": percent,
            "message": message
        })
        await progress_ws(task_id, phase, percent, message)
    
    # Create translation update callback to show progress in UI
    async def translation_update_callback(segments_batch: list):
        """Called as translations complete to update the UI progressively."""
        await broadcast_translation_progress(task_id, segments_batch)
    
    try:
        # Run the modular Phase 2 pipeline
        # Note: We now pass pre_transcribed_segments to avoid re-transcription
        success = await run_phase2_translation(
            task_id=task_id,
            speaker_config=user_config,
            target_language=target_language,
            tts_engine=tts_engine,
            is_resume=is_resume,
            resume_from_idx=resume_from_idx,
            progress_callback=progress_callback,
            translation_update_callback=translation_update_callback,
            pre_transcribed_segments=transcribed_segments
        )
        
        if success:
            await log_ws(task_id, "Phase 2 complete! Starting final assembly...", "success")
            # Phase 3 will be triggered by task_manager
        else:
            await log_ws(task_id, "Phase 2 completed with errors", "warning")
            
    except Exception as e:
        # FULL TRACEBACK LOGGING
        tb_str = traceback.format_exc()
        logger.error(f"Dubbing failed with full traceback:\n{tb_str}")
        
        await log_ws(task_id, f"Dubbing Failed: {str(e)}", "error")
        await update_status(task_id, {
            "status": "failed",
            "message": str(e),
            "error_message": str(e),
            "error_traceback": tb_str
        })
        raise  # Re-raise for task_manager to handle with full context


# -------------------------------------------------------------------------
# LEGACY WEBSOCKET UTILITIES (kept for compatibility)
# -------------------------------------------------------------------------

async def log_ws(task_id: str, msg: str, style: str = "info"):
    """Log to WebSocket AND Disk Persistence."""
    # 1. Disk
    append_log(task_id, msg, style)
    
    # 2. WebSocket (Live)
    if task_id in active_connections:
        try:
            dead_sockets = set()
            for ws in active_connections[task_id]:
                try:
                    await ws.send_text(json.dumps({
                        "type": "log",
                        "data": {"message": msg, "style": style}
                    }))
                except Exception:
                    dead_sockets.add(ws)
            
            for ws in dead_sockets:
                active_connections[task_id].discard(ws)
                
        except Exception as e:
            logger.warning(f"Failed to send log: {e}")


async def update_status(task_id: str, updates: dict):
    """Update status to WebSocket AND Disk Persistence."""
    # 1. Disk (JSON file for project manager - primary storage)
    ProjectManager.save_state(task_id, updates)
    
    # 1b. Also update SQLite database for API consistency
    from core.database import db
    try:
        db.update_task(task_id, **updates)
    except Exception as e:
        logger.warning(f"Failed to update SQLite for {task_id}: {e}")
    
    # 2. WebSocket
    state = ProjectManager.get_state(task_id)
    if task_id in active_connections and state:
        try:
            dead_sockets = set()
            for ws in active_connections[task_id]:
                try:
                    await ws.send_text(json.dumps({
                        "type": "status_update", "data": state
                    }))
                except Exception:
                    dead_sockets.add(ws)
            
            for ws in dead_sockets:
                active_connections[task_id].discard(ws)
                
        except Exception as e:
            logger.warning(f"Failed to send status update: {e}")


async def progress_ws(task_id: str, phase: str, percent: int, message: str):
    """Send progress update via WebSocket."""
    if task_id in active_connections:
        try:
            dead_sockets = set()
            for ws in active_connections[task_id]:
                try:
                    await ws.send_text(json.dumps({
                        "type": "progress",
                        "data": {
                            "phase": phase,
                            "percent": percent,
                            "message": message
                        }
                    }))
                except Exception:
                    dead_sockets.add(ws)
            
            for ws in dead_sockets:
                active_connections[task_id].discard(ws)
                
        except Exception as e:
            logger.warning(f"Failed to send progress: {e}")


async def broadcast_transcription_update(task_id: str, segments_with_transcription: list):
    """Broadcast transcription data to update the UI with speech + diarization."""
    if task_id not in active_connections:
        return
    
    try:
        dead_sockets = set()
        for ws in active_connections[task_id]:
            try:
                await ws.send_text(json.dumps({
                    "type": "transcription_update",
                    "data": {
                        "segments": segments_with_transcription
                    }
                }))
            except Exception:
                dead_sockets.add(ws)
        
        for ws in dead_sockets:
            active_connections[task_id].discard(ws)
            
    except Exception as e:
        logger.warning(f"Failed to broadcast transcription update: {e}")


async def broadcast_translation_progress(task_id: str, segments_batch: list):
    """Broadcast translation progress to update UI with translated text."""
    if task_id not in active_connections:
        return
    
    try:
        dead_sockets = set()
        for ws in active_connections[task_id]:
            try:
                await ws.send_text(json.dumps({
                    "type": "translation_progress",
                    "data": {
                        "segments": segments_batch
                    }
                }))
            except Exception:
                dead_sockets.add(ws)
        
        for ws in dead_sockets:
            active_connections[task_id].discard(ws)
            
    except Exception as e:
        logger.warning(f"Failed to broadcast translation progress: {e}")


# -------------------------------------------------------------------------
# CROSS-PLATFORM ASYNC SUBPROCESS HELPER (kept for compatibility)
# -------------------------------------------------------------------------

async def run_subprocess(cmd: list, **kwargs) -> tuple:
    """Cross-platform async subprocess execution."""
    loop = asyncio.get_event_loop()
    
    def _run_sync():
        return subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            **kwargs
        )
    
    result = await loop.run_in_executor(None, _run_sync)
    return result.returncode, result.stdout, result.stderr
