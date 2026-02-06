import asyncio
import shutil
import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core.database import db
from .state import connect_task_websocket, disconnect_task_websocket, broadcast_to_task
from .task_manager import task_manager

router = APIRouter()
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp_chunks")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------------

class SpeakerConfigUpdate(BaseModel):
    speakers: Dict[str, Any]  # speaker_id -> {name, action, merged_into?}

class TranslationRequest(BaseModel):
    tts_engine: str = "f5"
    target_language: str = "en"

class YouTubeDownloadRequest(BaseModel):
    url: str
    quality: str = "best"

class YouTubeUploadRequest(BaseModel):
    file_path: str
    filename: str
    tgt_lang: str = "en"
    src_lang: str = "auto"  # NEW: source language
    separate_audio: bool = False  # NEW: background separation
    tts_engine: str = "f5"  # NEW: TTS engine selection

class ResumeRequest(BaseModel):
    task_id: str
    from_phase: Optional[str] = None  # Optional: restart from specific phase

class RestartRequest(BaseModel):
    task_id: Optional[str] = None  # Made optional - we get this from path
    from_phase: Optional[str] = None  # "init", "identifying", "translating", etc."

# -------------------------------------------------------------------------
# Recovery Endpoint
# -------------------------------------------------------------------------

@router.post("/api/recover-tasks")
async def recover_tasks_endpoint():
    """
    Manually trigger recovery of interrupted tasks.
    Normally called automatically on startup, but available for manual retry.
    """
    try:
        await task_manager.recover_interrupted_tasks()
        return {"status": "recovery_initiated"}
    except Exception as e:
        raise HTTPException(500, f"Recovery failed: {str(e)}")

# -------------------------------------------------------------------------
# YouTube Download Endpoint
# -------------------------------------------------------------------------

@router.post("/api/youtube/download")
async def download_youtube_video(request: YouTubeDownloadRequest):
    """Download a video from YouTube and return the file path."""
    try:
        import yt_dlp
        
        # Generate unique filename
        video_id = str(uuid.uuid4())[:8]
        output_template = str(UPLOAD_DIR / f"youtube_{video_id}_%(title)s.%(ext)s")
        
        # yt-dlp options
        ydl_opts = {
            'format': 'best[height<=720]/best',  # Prefer 720p or lower for processing speed
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            
            # Check if file exists (yt-dlp might change extension)
            actual_file = Path(downloaded_file)
            if not actual_file.exists():
                # Try common video extensions
                for ext in ['.mp4', '.webm', '.mkv']:
                    test_file = actual_file.with_suffix(ext)
                    if test_file.exists():
                        actual_file = test_file
                        break
            
            if not actual_file.exists():
                raise HTTPException(500, "Downloaded file not found")
            
            # Get the actual filename
            filename = actual_file.name
            
            return {
                "success": True,
                "file_path": str(actual_file),
                "filename": filename,
                "title": info.get('title', 'Unknown'),
                "duration": info.get('duration', 0),
                "uploader": info.get('uploader', 'Unknown')
            }
            
    except Exception as e:
        raise HTTPException(500, f"YouTube download failed: {str(e)}")

@router.get("/api/youtube/download-file")
async def download_youtube_file(file_path: str = Query(..., description="Path to the downloaded YouTube video")):
    """
    Serve a previously downloaded YouTube video file for direct download.
    This allows users to retrieve the original downloaded video without re-downloading from YouTube.
    """
    try:
        path = Path(file_path)
        
        # Security check: ensure the path is within the uploads directory
        # This prevents directory traversal attacks
        try:
            path.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(403, "Access denied: file path is not in allowed directory")
        
        if not path.exists():
            raise HTTPException(404, "Video file not found")
        
        if not path.is_file():
            raise HTTPException(400, "Invalid file path")
        
        # Determine media type based on extension
        ext = path.suffix.lower()
        media_types = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.mov': 'video/quicktime',
        }
        media_type = media_types.get(ext, 'application/octet-stream')
        
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
            content_disposition_type='attachment'  # Force download rather than inline playback
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to serve video file: {str(e)}")

@router.post("/api/upload-youtube")
async def upload_youtube_video(
    background_tasks: BackgroundTasks,
    request: YouTubeUploadRequest
):
    """Create a task from an already downloaded YouTube video."""
    task_id = str(uuid.uuid4())
    
    # Verify the file exists
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(404, "Downloaded video file not found")
    
    # Create database entry - use file_path as the video_path for consistency
    db.create_task(task_id, request.filename, str(file_path))
    
    # Update the task with all new configuration options
    # FIX: Ensure tgt_lang is always set, never None
    tgt_lang = request.tgt_lang if request.tgt_lang else "en"
    
    db.update_task(
        task_id,
        tgt_lang=tgt_lang,
        src_lang=request.src_lang,  # NEW
        separate_audio=request.separate_audio,  # NEW
        tts_engine=request.tts_engine,  # NEW
        status='queued',
        phase='init',
        source='youtube',
        video_path=str(file_path),
        original_filename=request.filename,
        input_filename=request.filename,
        was_running_at_shutdown=0,
        resume_attempts=0
    )
    
    # Start background processing
    background_tasks.add_task(task_manager.start_task, task_id)
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "YouTube video uploaded, starting speaker identification..."
    }

# -------------------------------------------------------------------------
# Task Control Endpoints (Resume/Restart)
# -------------------------------------------------------------------------

@router.post("/api/projects/{task_id}/resume")
async def resume_task(task_id: str):
    """
    Resume a paused, failed, or interrupted task.
    This is for continuing from where the task left off.
    """
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Project not found")
    
    # Check if task can be resumed
    allowed_statuses = ['paused', 'failed', 'error', 'resuming']
    if task.get('status') not in allowed_statuses:
        # Also allow resuming "processing" tasks that might be stuck
        if task.get('status') == 'processing':
            # Check if task is actually running
            if task_id not in task_manager.active_tasks:
                # Task is marked processing but not running - can resume
                pass
            else:
                raise HTTPException(400, f"Task is already running")
        else:
            raise HTTPException(400, f"Cannot resume task with status '{task.get('status')}'")
    
    try:
        await task_manager.resume_task(task_id)
        return {
            "task_id": task_id,
            "status": "resuming",
            "message": "Task is resuming from last checkpoint"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to resume task: {str(e)}")

@router.post("/api/projects/{task_id}/restart")
async def restart_task_endpoint(
    task_id: str,
    request: RestartRequest = None
):
    """
    Restart a task from the beginning or a specific phase.
    This clears progress and starts fresh, but keeps the uploaded file.
    """
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Project not found")
    
    from_phase = request.from_phase if request else None
    
    # Validate phase if provided
    valid_phases = ['init', 'identifying', 'translating', 'recomposing']
    if from_phase and from_phase not in valid_phases:
        raise HTTPException(400, f"Invalid phase '{from_phase}'. Must be one of: {valid_phases}")
    
    try:
        fresh_task = await task_manager.restart_task(task_id, from_phase)
        return {
            "task_id": task_id,
            "status": "restarting",
            "from_phase": from_phase or "init",
            "message": f"Task restarted from {from_phase or 'the beginning'}"
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to restart task: {str(e)}")

# -------------------------------------------------------------------------
# WEBSOCKET (Live Updates + State Sync)
# -------------------------------------------------------------------------

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await connect_task_websocket(task_id, websocket)
    try:
        while True:
            # Keep connection alive, handle client commands
            data = await websocket.receive_json()
            msg_type = data.get('type')
            
            if msg_type == 'ping':
                await websocket.send_json({'type': 'pong'})
            elif msg_type == 'validate_speakers':
                # User submitted speaker validation
                await handle_speaker_validation(task_id, data.get('config', {}))
            elif msg_type == 'start_translation':
                # User wants to start translation phase
                await task_manager.start_task(task_id)
            elif msg_type == 'control':
                # Pause/resume/cancel
                action = data.get('action')
                if action == 'cancel':
                    await task_manager.cancel_task(task_id)
                elif action == 'pause':
                    await task_manager.pause_task(task_id)
                elif action == 'resume':
                    await task_manager.resume_task(task_id)
                elif action == 'restart':
                    from_phase = data.get('from_phase')
                    await task_manager.restart_task(task_id, from_phase)
                    
    except WebSocketDisconnect:
        await disconnect_task_websocket(task_id, websocket)
    except Exception as e:
        try:
            await websocket.send_json({'type': 'error', 'message': str(e)})
        except:
            pass
        await disconnect_task_websocket(task_id, websocket)

async def handle_speaker_validation(task_id: str, config: Dict[str, Any]):
    """Process user-validated speaker configuration."""
    # Validate config
    valid_actions = {'translate', 'remove', 'dub'}  # Added 'dub' as valid action
    for spk_id, info in config.items():
        if info.get('action') not in valid_actions:
            raise HTTPException(400, f"Invalid action for speaker {spk_id}")
    
    # Update database FIRST - atomically transition to next phase
    db.update_task(
        task_id,
        speaker_config=config,
        phase='translating',  # IMMEDIATELY change phase
        status='queued',      # Set to queued so task_manager will pick it up
        was_running_at_shutdown=0,  # Safe to pause here
        message='Speaker validation accepted, starting translation...'
    )
    
    # Send SINGLE confirmation message with full state
    # This replaces the multiple redundant broadcasts that caused race conditions
    confirmation_data = {
        'type': 'validation_accepted',
        'data': {
            'task_id': task_id,
            'status': 'queued',
            'phase': 'translating',
            'progress': 35,
            'message': 'Speaker validation accepted, starting translation...',
            'speaker_config': config,
            'was_running_at_shutdown': False
        }
    }
    
    await broadcast_to_task(task_id, confirmation_data)
    
    # Auto-start translation if configured
    auto_start = config.get('_auto_start', True)
    if auto_start:
        # Use asyncio.create_task to not block the WebSocket response
        asyncio.create_task(task_manager.start_task(task_id))

# -------------------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------------------

@router.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tgt_lang: str = Form("en"),
    src_lang: str = Form("auto"),  # NEW: source language
    separate_audio: str = Form("false"),  # NEW: background separation (string from form)
    tts_engine: str = Form("f5")  # NEW: TTS engine
):
    """Upload video and start Phase 1 (diarization)."""
    task_id = str(uuid.uuid4())
    
    # Parse boolean from form string
    separate_audio_bool = separate_audio.lower() in ('true', '1', 'yes', 'on')
    
    # Ensure tgt_lang is never None
    if not tgt_lang or tgt_lang.strip() == "":
        tgt_lang = "en"
    
    # Ensure directories
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Create database entry
    db.create_task(task_id, file.filename, str(file_path))
    
    # Update the task with all configuration options
    db.update_task(
        task_id,
        tgt_lang=tgt_lang,
        src_lang=src_lang,  # NEW
        separate_audio=separate_audio_bool,  # NEW
        tts_engine=tts_engine,  # NEW
        status='queued',
        phase='init',
        source='upload',
        video_path=str(file_path),
        original_filename=file.filename,
        input_filename=file.filename,
        was_running_at_shutdown=0,
        resume_attempts=0
    )
    
    # Start background processing
    background_tasks.add_task(task_manager.start_task, task_id)
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Video uploaded, starting speaker identification..."
    }

@router.get("/api/projects")
async def list_projects():
    """List all projects for dashboard."""
    tasks = db.list_tasks(limit=50)
    return {"tasks": tasks}

@router.get("/api/projects/{task_id}")
async def get_project(task_id: str):
    """Get full project state (for resume/reconnect)."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Project not found")
    
    # Enrich with translation segments if available
    if task.get('phase') in ['translating', 'recomposing', 'complete']:
        translations = db.get_translation_segments(task_id)
        task['translations'] = translations
    
    # Add resumption info
    can_resume = task.get('status') in ['paused', 'failed', 'error', 'resuming']
    can_restart = task.get('status') not in ['processing', 'queued']
    
    task['resumption'] = {
        'can_resume': can_resume,
        'can_restart': can_restart,
        'resume_attempts': task.get('resume_attempts', 0),
        'was_interrupted': task.get('was_running_at_shutdown', False)
    }
    
    return task

@router.delete("/api/projects/{task_id}")
async def delete_project(task_id: str):
    """Delete project and all associated data."""
    # Cancel if running
    if task_id in task_manager.active_tasks:
        await task_manager.cancel_task(task_id)
    
    # Delete from database
    db.delete_task(task_id)
    
    # Clean up files (optional, async)
    def cleanup():
        paths = [
            Path("uploads") / f"{task_id}_*",
            Path("temp_chunks") / task_id,
            Path("outputs") / task_id,
        ]
        for pattern in paths:
            for p in Path('.').glob(str(pattern)):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
    
    import threading
    threading.Thread(target=cleanup).start()
    
    return {"status": "deleted"}

@router.get("/api/projects/{task_id}/segments")
async def get_segments(task_id: str, with_audio: bool = False):
    """Get speech segments with optional audio URLs."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Project not found")
    
    segments = task.get('segments', [])
    
    if with_audio and task.get('phase') == 'awaiting_validation':
        # Add audio URLs for preview
        for seg in segments:
            seg['audio_url'] = f"/temp_chunks/{task_id}/segment_{seg['idx']:04d}.wav"
    
    return {"segments": segments}

@router.get("/api/projects/{task_id}/speaker_samples/{speaker_id}")
async def get_speaker_sample(task_id: str, speaker_id: int):
    """Get audio file for a speaker sample."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Project not found")
    
    config = task.get('speaker_config', {})
    spk_info = config.get(str(speaker_id))
    
    if not spk_info:
        raise HTTPException(404, "Speaker not found")
    
    sample_path = Path("temp_chunks") / task_id / "speaker_samples" / f"speaker_{speaker_id}_sample.wav"
    if not sample_path.exists():
        raise HTTPException(404, "Sample file not found")
    
    return FileResponse(sample_path, media_type="audio/wav")

@router.post("/api/projects/{task_id}/validate")
async def validate_speakers_api(task_id: str, update: SpeakerConfigUpdate):
    """REST endpoint for speaker validation (alternative to WebSocket)."""
    await handle_speaker_validation(task_id, update.speakers)
    return {"status": "accepted"}

@router.get("/api/projects/{task_id}/download")
async def download_result(task_id: str):
    """Download final video file."""
    task = db.get_task(task_id)
    if not task or not task.get('output_path'):
        raise HTTPException(404, "Output not ready")
    
    output_path = Path(task['output_path'])
    if not output_path.exists():
        raise HTTPException(404, "File not found on disk")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"dubbed_{task['original_filename']}"
    )

@router.get("/api/projects/{task_id}/preview/{segment_idx}")
async def preview_segment_audio(task_id: str, segment_idx: int):
    """Get audio for a specific segment (original or translated)."""
    # Check if translated version exists
    translations = db.get_translation_segments(task_id)
    trans = next((t for t in translations if t['segment_idx'] == segment_idx), None)
    
    if trans and trans.get('audio_path') and Path(trans['audio_path']).exists():
        return FileResponse(trans['audio_path'], media_type="audio/wav")
    
    # Fall back to original segment
    task = db.get_task(task_id)
    if task and task.get('segments') and segment_idx < len(task['segments']):
        seg_path = task['segments'][segment_idx].get('audio_path')
        if seg_path and Path(seg_path).exists():
            return FileResponse(seg_path, media_type="audio/wav")
    
    raise HTTPException(404, "Audio not found")
