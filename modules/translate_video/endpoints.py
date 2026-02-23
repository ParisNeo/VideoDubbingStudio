# modules/translate_video/endpoints.py
"""
Video Translation API Endpoints

Handles video upload, YouTube download, speaker validation,
and task management for the video dubbing pipeline.
"""

# CRITICAL FIX: Ensure project root is in Python path before absolute imports
# This prevents "No module named 'core'" errors when the server is run
# from different working directories or via IDE configurations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import asyncio
import shutil
import os
import uuid
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core.database import db
from modules.translate_video.state import connect_task_websocket, disconnect_task_websocket, broadcast_to_task
from modules.translate_video.task_manager import task_manager

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
    speakers: Dict[str, Any]

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
    src_lang: str = "auto"
    separate_audio: bool = False
    tts_engine: str = "f5"

class ResumeRequest(BaseModel):
    task_id: str
    from_phase: Optional[str] = None

class RestartRequest(BaseModel):
    task_id: Optional[str] = None
    from_phase: Optional[str] = None

# -------------------------------------------------------------------------
# Recovery Endpoint
# -------------------------------------------------------------------------

@router.post("/api/recover-tasks")
async def recover_tasks_endpoint():
    try:
        await task_manager.recover_interrupted_tasks()
        return {"status": "recovery_initiated"}
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Recovery endpoint failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Recovery failed: {str(e)}\n\nFull traceback in server logs")

# -------------------------------------------------------------------------
# YouTube Download Endpoint
# -------------------------------------------------------------------------

@router.post("/api/youtube/download")
async def download_youtube_video(request: YouTubeDownloadRequest):
    try:
        import yt_dlp
        
        video_id = str(uuid.uuid4())[:8]
        output_template = str(UPLOAD_DIR / f"youtube_{video_id}_%(title)s.%(ext)s")
        
        ydl_opts = {
            'format': 'best[height<=720]/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
            downloaded_file = ydl.prepare_filename(info)
            
            actual_file = Path(downloaded_file)
            if not actual_file.exists():
                for ext in ['.mp4', '.webm', '.mkv']:
                    test_file = actual_file.with_suffix(ext)
                    if test_file.exists():
                        actual_file = test_file
                        break
            
            if not actual_file.exists():
                raise HTTPException(500, "Downloaded file not found")
            
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
        tb_str = traceback.format_exc()
        print(f"YouTube download failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"YouTube download failed: {str(e)}\n\nFull traceback in server logs")

@router.get("/api/youtube/download-file")
async def download_youtube_file(file_path: str = Query(..., description="Path to the downloaded YouTube video")):
    try:
        path = Path(file_path)
        
        try:
            path.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(403, "Access denied: file path is not in allowed directory")
        
        if not path.exists():
            raise HTTPException(404, "Video file not found")
        
        if not path.is_file():
            raise HTTPException(400, "Invalid file path")
        
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
            content_disposition_type='attachment'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Download file endpoint failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to serve video file: {str(e)}\n\nFull traceback in server logs")

@router.post("/api/upload-youtube")
async def upload_youtube_video(
    background_tasks: BackgroundTasks,
    request: YouTubeUploadRequest
):
    try:
        task_id = str(uuid.uuid4())
        
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise HTTPException(404, "Downloaded video file not found")
        
        db.create_task(task_id, request.filename, str(file_path))
        
        # CRITICAL FIX: Ensure we use the provided tgt_lang, never default to 'en'
        tgt_lang = request.tgt_lang if request.tgt_lang and request.tgt_lang.strip() else "en"
        src_lang = request.src_lang if request.src_lang and request.src_lang.strip() else "auto"
        
        print(f"CRITICAL: YouTube upload - src_lang={src_lang}, tgt_lang={tgt_lang}, tts_engine={request.tts_engine}")
        
        db.update_task(
            task_id,
            tgt_lang=tgt_lang,
            src_lang=src_lang,
            separate_audio=request.separate_audio,
            tts_engine=request.tts_engine,
            status='queued',
            phase='init',
            source='youtube',
            video_path=str(file_path),
            original_filename=request.filename,
            input_filename=request.filename,
            was_running_at_shutdown=0,
            resume_attempts=0
        )
        
        # Verify the task was saved correctly
        saved_task = db.get_task(task_id)
        print(f"CRITICAL: Verified saved task - tgt_lang={saved_task.get('tgt_lang')}, src_lang={saved_task.get('src_lang')}")
        
        background_tasks.add_task(task_manager.start_task, task_id)
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "YouTube video uploaded, starting speaker identification..."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"YouTube upload endpoint failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to process YouTube upload: {str(e)}\n\nFull traceback in server logs")

# -------------------------------------------------------------------------
# Task Control Endpoints
# -------------------------------------------------------------------------

@router.post("/api/projects/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    try:
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, "Project not found")
        
        await task_manager.cancel_task(task_id)
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Cancel task failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to cancel task: {str(e)}\n\nFull traceback in server logs")

@router.post("/api/projects/{task_id}/resume")
async def resume_task(task_id: str):
    try:
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, "Project not found")
        
        # If the task is already in active_tasks, we'll force-stop it first to allow a clean resume.
        if task_id in task_manager.active_tasks:
            print(f"Task {task_id} is hung or running, force-stopping before resume...")
            await task_manager.cancel_task(task_id)
            # Short sleep to allow cancellation to propagate
            await asyncio.sleep(0.5)

        await task_manager.resume_task(task_id)
        return {
            "task_id": task_id,
            "status": "resuming",
            "message": "Task is resuming from last checkpoint"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Resume task failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to resume task: {str(e)}\n\nFull traceback in server logs")

@router.post("/api/projects/{task_id}/restart")
async def restart_task_endpoint(
    task_id: str,
    request: RestartRequest = None
):
    try:
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, "Project not found")
        
        from_phase = request.from_phase if request else None
        
        valid_phases = ['init', 'identifying', 'translating', 'recomposing']
        if from_phase and from_phase not in valid_phases:
            raise HTTPException(400, f"Invalid phase '{from_phase}'. Must be one of: {valid_phases}")
        
        fresh_task = await task_manager.restart_task(task_id, from_phase)
        return {
            "task_id": task_id,
            "status": "restarting",
            "from_phase": from_phase or "init",
            "message": f"Task restarted from {from_phase or 'the beginning'}"
        }
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Restart task failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to restart task: {str(e)}\n\nFull traceback in server logs")

# -------------------------------------------------------------------------
# WEBSOCKET (Live Updates + State Sync)
# -------------------------------------------------------------------------

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    connected = False
    try:
        # First accept the connection
        await websocket.accept()
        connected = True
        
        # Then register it
        await connect_task_websocket(task_id, websocket)
        
        # Send initial state immediately
        from core.database import db
        task = db.get_task(task_id)
        if task:
            await websocket.send_json({
                'type': 'state_sync',
                'data': task
            })
        
        # Main message loop
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get('type')
                
                if msg_type == 'ping':
                    await websocket.send_json({'type': 'pong'})
                elif msg_type == 'validate_speakers':
                    await handle_speaker_validation(task_id, data.get('config', {}))
                elif msg_type == 'validate_translation':
                    await handle_translation_validation(task_id, data.get('config', {}))
                elif msg_type == 'start_translation':
                    await task_manager.start_task(task_id)
                elif msg_type == 'start_tts':
                    await task_manager.start_task(task_id)
                elif msg_type == 'control':
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
                # Client disconnected - break out of loop cleanly
                print(f"WebSocket client disconnected for task {task_id}")
                break
            except Exception as loop_err:
                # Log but don't break the connection for message handling errors
                print(f"WebSocket message error for task {task_id}: {loop_err}")
                # Check if it's a disconnect-related error
                err_str = str(loop_err).lower()
                if 'disconnect' in err_str or 'cannot call' in err_str and 'receive' in err_str:
                    print(f"Disconnect detected, breaking loop for task {task_id}")
                    break
                try:
                    await websocket.send_json({
                        'type': 'error', 
                        'message': f'Message handling error: {str(loop_err)}'
                    })
                except:
                    # If we can't send, client is likely gone
                    break
                # Continue the loop for other errors
                continue
                    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected normally for task {task_id}")
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"WebSocket error for task {task_id}:\n{tb_str}")
        if connected:
            try:
                await websocket.send_json({'type': 'error', 'message': str(e), 'traceback': tb_str})
            except:
                pass
    finally:
        # Always cleanup
        if connected:
            try:
                await disconnect_task_websocket(task_id, websocket)
            except Exception as cleanup_err:
                print(f"WebSocket cleanup error for task {task_id}: {cleanup_err}")

async def handle_speaker_validation(task_id: str, config: Dict[str, Any]):
    """
    Step 1: Speakers validated -> Start Transcription.
    """
    try:
        # Merge speaker config
        existing_task = db.get_task(task_id)
        existing_config = existing_task.get('speaker_config', {}) if existing_task else {}
        
        merged_config = {}
        for spk_id, new_info in config.items():
            if spk_id.startswith('_'): continue
            merged = dict(existing_config.get(spk_id, {}))
            merged.update(new_info)
            merged_config[spk_id] = merged
            
        # Update and move to TRANSCRIPTION
        db.update_task(task_id, 
            speaker_config=merged_config,
            phase='transcribing', 
            status='queued',
            message='Speakers confirmed - starting transcription...'
        )
        
        asyncio.create_task(task_manager.start_task(task_id))
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Speaker validation failed: {str(e)}")

async def handle_transcription_validation(task_id: str, segments: List[Dict]):
    """
    Step 2: Transcription validated -> Start Translation.
    """
    try:
        db.update_task(task_id, 
            segments=segments,
            phase='translating', 
            status='queued',
            message='Transcription confirmed - starting translation...'
        )
        asyncio.create_task(task_manager.start_task(task_id))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Transcription validation failed: {str(e)}")

async def handle_translation_validation(task_id: str, segments: List[Dict]):
    """
    Step 3: Translation validated -> Start TTS.
    """
    try:
        # Handle wrapped format if necessary
        if isinstance(segments, dict) and 'edited_segments' in segments:
            segments = segments['edited_segments']
            
        # Update segments if they are full objects, or merge edits
        # Ideally frontend sends full updated segment list
        
        db.update_task(task_id, 
            segments=segments,
            phase='synthesizing', 
            status='queued',
            message='Translation confirmed - starting voice synthesis...'
        )
        asyncio.create_task(task_manager.start_task(task_id))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Translation validation failed: {str(e)}")

async def handle_audio_validation(task_id: str):
    """
    Step 4: Audio validated -> Start Final Assembly.
    """
    try:
        db.update_task(task_id, 
            phase='recomposing', 
            status='queued',
            message='Audio confirmed - starting final assembly...'
        )
        asyncio.create_task(task_manager.start_task(task_id))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Audio validation failed: {str(e)}")


async def handle_translation_validation(task_id: str, config: Any):
    """
    Handle translation review and confirmation.
    """
    try:
        # Robust extraction: handle if config is a list or a dict
        if isinstance(config, list):
            edited_segments = config
            proceed_to_tts = True
        else:
            edited_segments = config.get('edited_segments')
            proceed_to_tts = config.get('proceed_to_tts', True)
        
        existing_task = db.get_task(task_id)
        if not existing_task:
            raise HTTPException(404, "Task not found")
        
        # Preserve language settings
        tgt_lang = existing_task.get('tgt_lang') or existing_task.get('target_language') or 'en'
        src_lang = existing_task.get('src_lang') or 'auto'
        tts_engine = existing_task.get('tts_engine') or 'f5'
        
        # Update segments with any user edits
        segments = existing_task.get('segments', [])
        if edited_segments:
            # Merge user edits with existing segments
            for edited in edited_segments:
                idx = edited.get('idx')
                for seg in segments:
                    if seg.get('idx') == idx:
                        seg['translated_text'] = edited.get('translated_text', seg.get('translated_text'))
                        seg['original_text'] = edited.get('original_text', seg.get('original_text'))
                        break
        
        print(f"Translation validation: src={src_lang}, tgt={tgt_lang}, edited={edited_segments is not None}")
        
        # Update task - move to TTS synthesis
        db.update_task(
            task_id,
            segments=segments,
            phase='synthesizing',  # Changed from 'translating' to avoid re-translation loop
            status='queued',
            was_running_at_shutdown=0,
            message='Translation validated - starting voice synthesis...',
            tgt_lang=tgt_lang,
            src_lang=src_lang,
            tts_engine=tts_engine
        )
        
        # Send confirmation
        confirmation_data = {
            'type': 'translation_validated',
            'data': {
                'task_id': task_id,
                'status': 'queued',
                'phase': 'translating',
                'progress': 60,
                'message': 'Starting voice synthesis...',
                'segment_count': len(segments),
                'tgt_lang': tgt_lang,
                'src_lang': src_lang
            }
        }
        
        await broadcast_to_task(task_id, confirmation_data)
        
        # Auto-start TTS
        if proceed_to_tts:
            asyncio.create_task(task_manager.start_task(task_id))
            
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Translation validation failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to process translation validation: {str(e)}\n\nFull traceback in server logs")

# -------------------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------------------

@router.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tgt_lang: str = Form("en"),
    src_lang: str = Form("auto"),
    separate_audio: str = Form("false"),
    tts_engine: str = Form("f5")
):
    try:
        task_id = str(uuid.uuid4())
        
        separate_audio_bool = separate_audio.lower() in ('true', '1', 'yes', 'on')
        
        # CRITICAL FIX: Ensure we never accept None or empty tgt_lang
        if not tgt_lang or tgt_lang.strip() == "":
            tgt_lang = "en"
        
        # Log what we received for debugging
        print(f"CRITICAL: Regular upload - file={file.filename}, src_lang={src_lang}, tgt_lang={tgt_lang}, tts_engine={tts_engine}, separate_audio={separate_audio_bool}")
        
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        db.create_task(task_id, file.filename, str(file_path))
        
        db.update_task(
            task_id,
            tgt_lang=tgt_lang,
            src_lang=src_lang,
            separate_audio=separate_audio_bool,
            tts_engine=tts_engine,
            status='queued',
            phase='init',
            source='upload',
            video_path=str(file_path),
            original_filename=file.filename,
            input_filename=file.filename,
            was_running_at_shutdown=0,
            resume_attempts=0
        )
        
        # Verify the task was saved correctly
        saved_task = db.get_task(task_id)
        print(f"CRITICAL: Verified saved task after regular upload - tgt_lang={saved_task.get('tgt_lang')}, target_language={saved_task.get('target_language')}, src_lang={saved_task.get('src_lang')}")
        
        background_tasks.add_task(task_manager.start_task, task_id)
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Video uploaded, starting speaker identification..."
        }
        
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Upload endpoint failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Upload failed: {str(e)}\n\nFull traceback in server logs")

@router.get("/api/projects")
async def list_projects():
    try:
        tasks = db.list_tasks(limit=50)
        return {"tasks": tasks}
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"List projects failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to list projects: {str(e)}\n\nFull traceback in server logs")

@router.get("/api/projects/{task_id}")
async def get_project(task_id: str):
    try:
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, "Project not found")
        
        if task.get('phase') in ['translating', 'recomposing', 'complete']:
            translations = db.get_translation_segments(task_id)
            task['translations'] = translations
        
        can_resume = task.get('status') in ['paused', 'failed', 'error', 'resuming']
        can_restart = task.get('status') not in ['processing', 'queued']
        
        task['resumption'] = {
            'can_resume': can_resume,
            'can_restart': can_restart,
            'resume_attempts': task.get('resume_attempts', 0),
            'was_interrupted': task.get('was_running_at_shutdown', False)
        }
        
        return task
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Get project failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to get project: {str(e)}\n\nFull traceback in server logs")

@router.delete("/api/projects/{task_id}")
async def delete_project(task_id: str, force: bool = True):
    """Delete a project, force-stopping if running."""
    try:
        # Always try to cancel/stop the task first
        if task_id in task_manager.active_tasks:
            await task_manager.cancel_task(task_id)
            # Wait a moment for graceful shutdown
            await asyncio.sleep(0.5)
        
        # Clear running flags to ensure deletion succeeds
        db.update_task(task_id, was_running_at_shutdown=0, status='cancelled')
        
        db.delete_task(task_id)
        
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
        
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Delete project failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to delete project: {str(e)}\n\nFull traceback in server logs")

@router.get("/api/projects/{task_id}/segments")
async def get_segments(task_id: str, with_audio: bool = False):
    try:
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, "Project not found")
        
        segments = task.get('segments', [])
        
        if with_audio and task.get('phase') == 'awaiting_validation':
            for seg in segments:
                seg['audio_url'] = f"/temp_chunks/{task_id}/segment_{seg['idx']:04d}.wav"
        
        return {"segments": segments}
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Get segments failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to get segments: {str(e)}\n\nFull traceback in server logs")

@router.get("/api/projects/{task_id}/speaker_samples/{speaker_id}")
async def get_speaker_sample(task_id: str, speaker_id: int):
    try:
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
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Get speaker sample failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to get speaker sample: {str(e)}\n\nFull traceback in server logs")

@router.post("/api/projects/{task_id}/validate-speakers")
async def validate_speakers_api(task_id: str, update: SpeakerConfigUpdate):
    """Step 1: Validate speakers, trigger transcription."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Project {task_id} not found in database")
        
    await handle_speaker_validation(task_id, update.speakers)
    return {"status": "accepted", "next_phase": "transcribing"}

@router.post("/api/projects/{task_id}/validate-transcription")
async def validate_transcription_api(task_id: str, request: Request):
    """Step 2: Validate text, trigger translation."""
    body = await request.json()
    segments = body.get('segments', [])
    await handle_transcription_validation(task_id, segments)
    return {"status": "accepted", "next_phase": "translating"}

@router.post("/api/projects/{task_id}/validate-translation")
async def validate_translation_api(task_id: str, request: Request):
    """Step 3: Validate translation, trigger TTS."""
    body = await request.json()
    segments = body.get('segments', [])
    await handle_translation_validation(task_id, segments)
    return {"status": "accepted", "next_phase": "synthesizing"}

@router.post("/api/projects/{task_id}/validate-audio")
async def validate_audio_api(task_id: str):
    """Step 4: Validate audio, trigger assembly."""
    await handle_audio_validation(task_id)
    return {"status": "accepted", "next_phase": "recomposing"}
    
    
@router.get("/api/projects/{task_id}/download")
async def download_result(task_id: str):
    try:
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, "Project not found")
            
        # Try to find the file in the standard location if output_path is missing or wrong
        output_path = Path(f"outputs/{task_id}/dubbed_video.mp4")
        
        if not output_path.exists() and task.get('output_path'):
            output_path = Path(task['output_path'])

        if not output_path.exists():
            raise HTTPException(404, f"Final video file not found at {output_path}")
        
        # Use filename or input_filename as fallback for original_filename
        download_name = task.get('filename') or task.get('input_filename') or 'video.mp4'
        # Add dubbed_ prefix but keep original extension
        stem = Path(download_name).stem
        ext = output_path.suffix or '.mp4'
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"dubbed_{stem}{ext}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Download result failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to download result: {str(e)}\n\nFull traceback in server logs")

@router.get("/api/projects/{task_id}/preview/{segment_idx}")
async def preview_segment_audio(task_id: str, segment_idx: int):
    """Unified slice-and-stream service for both transcription and translation review."""
    import subprocess
    from fastapi import Response
    try:
        task = db.get_task(task_id)
        if not task: raise HTTPException(404, "Task not found")
        
        all_segs = task.get('segments', []) or task.get('transcribed_segments', [])
        if segment_idx >= len(all_segs):
            raise HTTPException(404, f"Segment {segment_idx} not found")
            
        seg = all_segs[segment_idx]

        # Priority 1: Check if synthesized TTS audio exists
        if seg.get('audio_path'):
            audio_path = Path(seg['audio_path'])
            if audio_path.exists():
                return FileResponse(str(audio_path), media_type="audio/wav")

        # Priority 2: Slice from Master Audio
        master_wav = task.get('master_audio') or task.get('file_path')
        if not master_wav:
            raise HTTPException(404, "Source audio path missing")
            
        master_path = Path(master_wav)
        if not master_path.exists():
            raise HTTPException(404, f"Source file not found at {master_wav}")

        start = seg.get('start', 0)
        end = seg.get('end', 0)
        duration = max(0.1, end - start)

        # Extract precise slice using FFmpeg - using .run for better stability on Windows
        cmd = [
            "ffmpeg", "-y", 
            "-ss", str(round(start, 3)), 
            "-t", str(round(duration, 3)),
            "-i", str(master_path.absolute()), 
            "-f", "wav", "pipe:1"
        ]
        
        # Hide the console window on Windows to prevent flickering
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            startupinfo=startupinfo,
            check=True
        )
        
        return Response(content=result.stdout, media_type="audio/wav")
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Preview segment audio failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to preview audio: {str(e)}\n\nFull traceback in server logs")
