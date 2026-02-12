# modules/translate_video/endpoints.py
import asyncio
import shutil
import os
import uuid
import json
import traceback
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
        
        allowed_statuses = ['paused', 'failed', 'error', 'resuming']
        if task.get('status') not in allowed_statuses:
            if task.get('status') == 'processing':
                if task_id not in task_manager.active_tasks:
                    pass
                else:
                    raise HTTPException(400, f"Task is already running")
            else:
                raise HTTPException(400, f"Cannot resume task with status '{task.get('status')}'")
        
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
    try:
        await connect_task_websocket(task_id, websocket)
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get('type')
                
                if msg_type == 'ping':
                    await websocket.send_json({'type': 'pong'})
                elif msg_type == 'validate_speakers':
                    await handle_speaker_validation(task_id, data.get('config', {}))
                elif msg_type == 'start_translation':
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
            await disconnect_task_websocket(task_id, websocket)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"WebSocket error for task {task_id}:\n{tb_str}")
            try:
                await websocket.send_json({'type': 'error', 'message': str(e), 'traceback': tb_str})
            except:
                pass
            await disconnect_task_websocket(task_id, websocket)
            
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"WebSocket connection failed for task {task_id}:\n{tb_str}")
        raise

async def handle_speaker_validation(task_id: str, config: Dict[str, Any]):
    try:
        valid_actions = {'translate', 'remove', 'dub'}
        for spk_id, info in config.items():
            if info.get('action') not in valid_actions:
                raise HTTPException(400, f"Invalid action for speaker {spk_id}")
        
        # CRITICAL FIX: Merge with existing speaker_config to preserve sample_path
        # The frontend may not send sample_path, so we need to preserve it from Phase 1
        existing_task = db.get_task(task_id)
        existing_config = existing_task.get('speaker_config', {}) if existing_task else {}
        
        # Merge configs: new values take precedence, but preserve sample_path and other fields
        merged_config = {}
        for spk_id_str, new_info in config.items():
            existing_info = existing_config.get(spk_id_str, {})
            # Start with existing info, then override with new info
            merged_info = dict(existing_info)
            merged_info.update(new_info)
            merged_config[spk_id_str] = merged_info
        
        # Log what we're preserving for debugging
        for spk_id_str, merged_info in merged_config.items():
            if 'sample_path' in merged_info:
                print(f"Preserved sample_path for speaker {spk_id_str}: {merged_info['sample_path']}")
            else:
                print(f"WARNING: No sample_path for speaker {spk_id_str}, will rely on fallback paths")
        
        # CRITICAL FIX: Retrieve and preserve language settings from existing task
        # This is the key fix - we MUST preserve these settings that were set during upload
        # The database stores 'target_language' but we need to read it properly
        tgt_lang = 'en'
        src_lang = 'auto'
        tts_engine = 'f5'
        separate_audio = False
        
        if existing_task:
            # Get the stored values - check both tgt_lang (API alias) and target_language (DB column)
            # The get_task method should populate tgt_lang from target_language
            tgt_lang = existing_task.get('tgt_lang') or existing_task.get('target_language') or 'en'
            src_lang = existing_task.get('src_lang') or 'auto'
            tts_engine = existing_task.get('tts_engine') or 'f5'
            separate_audio = existing_task.get('separate_audio', False)
            
            # Ensure we never have None values
            if not tgt_lang or str(tgt_lang).strip() == "":
                tgt_lang = 'en'
            if not src_lang or str(src_lang).strip() == "":
                src_lang = 'auto'
            if not tts_engine or str(tts_engine).strip() == "":
                tts_engine = 'f5'
        
        print(f"CRITICAL: handle_speaker_validation - preserving lang settings: src={src_lang}, tgt={tgt_lang}, engine={tts_engine}")
        
        db.update_task(
            task_id,
            speaker_config=merged_config,
            phase='translating',
            status='queued',
            was_running_at_shutdown=0,
            message='Speaker validation accepted, starting translation...',
            # CRITICAL: Explicitly preserve these settings in the update
            tgt_lang=tgt_lang,
            src_lang=src_lang,
            tts_engine=tts_engine,
            separate_audio=separate_audio
        )
        
        # Re-verify the task was updated correctly
        verify_task = db.get_task(task_id)
        print(f"CRITICAL: Post-validation verification - tgt_lang={verify_task.get('tgt_lang')}, target_language={verify_task.get('target_language')}, src_lang={verify_task.get('src_lang')}")
        
        confirmation_data = {
            'type': 'validation_accepted',
            'data': {
                'task_id': task_id,
                'status': 'queued',
                'phase': 'translating',
                'progress': 35,
                'message': 'Speaker validation accepted, starting translation...',
                'speaker_config': merged_config,
                'was_running_at_shutdown': False,
                'tgt_lang': tgt_lang,
                'src_lang': src_lang
            }
        }
        
        await broadcast_to_task(task_id, confirmation_data)
        
        auto_start = config.get('_auto_start', True)
        if auto_start:
            asyncio.create_task(task_manager.start_task(task_id))
            
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Speaker validation failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to process speaker validation: {str(e)}\n\nFull traceback in server logs")

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
async def delete_project(task_id: str):
    try:
        if task_id in task_manager.active_tasks:
            await task_manager.cancel_task(task_id)
        
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

@router.post("/api/projects/{task_id}/validate")
async def validate_speakers_api(task_id: str, update: SpeakerConfigUpdate):
    try:
        await handle_speaker_validation(task_id, update.speakers)
        return {"status": "accepted"}
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Validate speakers API failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to validate speakers: {str(e)}\n\nFull traceback in server logs")

@router.get("/api/projects/{task_id}/download")
async def download_result(task_id: str):
    try:
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
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Download result failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to download result: {str(e)}\n\nFull traceback in server logs")

@router.get("/api/projects/{task_id}/preview/{segment_idx}")
async def preview_segment_audio(task_id: str, segment_idx: int):
    try:
        translations = db.get_translation_segments(task_id)
        trans = next((t for t in translations if t['segment_idx'] == segment_idx), None)
        
        if trans and trans.get('audio_path') and Path(trans['audio_path']).exists():
            return FileResponse(trans['audio_path'], media_type="audio/wav")
        
        task = db.get_task(task_id)
        if task and task.get('segments') and segment_idx < len(task['segments']):
            seg_path = task['segments'][segment_idx].get('audio_path')
            if seg_path and Path(seg_path).exists():
                return FileResponse(seg_path, media_type="audio/wav")
        
        raise HTTPException(404, "Audio not found")
        
    except HTTPException:
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Preview segment audio failed with traceback:\n{tb_str}")
        raise HTTPException(500, f"Failed to preview audio: {str(e)}\n\nFull traceback in server logs")
