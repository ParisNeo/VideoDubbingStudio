from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from .logic import process_transcription, run_transcription_stage_1, run_transcription_stage_2
import shutil
import uuid
import traceback
from pathlib import Path
from typing import Optional

from core.database import db

router = APIRouter(prefix="/transcribe", tags=["Transcribe"])
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp_chunks")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# Store active transcription tasks
active_tasks = {}

@router.post("/upload")
async def upload_for_transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    src_lang: str = Form("auto"),
    tgt_lang: Optional[str] = Form(None),
    use_diarization: str = Form("false"),
    whisper_model: str = Form("large-v2"),
    vad_threshold: float = Form(0.20)
):
    try:
        task_id = str(uuid.uuid4())
        do_diarization = use_diarization.lower() in ('true', '1', 'yes', 'on')
        
        file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 1. Register in Database
        db.create_task(task_id, file.filename, str(file_path))
        db.update_task(
            task_id,
            source='transcribe',
            src_lang=src_lang,
            tgt_lang=tgt_lang or "",
            whisper_model=whisper_model,
            vad_threshold=vad_threshold,
            status='queued',
            phase='identifying' if do_diarization else 'transcribing'
        )

        # 2. CRITICAL: Initialize task in active_tasks memory for WebSocket sync
        active_tasks[task_id] = {
            "task_id": task_id,
            "file_path": str(file_path),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "status": "queued",
            "progress": 0,
            "speakers_confirmed": False
        }
        
        # 3. Start background worker
        if do_diarization:
            background_tasks.add_task(process_with_diarization_task, task_id)
        else:
            background_tasks.add_task(process_simple_task, task_id)
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Transcription task registered and starting..."
        }
        
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Transcription upload failed:\n{tb_str}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@router.get("/preview/{task_id}/{segment_idx}")
async def preview_transcription_segment(task_id: str, segment_idx: int):
    """Stream a specific segment of the master audio for review."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    
    # Check if we have the master audio path
    master_wav = task.get("master_audio")
    if not master_wav:
        # Fallback for simple transcription (no diarization)
        master_wav = task.get("file_path")
        
    segments = task.get("segments", [])
    
    if not segments or segment_idx >= len(segments):
        raise HTTPException(404, "Segment not found")
        
    seg = segments[segment_idx]
    start, duration = seg["start"], seg["end"] - seg["start"]
    
    # Use FFmpeg to extract the segment to a buffer
    cmd = [
        "ffmpeg", "-ss", str(start), "-t", str(duration),
        "-i", master_wav, "-f", "wav", "pipe:1"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    
    from fastapi.responses import Response
    return Response(content=stdout, media_type="audio/wav")

@router.get("/status/{task_id}")
async def get_transcription_status(task_id: str):
    """Get current status of a transcription task from the database."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found in database")
    
    return {
        "task_id": task_id,
        "status": task.get("status"),
        "progress": task.get("progress", 0),
        "message": task.get("message", ""),
        "original_text": task.get("original_text"),
        "translated_text": task.get("translated_text"),
        "speakers": task.get("speaker_config"),
        "segments": task.get("segments"),
        "filename": task.get("filename"),
        "duration": task.get("duration")
    }

@router.websocket("/ws/{task_id}")
async def transcription_websocket(websocket: WebSocket, task_id: str):
    """WebSocket for real-time transcription updates with diarization."""
    await websocket.accept()
    
    # Try to recover task from database if not in memory (e.g. page refresh)
    if task_id not in active_tasks:
        db_task = db.get_task(task_id)
        if db_task and db_task.get('source') == 'transcribe':
            active_tasks[task_id] = {
                "task_id": task_id,
                "file_path": db_task.get('file_path'),
                "src_lang": db_task.get('src_lang'),
                "tgt_lang": db_task.get('tgt_lang'),
                "status": db_task.get('status'),
                "progress": db_task.get('progress', 0),
                "speakers_confirmed": False
            }
        else:
            await websocket.send_json({"type": "error", "message": "Task not found"})
            await websocket.close()
            return
    
    task = active_tasks[task_id]
    task["websocket"] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "confirm_speakers":
                # User confirmed speaker config
                task["speaker_config"] = data.get("speakers", {})
                task["speakers_confirmed"] = True
                
    except WebSocketDisconnect:
        print(f"Transcription WebSocket disconnected for {task_id}")
    except Exception as e:
        print(f"Transcription WebSocket error: {e}")
    finally:
        if task_id in active_tasks:
            active_tasks[task_id].pop("websocket", None)

async def process_simple_task(task_id: str):
    """Simple transcription without diarization."""
    task = db.get_task(task_id)
    if not task: return
    
    try:
        db.update_task(task_id, status="processing", progress=10, message="Transcribing audio...")
        
        result = await process_transcription(
            task["file_path"],
            translate_to=task.get("tgt_lang")
        )
        
        db.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Transcription complete",
            original_text=result.get("original_text"),
            translated_text=result.get("translated_text"),
            segments=result.get("segments"),
            duration=result.get("duration")
        )
        
    except Exception as e:
        task["status"] = "failed"
        task["error_message"] = str(e)
        print(f"Simple transcription failed: {e}")

async def process_with_diarization_task(task_id: str):
    """Transcription with speaker diarization."""
    from .logic import run_transcription_stage_1, run_transcription_stage_2
    import asyncio
    
    task = active_tasks.get(task_id)
    if not task: return
    
    async def send_progress(percent, message):
        task["progress"] = percent
        task["message"] = message
        ws = task.get("websocket")
        if ws:
            try: await ws.send_json({"type": "progress", "data": {"percent": percent, "message": message}})
            except: pass

    try:
        # STAGE 1
        db.update_task(task_id, status="processing", message="Running Stage 1...")
        interim = await run_transcription_stage_1(
            task["file_path"], 
            progress_callback=send_progress, 
            task_id=task_id
        )
        # Store metadata in DB so status/preview can find it
        db.update_task(task_id, 
            master_audio=interim["master_wav"], 
            speaker_config=interim["speaker_config"]
        )
        task["interim_data"] = interim

        # WAIT FOR UI CONFIRMATION
        ws = task.get("websocket")
        if ws:
            await ws.send_json({
                "type": "speakers_detected",
                "data": {"speakers": interim["speaker_config"], "count": interim["num_speakers"]}
            })
        
        # Block until flag set by websocket handler
        while not task.get("speakers_confirmed"):
            await asyncio.sleep(0.5)

        # STAGE 2
        result = await run_transcription_stage_2(
            task_id,
            interim,
            task["speaker_config"],
            src_lang=task.get("src_lang", "auto"),
            tgt_lang=task.get("tgt_lang"),
            progress_callback=send_progress
        )
        
        task["result"] = result
        task["status"] = "completed"
        task["progress"] = 100
        task["message"] = "Transcription complete"
        
        if ws:
            try:
                await ws.send_json({
                    "type": "transcription_complete",
                    "data": result
                })
            except:
                pass
        
    except Exception as e:
        task["status"] = "failed"
        task["error_message"] = str(e)
        print(f"Diarization transcription failed: {e}")
        
        if ws:
            try:
                await ws.send_json({
                    "type": "error",
                    "data": {"message": str(e)}
                })
            except:
                pass