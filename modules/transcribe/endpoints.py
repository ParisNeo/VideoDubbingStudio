from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from .logic import process_transcription
import shutil
import uuid
from pathlib import Path

router = APIRouter(prefix="/transcribe", tags=["Transcribe"])
UPLOAD_DIR = Path("uploads")

@router.post("/")
async def transcribe_media(
    file: UploadFile = File(...),
    translate_to: str = Form(None)
):
    task_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    # Process immediately (or use background tasks for async)
    result = await process_transcription(str(file_path), translate_to)
    
    return {"task_id": task_id, "result": result}
