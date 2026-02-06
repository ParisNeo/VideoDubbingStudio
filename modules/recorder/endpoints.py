from fastapi import APIRouter, UploadFile, File
import shutil
import uuid
from pathlib import Path

router = APIRouter(prefix="/recorder", tags=["Recorder"])
UPLOAD_DIR = Path("uploads")

@router.post("/save")
async def save_recording(file: UploadFile = File(...)):
    # Save blob from frontend
    task_id = str(uuid.uuid4())
    # Frontend usually sends 'blob' as filename, detect ext
    ext = ".webm" 
    if file.content_type == "audio/wav": ext = ".wav"
    if file.content_type == "video/mp4": ext = ".mp4"
    
    filename = f"rec_{task_id}{ext}"
    fpath = UPLOAD_DIR / filename
    
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    return {"file_path": str(fpath), "filename": filename, "task_id": task_id}
