from fastapi import APIRouter, UploadFile, File, Form
from .logic import analyze_media
import shutil
import uuid
from pathlib import Path

router = APIRouter(prefix="/analysis", tags=["Analysis"])
UPLOAD_DIR = Path("uploads")

@router.post("/")
async def analyze_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("summary") # summary or meeting_report
):
    task_id = str(uuid.uuid4())
    fpath = UPLOAD_DIR / f"{task_id}_{file.filename}"
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    result = await analyze_media(str(fpath), mode)
    return result
