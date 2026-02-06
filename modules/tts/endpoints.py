# modules/tts/endpoints.py
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from .logic import generate_speech
import shutil, uuid, soundfile as sf
from pathlib import Path

router = APIRouter(prefix="/tts", tags=["TTS"])
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

@router.post("/generate")
async def tts_generate(
    text: str = Form(...),
    reffile: UploadFile = File(...),
    engine: str = Form("f5"),  # NEW: "f5" | "fishspeech"
):
    task_id = str(uuid.uuid4())
    ref_path = UPLOAD_DIR / f"{task_id}_ref_{reffile.filename}"
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reffile.file, f)

    audio, sr = generate_speech(text, str(ref_path), engine=engine)
    out_path = OUTPUT_DIR / f"{task_id}_tts.wav"
    sf.write(out_path, audio, sr)
    return FileResponse(out_path, media_type="audio/wav", filename="generated.wav")
