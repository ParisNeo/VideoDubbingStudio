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
    engine: str = Form("f5"),  # "f5" | "fishspeech" | "lollms"
):
    """
    Generate speech using voice cloning with specified TTS engine.
    
    Supports F5-TTS (local), FishSpeech (local), and LoLLMs (API-based with voice cloning).
    All engines use the uploaded reference audio file for voice cloning.
    """
    task_id = str(uuid.uuid4())
    ref_path = UPLOAD_DIR / f"{task_id}_ref_{reffile.filename}"
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reffile.file, f)

    # Use the unified generate_speech function which handles all engines
    audio, sr = generate_speech(
        text=text,
        ref_audio_path=str(ref_path),
        engine=engine,
        response_format="wav" if engine == "lollms" else None,  # Use WAV for LoLLMs
    )
    
    out_path = OUTPUT_DIR / f"{task_id}_tts.wav"
    sf.write(out_path, audio, sr)
    
    return FileResponse(out_path, media_type="audio/wav", filename="generated.wav")
