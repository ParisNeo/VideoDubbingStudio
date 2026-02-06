import torch
import json
import soundfile as sf
import os
from pathlib import Path
from core.resources import manager

async def process_transcription(file_path: str, translate_to: str = None):
    # Load Whisper
    stt = manager.get_whisper()
    
    # 1. Transcribe
    result = stt(file_path, return_timestamps=True, generate_kwargs={"language": "english"}) # Or auto
    chunks = result.get("chunks", [])
    
    transcript = []
    full_text = result.get("text", "")
    
    # 2. Optional Text Translation (using Lollms)
    lc = manager.get_lollms()
    translated_text = ""
    
    for chunk in chunks:
        text = chunk["text"]
        trans_text = text
        
        if translate_to and lc:
            # Simple chunk translation
            try:
                prompt = f"Translate this to {translate_to}: {text}"
                resp = lc.generate_text(prompt)
                trans_text = resp.strip()
            except:
                pass
                
        transcript.append({
            "start": chunk["timestamp"][0],
            "end": chunk["timestamp"][1],
            "text": text,
            "translated": trans_text
        })
        translated_text += trans_text + " "

    return {
        "original_text": full_text,
        "translated_text": translated_text if translate_to else None,
        "segments": transcript
    }
