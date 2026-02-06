from core.resources import manager
import os

async def analyze_media(file_path: str, mode: str = "summary"):
    # 1. Transcribe First
    stt = manager.get_whisper()
    res = stt(file_path)
    text = res["text"]
    
    # 2. LLM Processing
    lc = manager.get_lollms()
    if not lc: return {"error": "LLM not available"}
    
    if mode == "summary":
        system = "You are an expert summarizer. Summarize the following text concisely."
    elif mode == "meeting_report":
        system = "You are a secretary. Create a structured meeting report with Agenda, Key Points, and Action Items."
    else:
        return {"error": "Unknown mode"}
        
    response = lc.generate_text(f"{system}\n\nTEXT:\n{text}")
    return {"original_text": text, "analysis": response}
