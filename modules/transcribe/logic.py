"""
Enhanced transcription logic with optional speaker diarization.
"""

import torch
import json
import soundfile as sf
import os
import traceback
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from core.resources import manager

# Import diarization components from translate_video pipeline
from modules.translate_video.pipeline.phase1_diarization import (
    extract_audio, run_vad, SpeakerIdentifier, 
    transcribe_segments, extract_speaker_samples
)


def whisper_inference(
    audio_np: np.ndarray, 
    src_lang: str = "auto", 
    whisper_model: Optional[str] = None
) -> str:
    """Core shared Whisper inference function with error handling and VRAM cleanup."""
    import gc
    try:
        whisper_pipe = manager.get_whisper(whisper_model)
        model = whisper_pipe.model
        tokenizer = whisper_pipe.tokenizer
        
        if hasattr(whisper_pipe, 'feature_extractor'):
            feature_extractor = whisper_pipe.feature_extractor
        else:
            from transformers import AutoFeatureExtractor
            feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v2")
            
        device = manager.device
        model_dtype = next(model.parameters()).dtype

        # Process with feature extractor
        inputs = feature_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device).to(model_dtype)
        
        generate_kwargs = {}
        if src_lang and src_lang != 'auto':
            generate_kwargs["language"] = src_lang
        
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=1,
                condition_on_prev_tokens=False,
                **generate_kwargs
            )
        
        return tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    finally:
        # Aggressive memory cleanup after every GPU call
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

async def process_transcription(file_path: str, translate_to: Optional[str] = None):
    """Refactored simple transcription with robust format detection."""
    import librosa
    
    # Use librosa for duration (more robust than soundfile for varying formats)
    try:
        duration = librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"[Transcribe] Librosa duration check failed, trying fallback: {e}")
        # Final fallback for duration
        info = sf.info(file_path)
        duration = info.duration

    if duration > 30:
        return await process_long_audio(file_path, translate_to)
    
    # Load with librosa to handle non-standard WAV/MP3/MP4 containers
    audio_np, sr = librosa.load(file_path, sr=16000, mono=True)
    if len(audio_np.shape) > 1: audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    
    text = whisper_inference(audio_np.astype(np.float32))
    
    result = {
        "original_text": text,
        "text": text,
        "duration": duration,
        "filename": Path(file_path).name
    }
    if translate_to:
        result["translated_text"] = await translate_text(text, translate_to)
    return result


async def process_long_audio(file_path: str, translate_to: Optional[str] = None):
    """Process long audio using silence-aware chunking and aggressive VRAM management."""
    import subprocess
    import gc
    from modules.translate_video.pipeline.phase1_diarization import run_vad
    
    info = sf.info(file_path)
    total_duration = info.duration
    
    # 1. Use VAD to find natural silence gaps as split points
    speech_regions = run_vad(file_path)
    
    full_text = []
    all_segments = []
    
    # Group speech regions into max 30s blocks
    blocks = []
    current_block = []
    block_duration = 0
    
    for seg in speech_regions:
        seg_dur = seg['end'] - seg['start']
        if block_duration + seg_dur > 30 and current_block:
            blocks.append((current_block[0]['start'], current_block[-1]['end']))
            current_block = [seg]
            block_duration = seg_dur
        else:
            current_block.append(seg)
            block_duration += seg_dur
    
    if current_block:
        blocks.append((current_block[0]['start'], current_block[-1]['end']))

    # 2. Process natural blocks
    for i, (start, end) in enumerate(blocks):
        chunk_path = f"temp_chunks/trans_chunk_{i}.wav"
        Path(chunk_path).parent.mkdir(exist_ok=True)
        
        subprocess.run([
            "ffmpeg", "-y", "-i", file_path,
            "-ss", str(start), "-t", str(end - start),
            "-ar", "16000", "-ac", "1",
            chunk_path
        ], capture_output=True)
        
        # Transcribe and Force Cleanup
        chunk_result = await process_transcription(chunk_path, translate_to=None)
        
        # OOM Protection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Re-index segments
        if "segments" in chunk_result:
            for seg in chunk_result["segments"]:
                seg["start"] += start
                seg["end"] += start
                all_segments.append(seg)
        
        full_text.append(chunk_result["original_text"])
        
        try: os.remove(chunk_path)
        except: pass
    
    result = {
        "original_text": " ".join(full_text),
        "text": " ".join(full_text),
        "duration": total_duration,
        "segments": all_segments,
        "filename": Path(file_path).name
    }
    
    if translate_to:
        result["translated_text"] = await translate_text(result["original_text"], translate_to)
    
    return result


async def run_transcription_stage_1(
    file_path: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    task_id: str = ""
) -> Dict[str, Any]:
    """Stage 1: Diarization & Samples."""
    orig_name = Path(file_path).name
    if "_" in orig_name and len(orig_name.split("_")[0]) > 30: # Check if it has a UUID prefix
        orig_name = "_".join(orig_name.split("_")[1:])
    import functools
    loop = asyncio.get_running_loop()
    
    async def report(pct: int, msg: str):
        if progress_callback: await progress_callback(pct, msg)

    await report(5, "Extracting audio...")
    temp_dir = Path("temp_chunks") / task_id / "transcribe"
    temp_dir.mkdir(parents=True, exist_ok=True)
    master_wav = str(temp_dir / "master.wav")
    await loop.run_in_executor(None, functools.partial(extract_audio, file_path, master_wav))
    
    await report(15, "Identifying speakers...")
    identifier = SpeakerIdentifier(max_speakers=10)
    
    def diar_progress_sync(msg, pct, total):
        asyncio.run_coroutine_threadsafe(report(15 + int(pct * 0.25), msg), loop)
    
    segments, assignments, num_speakers = await loop.run_in_executor(
        None, functools.partial(identifier.run, master_wav, diar_progress_sync)
    )
    
    await report(40, "Extracting speaker samples...")
    speaker_samples, speaker_config = extract_speaker_samples(master_wav, segments, task_id)
    
    return {
        "master_wav": master_wav,
        "segments": segments,
        "speaker_config": speaker_config,
        "num_speakers": num_speakers,
        "original_filename": orig_name
    }

async def run_transcription_stage_2(
    task_id: str,
    interim_data: Dict[str, Any],
    speaker_config: Dict[str, Any],
    src_lang: str = "auto",
    tgt_lang: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """Stage 2: Apply Merge/Rename and Transcribe."""
    import functools
    loop = asyncio.get_running_loop()
    
    async def report(pct: int, msg: str):
        if progress_callback: await progress_callback(pct, msg)

    segments = interim_data["segments"]
    master_wav = interim_data["master_wav"]
    
    # 1. Apply Merges and Actions
    merge_map = {}
    for sid, info in speaker_config.items():
        if info.get('merged_into'):
            merge_map[int(sid)] = int(info['merged_into'])

    # Filter segments and update speaker IDs
    final_segments = []
    for seg in segments:
        sid = seg.speaker_id
        # Apply merge
        effective_id = merge_map.get(sid, sid)
        # Check if master or effective speaker is 'remove'
        action = speaker_config.get(str(effective_id), {}).get('action', 'dub')
        if action == 'remove' or not speaker_config.get(str(sid), {}).get('include', True):
            continue
            
        seg.speaker_id = effective_id
        final_segments.append(seg)

    # 2. Transcribe
    await report(50, "Transcribing segments...")
    def trans_progress(cur, tot):
        asyncio.run_coroutine_threadsafe(report(50 + int((cur/tot) * 35), f"Transcribed {cur}/{tot}"), loop)

    final_segments = await loop.run_in_executor(
        None, functools.partial(transcribe_segments, master_wav, final_segments, src_lang, trans_progress)
    )

    # 3. Finalize
    full_text = []
    for seg in final_segments:
        spk_name = speaker_config.get(str(seg.speaker_id), {}).get('name', f"Speaker {seg.speaker_id+1}")
        full_text.append(f"[{spk_name}]: {seg.original_text}")

    # Calculate duration for metadata
    import soundfile as sf
    duration = 0
    try:
        duration = sf.info(master_wav).duration
    except:
        pass

    result = {
        "original_text": "\n".join(full_text),
        "filename": interim_data.get("original_filename", "audio.wav"),
        "segments": [s.to_dict() for s in final_segments],
        "speakers": speaker_config,
        "duration": duration
    }

    if tgt_lang:
        await report(90, "Translating transcript...")
        # (Translation logic remains same as previous version but applied to final_segments)
        translator = manager.get_lollms_client()
        if translator:
            trans_full = []
            for seg in final_segments:
                prompt = f"Translate to {tgt_lang}: {seg.original_text}"
                seg.translated_text = translator.generate_text(prompt).strip()
                spk_name = speaker_config.get(str(seg.speaker_id), {}).get('name', f"Speaker {seg.speaker_id+1}")
                trans_full.append(f"[{spk_name}]: {seg.translated_text}")
            result["translated_text"] = "\n".join(trans_full)

    await report(100, "Complete!")
    return result


async def translate_text(text: str, target_lang: str) -> str:
    """Translate text using LoLLMs."""
    try:
        lc = manager.get_lollms_client()
        if not lc:
            return None
        
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'zh': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'ru': 'Russian'
        }
        
        lang_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"""Translate the following text to {lang_name}. 
Preserve the meaning, tone, and style. Only output the translation, no explanations.

Text: {text}

Translation to {lang_name}:"""
        
        result = lc.generate_text(prompt, temperature=0.3)
        return result.strip()
        
    except Exception as e:
        print(f"Translation failed: {e}")
        return None