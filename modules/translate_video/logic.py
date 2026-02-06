import asyncio
import json
import soundfile as sf
import subprocess
import traceback
import logging
from pathlib import Path
from typing import Dict, Any
import concurrent.futures
import numpy as np
import torch
import tempfile
import os
import warnings

from .diarization import LowVRAMDiarizer
from .state import active_connections
from .project_manager import ProjectManager, append_log

# Import new modular pipeline
from .pipeline.phase2_translation import run_phase2_translation

# Setup logger for this module
logger = logging.getLogger("translate_video.logic")

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
CHUNKS_DIR = Path("temp_chunks")
[d.mkdir(exist_ok=True, parents=True) for d in [UPLOAD_DIR, OUTPUT_DIR, CHUNKS_DIR]]

# Thread pool for CPU-bound diarization
_diarization_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# -------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------

async def log_ws(task_id: str, msg: str, style: str = "info"):
    """Log to WebSocket AND Disk Persistence."""
    # 1. Disk
    append_log(task_id, msg, style)
    
    # 2. WebSocket (Live)
    if task_id in active_connections:
        try:
            # Iterate over all connections for this task (active_connections[task_id] is a Set)
            dead_sockets = set()
            for ws in active_connections[task_id]:
                try:
                    await ws.send_text(json.dumps({
                        "type": "log",
                        "data": {"message": msg, "style": style}
                    }))
                except Exception:
                    dead_sockets.add(ws)
            
            # Clean up dead connections
            for ws in dead_sockets:
                active_connections[task_id].discard(ws)
                
        except Exception as e:
            logger.warning(f"Failed to send log: {e}")

async def update_status(task_id: str, updates: dict):
    """Update status to WebSocket AND Disk Persistence."""
    # 1. Disk (JSON file for project manager - primary storage)
    ProjectManager.save_state(task_id, updates)
    
    # 1b. Also update SQLite database for API consistency after page refresh
    # This ensures that db.get_task() returns current state
    from core.database import db
    try:
        db.update_task(task_id, **updates)
    except Exception as e:
        logger.warning(f"Failed to update SQLite for {task_id}: {e}")
        # Don't fail the whole update if SQLite fails, JSON is primary
    
    # 2. WebSocket
    state = ProjectManager.get_state(task_id)
    if task_id in active_connections and state:
        try:
            # Iterate over all connections for this task
            dead_sockets = set()
            for ws in active_connections[task_id]:
                try:
                    await ws.send_text(json.dumps({
                        "type": "status_update", "data": state
                    }))
                except Exception:
                    dead_sockets.add(ws)
            
            # Clean up dead connections
            for ws in dead_sockets:
                active_connections[task_id].discard(ws)
                
        except Exception as e:
            logger.warning(f"Failed to send status update: {e}")

async def progress_ws(task_id: str, phase: str, percent: int, message: str):
    """Send progress update via WebSocket."""
    if task_id in active_connections:
        try:
            # Iterate over all connections for this task (active_connections[task_id] is a Set)
            dead_sockets = set()
            for ws in active_connections[task_id]:
                try:
                    await ws.send_text(json.dumps({
                        "type": "progress",
                        "data": {
                            "phase": phase,
                            "percent": percent,
                            "message": message
                        }
                    }))
                except Exception:
                    dead_sockets.add(ws)
            
            # Clean up dead connections
            for ws in dead_sockets:
                active_connections[task_id].discard(ws)
                
        except Exception as e:
            logger.warning(f"Failed to send progress: {e}")

async def broadcast_transcription_update(task_id: str, segments_with_transcription: list):
    """Broadcast transcription data to update the UI with speech + diarization."""
    if task_id not in active_connections:
        return
    
    try:
        dead_sockets = set()
        for ws in active_connections[task_id]:
            try:
                await ws.send_text(json.dumps({
                    "type": "transcription_update",
                    "data": {
                        "segments": segments_with_transcription
                    }
                }))
            except Exception:
                dead_sockets.add(ws)
        
        # Clean up dead connections
        for ws in dead_sockets:
            active_connections[task_id].discard(ws)
            
    except Exception as e:
        logger.warning(f"Failed to broadcast transcription update: {e}")

# -------------------------------------------------------------------------
# CROSS-PLATFORM ASYNC SUBPROCESS HELPER
# -------------------------------------------------------------------------

async def run_subprocess(cmd: list, **kwargs) -> tuple:
    """Cross-platform async subprocess execution."""
    loop = asyncio.get_event_loop()
    
    def _run_sync():
        return subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            **kwargs
        )
    
    result = await loop.run_in_executor(None, _run_sync)
    return result.returncode, result.stdout, result.stderr

# -------------------------------------------------------------------------
# PHASE 1: IDENTIFICATION (NOW WITH TRANSCRIPTION)
# -------------------------------------------------------------------------

async def start_identification_task(task_id: str, video_path: str):
    """Phase 1: Extract audio, diarize, AND transcribe for immediate UI display."""
    await log_ws(task_id, f"Phase 1: Analyzing {Path(video_path).name}", "step")
    await update_status(task_id, {"status": "identifying", "progress": 5, "video_path": video_path})

    try:
        # 1. Audio Extraction
        await log_ws(task_id, "Extracting audio track...", "substep")
        await progress_ws(task_id, "audio_extraction", 5, "Extracting audio with FFmpeg...")
        master_wav = CHUNKS_DIR / f"{task_id}_master.wav"
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", str(master_wav)
        ]
        
        returncode, stdout, stderr = await run_subprocess(cmd)
        
        if returncode != 0:
            stderr_text = stderr.decode('utf-8', errors='replace') if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed with code {returncode}: {stderr_text}")

        await log_ws(task_id, "Audio extracted successfully", "success")
        await progress_ws(task_id, "audio_extraction", 10, "Audio extracted")

        # 2. Diarization with live progress reporting
        await log_ws(task_id, "Running speaker diarization...", "substep")
        
        # Check GPU
        if torch.cuda.is_available():
            await log_ws(task_id, f"GPU detected: {torch.cuda.get_device_name(0)}", "info")
        else:
            await log_ws(task_id, "WARNING: No GPU detected, using CPU (slow)", "warning")
        
        # Read audio in main thread
        audio_np, sr = sf.read(str(master_wav))
        
        # Create a queue to receive progress from the worker thread
        import queue
        progress_queue = queue.Queue()
        
        def run_diarization_with_progress():
            """Run diarization and put progress updates in queue."""
            def on_progress(msg, pct, total):
                progress_queue.put(("progress", msg, pct))
            
            diarizer = LowVRAMDiarizer(device=None)
            result = diarizer.run(str(master_wav), min_speech_duration=0.5,
                                progress_callback=on_progress)
            
            progress_queue.put(("done", result))
            return result
        
        # Start diarization in executor
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_diarization_executor, run_diarization_with_progress)
        
        # Poll for progress while waiting
        last_progress = 0
        while True:
            # Check if future is done
            if future.done():
                try:
                    result = future.result()
                    break
                except Exception as e:
                    raise e
            
            # Drain progress queue
            try:
                while True:
                    item = progress_queue.get_nowait()
                    if item[0] == "progress":
                        _, msg, pct = item
                        # Map 0-100 diarization progress to 10-30 overall
                        overall_pct = 10 + int(pct * 0.20)
                        if overall_pct > last_progress:
                            last_progress = overall_pct
                            await progress_ws(task_id, "diarization", overall_pct, msg)
                            # Also update main progress bar
                            await update_status(task_id, {
                                "progress": overall_pct,
                                "message": msg
                            })
            except queue.Empty:
                pass
            
            # Small sleep to prevent busy-waiting
            await asyncio.sleep(0.1)
        
        # Final progress
        await progress_ws(task_id, "diarization", 30, "Diarization complete")
        
        # Log timing info
        if 'timing' in result:
            t = result['timing']
            await log_ws(task_id, f"Timing: VAD={t['vad_seconds']:.1f}s, "
                       f"Embeds={t['embed_seconds']:.1f}s, "
                       f"Cluster={t['cluster_seconds']:.1f}s, "
                       f"Total={t['total_seconds']:.1f}s", "info")
        
        assignments = result.get("assignments", {})
        segments = result.get("segments", [])
        
        # Add index to segments for easier reference
        for i, seg in enumerate(segments):
            seg['idx'] = i
        
        # 3. TRANSCRIBE ALL SEGMENTS FOR IMMEDIATE UI DISPLAY
        await log_ws(task_id, "Transcribing speech segments for preview...", "substep")
        await progress_ws(task_id, "transcription", 30, "Transcribing speech with Whisper...")
        
        segments_with_transcription = await transcribe_segments_for_preview(
            task_id, str(master_wav), segments, audio_np, sr
        )
        
        # Broadcast transcription to UI immediately
        await broadcast_transcription_update(task_id, segments_with_transcription)
        await progress_ws(task_id, "transcription", 35, "Transcription complete - review below")
        
        # 4. Generate Speaker Config (with samples)
        await log_ws(task_id, "Generating reference samples...", "substep")
        
        speaker_config = {}
        unique = set(assignments.values())
        
        for spk_id in unique:
            best_idx = -1
            max_len = 0
            
            for seg_idx_raw, owner in assignments.items():
                if owner == spk_id:
                    seg_idx = int(seg_idx_raw)
                    if seg_idx >= len(segments): continue
                    
                    l = segments[seg_idx]["end"] - segments[seg_idx]["start"]
                    if l > max_len: 
                        max_len = l
                        best_idx = seg_idx
            
            if best_idx != -1:
                seg = segments[best_idx]
                sample_path = CHUNKS_DIR / f"{task_id}_spk_{spk_id}.wav"
                
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                chunk = audio_np[start_sample:end_sample]
                sf.write(str(sample_path), chunk, sr)
                
                speaker_config[str(spk_id)] = {
                    "name": f"Speaker {int(spk_id)+1}",
                    "action": "dub",
                    "sample_path": f"/temp_chunks/{sample_path.name}"
                }

        await log_ws(task_id, f"Found {len(unique)} speakers.", "success")
        
        # 5. Final Save & Trigger UI
        # Save to both JSON (ProjectManager) and SQLite (db) for consistency
        ProjectManager.save_state(task_id, {
            "segments": segments,
            "assignments": assignments,
            "master_audio": str(master_wav),
            "transcribed_segments": segments_with_transcription  # Store for Phase 2 reference
        })
        
        # Also save to SQLite for API consistency
        from core.database import db
        try:
            db.update_task(task_id, 
                segments=segments, 
                assignments=assignments, 
                master_audio=str(master_wav),
                transcribed_segments=segments_with_transcription
            )
        except Exception as e:
            logger.warning(f"Failed to save segments to SQLite for {task_id}: {e}")

        # FIXED: Use 'awaiting_validation' consistently
        await update_status(task_id, {
            "status": "awaiting_validation", 
            "phase": "awaiting_validation",
            "progress": 35,
            "speaker_config": speaker_config,
            "transcribed_segments": segments_with_transcription  # Include in status for UI
        })
        
        # Also broadcast speaker samples explicitly to trigger validation UI
        if task_id in active_connections:
            speaker_data = {}
            for spk_id, info in speaker_config.items():
                # Read audio file and encode as base64 for direct playback
                try:
                    import base64
                    sample_path = CHUNKS_DIR / f"{task_id}_spk_{spk_id}.wav"
                    if sample_path.exists():
                        with open(sample_path, 'rb') as f:
                            audio_bytes = f.read()
                            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        speaker_data[spk_id] = {
                            "audio_base64": audio_b64,
                            "default_name": info["name"],
                            "sample_rate": 16000
                        }
                except Exception as e:
                    logger.warning(f"Failed to encode speaker sample {spk_id}: {e}")
            
            if speaker_data:
                try:
                    dead_sockets = set()
                    for ws in active_connections[task_id]:
                        try:
                            await ws.send_text(json.dumps({
                                "type": "speaker_samples",
                                "data": speaker_data
                            }))
                        except Exception:
                            dead_sockets.add(ws)
                    for ws in dead_sockets:
                        active_connections[task_id].discard(ws)
                except Exception as e:
                    logger.warning(f"Failed to send speaker_samples: {e}")

    except Exception as e:
        traceback.print_exc()
        await log_ws(task_id, f"Error: {e}", "error")
        await update_status(task_id, {"status": "failed", "message": str(e)})


async def transcribe_segments_for_preview(task_id: str, audio_path: str, segments: list, 
                                          audio_np: np.ndarray, sr: int) -> list:
    """
    Transcribe all segments using Whisper for immediate UI display.
    Returns list of segments with original_text populated.
    
    ULTRA-ROBUST VERSION: Uses the raw model.generate() method to avoid pipeline issues on Windows.
    This completely bypasses the HuggingFace pipeline's input processing which causes WinError 87.
    
    CRITICAL FIX: Properly handles dtype mismatch between float32 inputs and float16 (half) model.
    """
    from core.resources import manager
    import torch
    
    segments_with_text = []
    total = len(segments)
    
    # Get the raw model and processor from the manager instead of the pipeline
    # We'll use direct model inference to avoid the pipeline's problematic input handling
    try:
        # Access the underlying model from the cached pipeline
        whisper_pipe = manager.get_whisper()
        # Get the model and processor from the pipeline
        model = whisper_pipe.model
        processor = whisper_pipe.tokenizer  # or whisper_pipe.feature_extractor
        
        # We need the feature extractor from the processor
        if hasattr(whisper_pipe, 'feature_extractor'):
            feature_extractor = whisper_pipe.feature_extractor
        else:
            # Fallback: load feature extractor directly
            from transformers import AutoFeatureExtractor
            feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v2")
        
        device = manager.device
        
        # CRITICAL: Get the model's dtype and ensure inputs match
        model_dtype = next(model.parameters()).dtype
        logger.info(f"Whisper model dtype: {model_dtype}, device: {device}")
        
        for i, seg in enumerate(segments):
            try:
                # Extract audio segment
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                chunk = audio_np[start_sample:end_sample]
                
                # Ensure mono and correct shape
                if len(chunk.shape) > 1:
                    chunk = chunk.mean(axis=1)
                
                # Resample to 16kHz if needed (Whisper expects 16kHz)
                if sr != 16000:
                    import librosa
                    chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
                
                # CRITICAL FIX: Convert audio to float32 for feature extractor
                # The feature extractor outputs float32, but model might be float16
                chunk = chunk.astype(np.float32)
                
                # Process audio with feature extractor
                inputs = feature_extractor(
                    chunk, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                )
                
                # CRITICAL FIX: Convert input_features to match model dtype
                input_features = inputs.input_features.to(device).to(model_dtype)
                
                # Generate with the model directly
                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_features,
                        max_length=448,
                        num_beams=1,
                        condition_on_prev_tokens=False,
                    )
                
                # Decode
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                text = transcription.strip()
                
                # Create enriched segment
                seg_with_text = {
                    "idx": seg.get("idx", i),
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker_id": seg.get("speaker_id", 0),
                    "original_text": text,
                    "translated_text": "",  # Will be filled in Phase 2
                    "status": "transcribed"
                }
                segments_with_text.append(seg_with_text)
                
                # Progress update every few segments
                if i % 5 == 0 or i == total - 1:
                    progress_pct = 30 + int((i / total) * 5)  # 30-35% range
                    await progress_ws(task_id, "transcription", progress_pct, 
                        f"Transcribed {i+1}/{total} segments...")
                    
            except Exception as e:
                logger.error(f"Failed to transcribe segment {i}: {e}")
                # Include segment with error note
                segments_with_text.append({
                    "idx": seg.get("idx", i),
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker_id": seg.get("speaker_id", 0),
                    "original_text": f"[Transcription error: {str(e)[:50]}]",
                    "translated_text": "",
                    "status": "error"
                })
        
        # Clear model from memory to free VRAM
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        logger.error(f"Failed to load Whisper model for direct inference: {e}")
        # Fallback: return empty transcriptions
        for i, seg in enumerate(segments):
            segments_with_text.append({
                "idx": seg.get("idx", i),
                "start": seg["start"],
                "end": seg["end"],
                "speaker_id": seg.get("speaker_id", 0),
                "original_text": "[Whisper initialization failed]",
                "translated_text": "",
                "status": "error"
            })
    
    return segments_with_text

# -------------------------------------------------------------------------
# PHASE 2: DUBBING (Translation + TTS)
# -------------------------------------------------------------------------

async def start_dubbing_task(task_id: str, user_config: Dict[str, Any], is_resume: bool = False, resume_from_idx: int = -1):
    """Phase 2: Translation and TTS generation using modular pipeline with translation broadcasting."""
    await log_ws(task_id, "Phase 2: Starting Translation & Dubbing", "step")
    
    # Get task info for language and TTS engine settings
    from core.database import db
    task = db.get_task(task_id)
    
    target_language = task.get('tgt_lang', 'en') if task else 'en'
    tts_engine = task.get('tts_engine', 'f5') if task else 'f5'
    
    # Get pre-transcribed segments if available
    transcribed_segments = task.get('transcribed_segments', []) if task else []
    
    # Log GPU info
    if torch.cuda.is_available():
        await log_ws(task_id, f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)", "info")
    else:
        await log_ws(task_id, "WARNING: Running on CPU, dubbing will be slow", "warning")
    
    # Update status to processing
    await update_status(task_id, {
        "status": "processing",
        "phase": "translating",
        "progress": 35,
        "message": "Starting translation pipeline..."
    })
    
    # Create progress callback that updates WebSocket AND broadcasts translations
    async def progress_callback(phase: str, percent: int, message: str):
        await log_ws(task_id, message, "info")
        await update_status(task_id, {
            "progress": percent,
            "message": message
        })
        await progress_ws(task_id, phase, percent, message)
    
    # Create translation update callback to show progress in UI
    async def translation_update_callback(segments_batch: list):
        """Called as translations complete to update the UI progressively."""
        await broadcast_translation_progress(task_id, segments_batch)
    
    try:
        # Run the modular Phase 2 pipeline with translation broadcasting
        # Note: We now pass pre_transcribed_segments to avoid re-transcription
        success = await run_phase2_translation(
            task_id=task_id,
            speaker_config=user_config,
            target_language=target_language,
            tts_engine=tts_engine,
            is_resume=is_resume,
            resume_from_idx=resume_from_idx,
            progress_callback=progress_callback,
            translation_update_callback=translation_update_callback,
            pre_transcribed_segments=transcribed_segments  # Pass pre-transcribed segments
        )
        
        if success:
            await log_ws(task_id, "Phase 2 complete! Starting final assembly...", "success")
            # Phase 3 will be triggered by task_manager
        else:
            await log_ws(task_id, "Phase 2 completed with errors", "warning")
            
    except Exception as e:
        traceback.print_exc()
        await log_ws(task_id, f"Dubbing Failed: {e}", "error")
        await update_status(task_id, {
            "status": "failed",
            "message": str(e),
            "error_message": str(e)
        })
        raise  # Re-raise for task_manager to handle


async def broadcast_translation_progress(task_id: str, segments_batch: list):
    """Broadcast translation progress to update UI with translated text."""
    if task_id not in active_connections:
        return
    
    try:
        dead_sockets = set()
        for ws in active_connections[task_id]:
            try:
                await ws.send_text(json.dumps({
                    "type": "translation_progress",
                    "data": {
                        "segments": segments_batch
                    }
                }))
            except Exception:
                dead_sockets.add(ws)
        
        # Clean up dead connections
        for ws in dead_sockets:
            active_connections[task_id].discard(ws)
            
    except Exception as e:
        logger.warning(f"Failed to broadcast translation progress: {e}")
