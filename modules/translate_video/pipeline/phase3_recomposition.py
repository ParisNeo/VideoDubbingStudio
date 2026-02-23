# CRITICAL: SVML Workaround - MUST be first, before ANY numpy/scipy imports
# This prevents LLVM errors with Intel MKL/SVML on Windows
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"  # Disable AVX/AVX2/SVML
os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX2,AVX"  # Disable NumPy AVX

# Also set thread limits for any already-loaded OpenMP
try:
    import ctypes
    ctypes.CDLL(None).omp_set_num_threads(1)
except:
    pass

# Now safe to import numpy/scipy
import asyncio
import numpy as np
import soundfile as sf
import subprocess
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass

from core.database import db

logger = logging.getLogger("phase3_recomposition")


# -------------------------------------------------------------------------------------------------------------------------------
# DATA MODELS
# -------------------------------------------------------------------------------------------------------------------------------

@dataclass
class AudioSegment:
    """Represents an audio segment with timing."""
    start: float
    end: float
    audio_path: str
    speaker_id: int


# -------------------------------------------------------------------------------------------------------------------------------
# AUDIO PROCESSING
# -------------------------------------------------------------------------------------------------------------------------------

def apply_crossfade(audio: np.ndarray, 
                    fade_samples: int = 480,
                    sample_rate: int = 48000) -> np.ndarray:
    """Apply crossfade to beginning and end of audio."""
    if len(audio) < fade_samples * 2:
        return audio
    
    result = audio.copy()
    
    # Fade in
    fade_in = np.linspace(0, 1, fade_samples)
    result[:fade_samples] *= fade_in
    
    # Fade out
    fade_out = np.linspace(1, 0, fade_samples)
    result[-fade_samples:] *= fade_out
    
    return result


def build_speech_track(
    segments: List[AudioSegment],
    output_path: Path,
    target_sample_rate: int = 48000,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Build continuous speech track from all segments.
    
    Returns: path to speech track file
    """
    if not segments:
        # Create silent track (need duration from somewhere)
        sf.write(output_path, np.zeros(target_sample_rate), target_sample_rate)
        return str(output_path)
    
    # Find max end time
    max_end = max(s.end for s in segments)
    total_samples = int(max_end * target_sample_rate)
    
    # Initialize silent track
    track = np.zeros(total_samples, dtype=np.float32)
    
    for i, seg in enumerate(segments):
        try:
            # Load segment audio
            seg_audio, seg_sr = sf.read(seg.audio_path)
            
            # Ensure mono
            if len(seg_audio.shape) > 1:
                seg_audio = seg_audio.mean(axis=1)
            
            # Resample to target rate using simple method (avoid librosa/SVML)
            if seg_sr != target_sample_rate:
                # Simple resampling using numpy
                resample_ratio = target_sample_rate / seg_sr
                new_length = int(len(seg_audio) * resample_ratio)
                if new_length > 0:
                    indices = np.linspace(0, len(seg_audio) - 1, new_length)
                    indices_floor = np.floor(indices).astype(np.int64)
                    indices_ceil = np.minimum(indices_floor + 1, len(seg_audio) - 1)
                    fractions = indices - indices_floor
                    seg_audio = seg_audio[indices_floor] * (1 - fractions) + seg_audio[indices_ceil] * fractions
                    seg_audio = seg_audio.astype(np.float32)
            
            # Place in track without pitch-shifting stretch
            start_sample = int(seg.start * target_sample_rate)
            
            # Ensure track is long enough to hold this audio
            track_end_needed = start_sample + len(seg_audio)
            if track_end_needed > len(track):
                track = np.pad(track, (0, track_end_needed - len(track)))
            
            # Apply crossfade and add to track
            seg_audio = apply_crossfade(seg_audio, 
                                        fade_samples=target_sample_rate // 100,  # 10ms
                                        sample_rate=target_sample_rate)
            
            track[start_sample:track_end_needed] += seg_audio
            
            if progress_callback:
                progress_callback(i + 1, len(segments))
                
        except Exception as e:
            logger.error(f"Failed to place segment at {seg.start}s: {e}")
    
    # Normalize
    peak = np.max(np.abs(track))
    if peak > 0.99:
        track = track / peak * 0.99
    
    # Save
    sf.write(output_path, track, target_sample_rate)
    
    return str(output_path)


def separate_background_demucs(
    audio_path: str,
    output_dir: Path,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Optional[str]:
    """
    Separate background audio using Demucs.
    
    Returns: path to background audio, or None if failed
    """
    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        
        # Load model
        from demucs.pretrained import get_model
        model = get_model("htdemucs")
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to model format
        wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
        
        if torch.cuda.is_available():
            wav = wav.cuda()
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(model, wav[None], split=True, overlap=0.25)[0]
        
        # sources: [drums, bass, other, vocals]
        # Background = drums + bass + other
        background = sources[0] + sources[1] + sources[2]
        background = background.mean(dim=0).cpu().numpy()
        
        # Save
        bg_path = output_dir / "background.wav"
        sf.write(bg_path, background, model.samplerate)
        
        if progress_callback:
            progress_callback("Background separation complete")
        
        return str(bg_path)
        
    except Exception as e:
        logger.error(f"Demucs separation failed: {e}")
        return None


def mix_audio_tracks(
    speech_path: str,
    background_path: Optional[str],
    output_path: Path,
    target_duration: Optional[float] = None,
    speech_gain_db: float = -6,
    background_gain_db: float = -20,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Mix speech with background audio.
    
    Returns: path to mixed audio
    """
    # Load speech
    speech, sr = sf.read(speech_path)
    
    # Ensure target sample rate
    target_sr = 48000
    if sr != target_sr:
        # Simple resampling
        resample_ratio = target_sr / sr
        new_length = int(len(speech) * resample_ratio)
        if new_length > 0:
            indices = np.linspace(0, len(speech) - 1, new_length)
            indices_floor = np.floor(indices).astype(np.int64)
            indices_ceil = np.minimum(indices_floor + 1, len(speech) - 1)
            fractions = indices - indices_floor
            speech = speech[indices_floor] * (1 - fractions) + speech[indices_ceil] * fractions
            speech = speech.astype(np.float32)
        sr = target_sr
    
    # If no background, just apply speech gain
    if not background_path or not Path(background_path).exists():
        speech_gain = 10 ** (speech_gain_db / 20)
        speech = speech * speech_gain
        
        # Normalize
        peak = np.max(np.abs(speech))
        if peak > 0.99:
            speech = speech / peak * 0.99
        
        sf.write(output_path, speech, sr)
        return str(output_path)
    
    # Load background
    bg, bg_sr = sf.read(background_path)
    
    # Resample background
    if bg_sr != sr:
        resample_ratio = sr / bg_sr
        new_length = int(len(bg) * resample_ratio)
        if new_length > 0:
            indices = np.linspace(0, len(bg) - 1, new_length)
            indices_floor = np.floor(indices).astype(np.int64)
            indices_ceil = np.minimum(indices_floor + 1, len(bg) - 1)
            fractions = indices - indices_floor
            bg = bg[indices_floor] * (1 - fractions) + bg[indices_ceil] * fractions
            bg = bg.astype(np.float32)
    
    # Match lengths
    if len(bg) < len(speech):
        # Loop background
        repeats = int(np.ceil(len(speech) / len(bg)))
        bg = np.tile(bg, repeats)[:len(speech)]
    else:
        bg = bg[:len(speech)]
    
    # Simple ducking: background lowers when speech present
    # Use simple moving average instead of convolution to avoid SciPy
    window_size = sr // 10  # 100ms window
    speech_power = np.zeros(len(speech))
    for i in range(len(speech)):
        start = max(0, i - window_size // 2)
        end = min(len(speech), i + window_size // 2)
        speech_power[i] = np.mean(speech[start:end] ** 2)
    
    speech_present = speech_power > 0.0001  # Threshold
    
    # Simple smoothing with box filter
    ducking = np.ones(len(speech))
    for i in range(len(speech)):
        start = max(0, i - window_size // 5)  # 20ms smoothing
        end = min(len(speech), i + window_size // 5)
        ducking[i] = 0.3 if np.any(speech_present[start:end]) else 1.0
    
    # Apply gains
    speech_gain = 10 ** (speech_gain_db / 20)
    bg_gain = 10 ** (background_gain_db / 20)
    
    mixed = speech * speech_gain + bg * bg_gain * ducking
    
    # Final normalize
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    
    # Save
    sf.write(output_path, mixed, sr)
    
    if progress_callback:
        progress_callback("Audio mixing complete")
    
    return str(output_path)


def merge_with_video(
    video_path: str,
    audio_path: str,
    output_path: Path,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Merge final audio with video using FFmpeg.
    
    Returns: path to final video
    
    Raises:
        RuntimeError: If merge fails
    """
    # First attempt: Use aac with strict flag (for older FFmpeg versions)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "-2",  # Enable experimental AAC encoder
        "-b:a", "192k",
        "-ar", "48000",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # If first attempt fails, try with libvo_aacenc (non-experimental alternative)
    if result.returncode != 0:
        error_msg = result.stderr
        
        # Check if it's the AAC experimental encoder error
        if "experimental" in error_msg.lower() or "libvo_aacenc" in error_msg.lower():
            logger.warning("AAC encoder failed, trying libvo_aacenc fallback...")
            
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "libvo_aacenc",  # Non-experimental AAC encoder
                "-b:a", "192k",
                "-ar", "48000",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-movflags", "+faststart",
                str(output_path)
            ]
            
            result_fallback = subprocess.run(cmd_fallback, capture_output=True, text=True)
            
            if result_fallback.returncode == 0:
                if progress_callback:
                    progress_callback("Video merge complete (using libvo_aacenc)")
                return str(output_path)
            
            # If fallback also fails, raise with both error messages
            raise RuntimeError(
                f"FFmpeg merge failed with both AAC encoders.\n"
                f"Original (aac -strict -2): {error_msg}\n"
                f"Fallback (libvo_aacenc): {result_fallback.stderr}"
            )
        
        # Not an AAC error, raise original error
        raise RuntimeError(f"FFmpeg merge failed: {error_msg}")
    
    if progress_callback:
        progress_callback("Video merge complete")
    
    return str(output_path)


def get_video_duration(video_path: str) -> float:
    """Get video duration using FFprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")
    
    return float(result.stdout.strip())


# -------------------------------------------------------------------------------------------------------------------------------
# MAIN PHASE 3 ENTRY POINT
# -------------------------------------------------------------------------------------------------------------------------------

async def run_phase3(
    task_id: str,
    original_video_path: str,
    use_demucs: bool = False,
    progress_callback: Optional[Callable[[str, int, str], None]] = None
) -> str:
    """
    Run complete Phase 3 pipeline.
    
    Args:
        task_id: The task ID
        original_video_path: Path to original video
        use_demucs: Whether to use Demucs for background separation
        progress_callback: Called with (phase, percent, message)
    
    Returns:
        Path to final video file
    
    Raises:
        RuntimeError: If assembly fails
    """
    
    import functools
    loop = asyncio.get_running_loop()
    
    def threadsafe_report(phase: str, percent: int, message: str):
        if progress_callback:
            asyncio.run_coroutine_threadsafe(progress_callback(phase, percent, message), loop)
        logger.info(f"[Phase 3] {percent}%: {message}")
    
    try:
        # Load task data
        task = db.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Load segments with audio paths
        seg_data = task.get('translation_segments', [])
        if not seg_data:
            raise ValueError("No translation segments found - run Phase 2 first")
        
        # Get all completed segments (including silent ones)
        segments = [
            AudioSegment(
                start=s['start'],
                end=s['end'],
                audio_path=s['audio_path'],
                speaker_id=s['speaker_id']
            )
            for s in seg_data
            if s.get('audio_path') and s.get('status') == 'completed'
        ]
        
        # Check if we have any segments at all
        if not segments:
            # Check if all segments failed vs. no segments were created
            failed_count = sum(1 for s in seg_data if s.get('status') == 'failed')
            if failed_count > 0:
                raise ValueError(f"All {failed_count} audio segments failed to synthesize")
            else:
                raise ValueError("No audio segments were created - Phase 2 may not have run")
        
        # Warn if all segments are silent (TTS failures)
        segments_with_errors = [s for s in seg_data if s.get('status') == 'completed' and s.get('error')]
        if len(segments_with_errors) == len(seg_data):
            logger.warning("All segments are silent (TTS failed for all). Video will have no dubbed audio.")
        
        threadsafe_report("loading", 82, f"Loaded {len(segments)} audio segments")
        
        # Setup output paths
        output_dir = Path("outputs") / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get master audio for Demucs
        master_audio = task.get('master_audio')
        
        # Step 1: Optional background separation
        background_path = None
        if use_demucs and master_audio:
            threadsafe_report("separating", 85, "Separating background audio...")
            bg_dir = Path("temp_chunks") / task_id / "demucs"
            bg_dir.mkdir(parents=True, exist_ok=True)
            background_path = await loop.run_in_executor(
                None, functools.partial(separate_background_demucs, master_audio, bg_dir)
            )
        
        # Step 2: Build speech track
        threadsafe_report("assembling", 88, "Building speech track...")
        
        speech_track_path = output_dir / "speech_track.wav"
        
        def speech_progress(current, total):
            pct = 88 + int((current / total) * 4)
            threadsafe_report("assembling", pct, f"Placed {current}/{total} segments")
        
        await loop.run_in_executor(
            None, functools.partial(build_speech_track, segments, speech_track_path, target_sample_rate=48000, progress_callback=speech_progress)
        )
        
        threadsafe_report("assembling", 92, "Speech track built")
        
        # CHECKPOINT: Save after speech track
        db.update_task(
            task_id,
            phase="recomposing",
            status="processing",
            progress=92,
            message="Speech track built - mixing audio..."
        )
        
        # Step 3: Mix audio
        threadsafe_report("mixing", 94, "Mixing speech with background...")
        
        final_audio_path = output_dir / "final_audio.wav"
        
        await loop.run_in_executor(
            None, functools.partial(
                mix_audio_tracks,
                str(speech_track_path),
                background_path,
                final_audio_path,
                target_duration=None,
                speech_gain_db=-6,
                background_gain_db=-20,
                progress_callback=lambda msg: threadsafe_report("mixing", 96, msg)
            )
        )
        
        threadsafe_report("mixing", 96, "Audio mixing complete")
        
        # CHECKPOINT: Save after audio mix
        db.update_task(
            task_id,
            progress=96,
            message="Audio mixed - rendering final video..."
        )
        
        # Step 4: Merge with video
        threadsafe_report("rendering", 98, "Rendering final video...")
        
        final_video_path = output_dir / "dubbed_video.mp4"
        
        await loop.run_in_executor(
            None, functools.partial(
                merge_with_video,
                original_video_path,
                str(final_audio_path),
                final_video_path,
                progress_callback=lambda msg: threadsafe_report("rendering", 99, msg)
            )
        )
        
        # Standardization: Ensure the path is relative to the project root for the static mount
        standard_path = f"outputs/{task_id}/dubbed_video.mp4"

        # Final checkpoint
        db.update_task(
            task_id,
            status="completed",
            phase="complete",
            progress=100,
            output_path=standard_path,
            message="Dubbing complete!"
        )
        
        threadsafe_report("complete", 100, "Video dubbing complete!")
        
        return str(final_video_path)
        
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Phase 3 failed:\n{tb_str}")
        
        db.update_task(
            task_id,
            status="failed",
            phase="recomposing",
            error_message=f"Phase 3 failed: {str(e)}",
            error_traceback=tb_str
        )
        
        raise RuntimeError(f"Phase 3 assembly failed: {str(e)}") from e


# Export for workflow tasks
__all__ = [
    'run_phase3',
    'AudioSegment',
    'build_speech_track',
    'mix_audio_tracks',
    'merge_with_video',
    'separate_background_demucs'
]

