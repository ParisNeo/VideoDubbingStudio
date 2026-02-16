"""
Phase 3: Final Video Assembly

Complete pipeline: Build speech track → Mix with background → Merge with video

This module handles the final assembly of the dubbed video with proper
audio synchronization and optional background preservation.

Checkpoints:
- After speech track built
- After final audio mixed
"""

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


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class AudioSegment:
    """Represents an audio segment with timing."""
    start: float
    end: float
    audio_path: str
    speaker_id: int


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

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


def time_stretch(audio: np.ndarray, 
                 target_duration: float,
                 current_sample_rate: int = 24000) -> np.ndarray:
    """
    Time-stretch audio to match target duration.
    
    Uses librosa if available, otherwise returns original.
    """
    current_duration = len(audio) / current_sample_rate
    
    if abs(current_duration - target_duration) < 0.05:
        return audio  # Close enough
    
    try:
        import librosa
        rate = current_duration / target_duration
        return librosa.effects.time_stretch(audio, rate=rate)
    except Exception as e:
        logger.warning(f"Time-stretch failed: {e}")
        return audio


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
            
            # Resample to target rate
            if seg_sr != target_sample_rate:
                import librosa
                seg_audio = librosa.resample(seg_audio, orig_sr=seg_sr, 
                                              target_sr=target_sample_rate)
            
            # Time-stretch to match original duration
            target_duration = seg.end - seg.start
            seg_audio = time_stretch(seg_audio, target_duration, target_sample_rate)
            
            # Place in track
            start_sample = int(seg.start * target_sample_rate)
            end_sample = min(start_sample + len(seg_audio), len(track))
            actual_len = end_sample - start_sample
            
            # Apply crossfade and add to track
            seg_audio = apply_crossfade(seg_audio[:actual_len], 
                                        fade_samples=target_sample_rate // 100,  # 10ms
                                        sample_rate=target_sample_rate)
            
            track[start_sample:end_sample] += seg_audio
            
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
        import librosa
        speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
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
        import librosa
        bg = librosa.resample(bg, orig_sr=bg_sr, target_sr=sr)
    
    # Match lengths
    if len(bg) < len(speech):
        # Loop background
        repeats = int(np.ceil(len(speech) / len(bg)))
        bg = np.tile(bg, repeats)[:len(speech)]
    else:
        bg = bg[:len(speech)]
    
    # Apply ducking: background lowers when speech present
    speech_rms = np.sqrt(np.convolve(speech**2, np.ones(sr//10)/(sr//10), mode='same'))
    speech_present = speech_rms > 0.01
    
    # Ducking envelope
    from scipy.ndimage import gaussian_filter1d
    bg_envelope = np.where(speech_present, 0.3, 1.0)
    bg_envelope = gaussian_filter1d(bg_envelope, sigma=sr//50)
    
    # Apply gains
    speech_gain = 10 ** (speech_gain_db / 20)
    bg_gain = 10 ** (background_gain_db / 20)
    
    mixed = speech * speech_gain + bg * bg_gain * bg_envelope
    
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
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")
    
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


# =============================================================================
# MAIN PHASE 3 ENTRY POINT
# =============================================================================

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
    
    async def report(phase: str, percent: int, message: str):
        if progress_callback:
            await progress_callback(phase, percent, message)
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
        
        await report("loading", 82, f"Loaded {len(segments)} audio segments")
        
        # Setup output paths
        output_dir = Path("outputs") / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get master audio for Demucs
        master_audio = task.get('master_audio')
        
        # Step 1: Optional background separation
        background_path = None
        if use_demucs and master_audio:
            await report("separating", 85, "Separating background audio...")
            bg_dir = Path("temp_chunks") / task_id / "demucs"
            bg_dir.mkdir(parents=True, exist_ok=True)
            background_path = separate_background_demucs(master_audio, bg_dir)
        
        # Step 2: Build speech track
        await report("assembling", 88, "Building speech track...")
        
        speech_track_path = output_dir / "speech_track.wav"
        
        def speech_progress(current, total):
            pct = 88 + int((current / total) * 4)
            asyncio.create_task(report("assembling", pct, 
                f"Placed {current}/{total} segments"))
        
        build_speech_track(segments, speech_track_path, 
                          progress_callback=speech_progress)
        
        await report("assembling", 92, "Speech track built")
        
        # CHECKPOINT: Save after speech track
        db.update_task(
            task_id,
            phase="recomposing",
            status="processing",
            progress=92,
            message="Speech track built - mixing audio..."
        )
        
        # Step 3: Mix audio
        await report("mixing", 94, "Mixing speech with background...")
        
        final_audio_path = output_dir / "final_audio.wav"
        
        mix_audio_tracks(
            str(speech_track_path),
            background_path,
            final_audio_path,
            progress_callback=lambda msg: asyncio.create_task(report("mixing", 96, msg))
        )
        
        await report("mixing", 96, "Audio mixing complete")
        
        # CHECKPOINT: Save after audio mix
        db.update_task(
            task_id,
            progress=96,
            message="Audio mixed - rendering final video..."
        )
        
        # Step 4: Merge with video
        await report("rendering", 98, "Rendering final video...")
        
        final_video_path = output_dir / "dubbed_video.mp4"
        
        merge_with_video(
            original_video_path,
            str(final_audio_path),
            final_video_path,
            progress_callback=lambda msg: asyncio.create_task(report("rendering", 99, msg))
        )
        
        # Final checkpoint
        db.update_task(
            task_id,
            status="completed",
            phase="complete",
            progress=100,
            output_path=str(final_video_path),
            message="Dubbing complete!"
        )
        
        await report("complete", 100, "Video dubbing complete!")
        
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
