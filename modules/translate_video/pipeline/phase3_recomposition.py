"""
Phase 3: Video Recomposition

Final assembly of the dubbed video:
1. Collect all generated TTS audio segments
2. Mix with background audio (optional, using Demucs separation)
3. Assemble continuous audio track with proper timing
4. Merge with original video using FFmpeg

Design principles:
- Precise audio synchronization
- Background audio preservation (optional)
- Proper audio levels and crossfades
- Resume support for final assembly step
"""

import asyncio
import numpy as np
import soundfile as sf
import subprocess
import logging
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass

from core.resources import manager
from core.database import db
from modules.translate_video.state import broadcast_to_task

logger = logging.getLogger("phase3_recomposition")


@dataclass
class AudioSegment:
    """Represents an audio segment with timing information."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    audio_path: str
    speaker_id: int
    is_speech: bool = True


class Phase3Recomposer:
    """
    Final video recomposition with audio mixing and synchronization.
    """
    
    def __init__(
        self,
        task_id: str,
        original_video_path: str,
        use_demucs: bool = False,
        progress_callback: Optional[Callable[[str, int, str], None]] = None
    ):
        self.task_id = task_id
        self.original_video_path = Path(original_video_path)
        self.use_demucs = use_demucs and manager.get_demucs() is not None
        self.progress_callback = progress_callback
        
        # Load task info
        self.task = db.get_task(task_id)
        if not self.task:
            raise ValueError(f"Task {task_id} not found")
        
        # Get output paths
        self.output_dir = Path("outputs") / task_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_video_path = self.output_dir / "dubbed_video.mp4"
        self.final_audio_path = self.output_dir / "dubbed_audio.wav"
        
        # Audio parameters
        self.target_sample_rate = 48000  # Standard for video
        
        # Load segments
        self.segments = self._load_translation_segments()
        
        # Background audio (if separated)
        self.background_audio_path: Optional[str] = None
        
        logger.info(f"Phase3 initialized: {len(self.segments)} segments, "
                   f"demucs={self.use_demucs}")
    
    def _load_translation_segments(self) -> List[AudioSegment]:
        """Load translated segments from database."""
        db_segments = db.get_translation_segments(self.task_id)
        
        segments = []
        for seg in db_segments:
            segments.append(AudioSegment(
                start=seg.get('start', 0),
                end=seg.get('end', 0),
                audio_path=seg.get('audio_path', ''),
                speaker_id=seg.get('speaker_id', 0),
                is_speech=True
            ))
        
        # Sort by start time
        segments.sort(key=lambda x: x.start)
        return segments
    
    async def run(self) -> Optional[str]:
        """
        Execute the full recomposition pipeline.
        
        Returns:
            Path to final video file, or None if failed
        """
        try:
            # Step 1: Background separation (if enabled)
            if self.use_demucs:
                await self._report_progress("separating", 80, "Separating background audio with Demucs...")
                await self._separate_background()
            
            # Step 2: Build speech track
            await self._report_progress("assembling", 82, "Assembling speech audio track...")
            speech_track_path = await self._build_speech_track()
            
            # Step 3: Mix with background
            await self._report_progress("mixing", 88, "Mixing speech with background...")
            final_audio_path = await self._mix_audio(speech_track_path)
            
            # Step 4: Merge with video
            await self._report_progress("rendering", 94, "Rendering final video...")
            final_video_path = await self._merge_with_video(final_audio_path)
            
            # Step 5: Update task status
            db.update_task(
                self.task_id,
                status="completed",
                phase="complete",
                progress=100,
                output_path=str(final_video_path),
                message="Dubbing complete!"
            )
            
            await self._report_progress("complete", 100, "Video dubbing complete!")
            
            return str(final_video_path)
            
        except Exception as e:
            # FULL TRACEBACK LOGGING
            tb_str = traceback.format_exc()
            logger.exception(f"Phase 3 failed with full traceback:\n{tb_str}")
            
            db.update_task(
                self.task_id,
                status="failed",
                error_message=f"Final assembly failed: {str(e)}",
                error_traceback=tb_str  # Store full traceback
            )
            raise  # Re-raise with full context
    
    async def _separate_background(self):
        """Separate background audio using Demucs."""
        try:
            # Check if already separated
            vocals_path = Path("temp_chunks") / self.task_id / "vocals.wav"
            background_path = Path("temp_chunks") / self.task_id / "background.wav"
            
            if vocals_path.exists() and background_path.exists():
                logger.info("Using existing Demucs separation")
                self.background_audio_path = str(background_path)
                return
            
            # Get master audio
            master_audio = self.task.get('master_audio')
            if not master_audio or not Path(master_audio).exists():
                logger.warning("No master audio for Demucs separation, skipping")
                self.use_demucs = False
                return
            
            import torch
            import torchaudio
            from demucs.apply import apply_model
            from demucs.audio import convert_audio
            
            # Load model
            model = manager.get_demucs()
            device = next(model.parameters()).device
            
            # Load audio
            wav, sr = torchaudio.load(master_audio)
            
            # Convert to model's expected format
            wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
            wav = wav.to(device)
            
            # Apply model
            with torch.no_grad():
                sources = apply_model(model, wav[None], split=True, overlap=0.25)[0]
            
            # sources: [n_sources, n_channels, time]
            # Order: drums, bass, other, vocals
            
            # Separate dir
            sep_dir = Path("temp_chunks") / self.task_id / "demucs"
            sep_dir.mkdir(parents=True, exist_ok=True)
            
            # Save vocals (for reference, not used)
            vocals = sources[3].mean(dim=0).cpu().numpy()
            sf.write(vocals_path, vocals, model.samplerate)
            
            # Save background (drums + bass + other)
            background = (sources[0] + sources[1] + sources[2]).mean(dim=0).cpu().numpy()
            sf.write(background_path, background, model.samplerate)
            
            self.background_audio_path = str(background_path)
            
            logger.info(f"Demucs separation complete: {background_path}")
            
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.warning(f"Demucs separation failed with traceback:\n{tb_str}, continuing without background")
            self.use_demucs = False
    
    async def _build_speech_track(self) -> str:
        """
        Build continuous speech track from all segments.
        
        Returns:
            Path to speech track WAV file
        """
        # Get video duration for track length
        video_duration = await self._get_video_duration()
        
        # Calculate total samples at target rate
        total_samples = int(video_duration * self.target_sample_rate)
        
        # Initialize silent track
        track = np.zeros(total_samples, dtype=np.float32)
        
        # Place each segment
        for seg in self.segments:
            if not seg.audio_path or not Path(seg.audio_path).exists():
                logger.warning(f"Missing audio for segment at {seg.start}s, skipping")
                continue
            
            try:
                # Load segment audio
                seg_audio, seg_sr = sf.read(seg.audio_path)
                
                # Ensure mono
                if len(seg_audio.shape) > 1:
                    seg_audio = seg_audio.mean(axis=1)
                
                # Resample to target rate if needed
                if seg_sr != self.target_sample_rate:
                    import librosa
                    seg_audio = librosa.resample(seg_audio, orig_sr=seg_sr, target_sr=self.target_sample_rate)
                
                # Calculate position in track
                start_sample = int(seg.start * self.target_sample_rate)
                
                # Adjust segment duration to match original timing
                target_duration = seg.end - seg.start
                current_duration = len(seg_audio) / self.target_sample_rate
                
                if abs(current_duration - target_duration) > 0.1:
                    # Time-stretch to match
                    try:
                        import librosa
                        rate = current_duration / target_duration
                        seg_audio = librosa.effects.time_stretch(seg_audio, rate=rate)
                        logger.debug(f"Time-stretched segment: {current_duration:.2f}s -> {target_duration:.2f}s")
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.warning(f"Time-stretch failed with traceback:\n{tb_str}")
                
                # Ensure we don't overflow
                end_sample = min(start_sample + len(seg_audio), len(track))
                actual_len = end_sample - start_sample
                
                # Apply crossfade
                seg_audio = self._apply_crossfade(seg_audio[:actual_len])
                
                # Add to track
                track[start_sample:end_sample] += seg_audio[:actual_len]
                
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Failed to place segment at {seg.start}s with traceback:\n{tb_str}")
        
        # Normalize track
        peak = np.max(np.abs(track))
        if peak > 0.99:
            track = track / peak * 0.99
        
        # Save speech track
        speech_path = self.output_dir / "speech_track.wav"
        sf.write(speech_path, track, self.target_sample_rate)
        
        return str(speech_path)
    
    def _apply_crossfade(self, audio: np.ndarray, fade_samples: int = 480) -> np.ndarray:
        """
        Apply crossfade to beginning and end of audio segment.
        
        Args:
            audio: Audio array
            fade_samples: Number of samples for fade (10ms at 48kHz)
        
        Returns:
            Audio with crossfade applied
        """
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
    
    async def _mix_audio(self, speech_track_path: str) -> str:
        """
        Mix speech track with background audio.
        
        Args:
            speech_track_path: Path to speech track WAV
        
        Returns:
            Path to final mixed audio
        """
        # Load speech track
        speech, sr = sf.read(speech_track_path)
        
        # Ensure correct sample rate
        if sr != self.target_sample_rate:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate
        
        # If no background, just return speech
        if not self.background_audio_path or not Path(self.background_audio_path).exists():
            logger.info("No background audio, using speech only")
            sf.write(self.final_audio_path, speech, sr)
            return str(self.final_audio_path)
        
        # Load and prepare background
        bg, bg_sr = sf.read(self.background_audio_path)
        
        # Resample background if needed
        if bg_sr != self.target_sample_rate:
            import librosa
            bg = librosa.resample(bg, orig_sr=bg_sr, target_sr=self.target_sample_rate)
        
        # Match lengths
        if len(bg) < len(speech):
            # Loop background to match length
            repeats = int(np.ceil(len(speech) / len(bg)))
            bg = np.tile(bg, repeats)[:len(speech)]
        else:
            bg = bg[:len(speech)]
        
        # Apply ducking: lower background when speech is present
        speech_rms = np.sqrt(np.convolve(speech**2, np.ones(sr//10)/(sr//10), mode='same'))
        speech_present = speech_rms > 0.01
        
        # Ducking envelope: 0.3 (quiet) when speech present, 1.0 when silent
        ducking = np.where(speech_present, 0.3, 1.0)
        
        # Smooth the ducking transitions
        from scipy.ndimage import gaussian_filter1d
        ducking = gaussian_filter1d(ducking, sigma=sr//50)  # 20ms smoothing
        
        # Apply ducking
        bg_ducked = bg * ducking
        
        # Mix with speech
        mixed = speech * 0.95 + bg_ducked * 0.8
        
        # Final normalization
        peak = np.max(np.abs(mixed))
        if peak > 0.99:
            mixed = mixed / peak * 0.99
        
        # Save
        sf.write(self.final_audio_path, mixed, sr)
        
        logger.info(f"Mixed audio saved: {self.final_audio_path}")
        
        return str(self.final_audio_path)
    
    async def _merge_with_video(self, audio_path: str) -> str:
        """
        Merge final audio with original video using FFmpeg.
        
        Args:
            audio_path: Path to final mixed audio
        
        Returns:
            Path to final video file
        """
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.original_video_path),  # Video input
            "-i", audio_path,                      # Audio input
            "-c:v", "copy",                        # Copy video stream (no re-encode)
            "-c:a", "aac",                         # AAC audio codec
            "-b:a", "192k",                        # Audio bitrate
            "-ar", "48000",                        # Audio sample rate
            "-map", "0:v:0",                       # Take video from first input
            "-map", "1:a:0",                       # Take audio from second input
            "-shortest",                           # Match shortest duration
            "-movflags", "+faststart",             # Web optimization
            str(self.final_video_path)
        ]
        
        # Run FFmpeg
        returncode, stdout, stderr = await self._run_ffmpeg(cmd)
        
        if returncode != 0:
            stderr_text = stderr.decode('utf-8', errors='replace') if stderr else "Unknown error"
            error_msg = f"FFmpeg merge failed: {stderr_text}"
            logger.error(f"FFmpeg failed with return code {returncode}: {stderr_text}")
            raise RuntimeError(error_msg)
        
        # Verify output exists
        if not self.final_video_path.exists():
            raise RuntimeError("FFmpeg completed but output file not found")
        
        logger.info(f"Final video saved: {self.final_video_path}")
        
        return str(self.final_video_path)
    
    async def _get_video_duration(self) -> float:
        """Get video duration in seconds using FFprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.original_video_path)
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            duration = float(stdout.decode().strip())
            return duration
            
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.warning(f"Failed to get video duration with traceback:\n{tb_str}, using fallback")
            # Fallback: estimate from segments
            if self.segments:
                return max(s.end for s in self.segments) + 1.0
            return 60.0  # Default 1 minute
    
    async def _run_ffmpeg(self, cmd: List[str]) -> Tuple[int, bytes, bytes]:
        """Run FFmpeg command asynchronously."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        return proc.returncode, stdout, stderr
    
    async def _report_progress(self, phase: str, percent: int, message: str):
        """Report progress via callback and broadcast."""
        if self.progress_callback:
            try:
                await self.progress_callback(phase, percent, message)
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.warning(f"Progress callback failed with traceback:\n{tb_str}")
        
        # Update database
        db.update_task(
            self.task_id,
            progress=percent,
            message=message
        )
        
        # Broadcast via WebSocket
        try:
            await broadcast_to_task(self.task_id, {
                'type': 'progress',
                'data': {
                    'phase': phase,
                    'percent': percent,
                    'message': message
                }
            })
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.warning(f"WebSocket broadcast failed with traceback:\n{tb_str}")


async def run_phase3_recomposition(
    task_id: str,
    original_video_path: str,
    use_demucs: bool = False,
    progress_callback: Optional[Callable[[str, int, str], None]] = None
) -> Optional[str]:
    """
    Convenience function to run Phase 3 recomposition.
    
    Args:
        task_id: The task ID
        original_video_path: Path to original video file
        use_demucs: Whether to use Demucs for background separation
        progress_callback: Optional callback for progress updates
    
    Returns:
        Path to final video file, or None if failed
    """
    recomposer = Phase3Recomposer(
        task_id=task_id,
        original_video_path=original_video_path,
        use_demucs=use_demucs,
        progress_callback=progress_callback
    )
    
    return await recomposer.run()
