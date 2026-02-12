"""
Phase 2, Subphase 1: Transcription

Handles speech-to-text conversion using Whisper for segments that don't have
pre-transcribed text. This subphase can be skipped entirely if all segments
were pre-transcribed in Phase 1.

Design principles:
- Load Whisper on-demand, unload immediately after to free VRAM
- Process segments in batches to show progress
- Graceful handling of transcription failures (continue with other segments)
"""

import asyncio
import numpy as np
import soundfile as sf
import torch
import logging
import traceback
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from core.resources import manager
from .phase2_models import TranslationSegment

logger = logging.getLogger("phase2_subphase1_transcription")


class TranscriptionSubphase:
    """
    Handles STT for segments needing transcription.
    VRAM-efficient: loads Whisper, transcribes, unloads immediately.
    """
    
    def __init__(
        self,
        task_id: str,
        master_audio_path: str,
        source_language: str = "auto",
        progress_callback: Optional[Callable[[str, int, str], None]] = None
    ):
        self.task_id = task_id
        self.master_audio_path = master_audio_path
        self.source_language = source_language
        self.progress_callback = progress_callback
        
        self._whisper = None
        self._loaded = False
        
        logger.info(f"TranscriptionSubphase initialized: src_lang={source_language}")
    
    async def run(self, segments: List[TranslationSegment]) -> List[TranslationSegment]:
        """
        Transcribe all segments that need it.
        
        Args:
            segments: List of TranslationSegment objects needing transcription
        
        Returns:
            Updated segments with original_text populated
        """
        if not segments:
            logger.info("No segments need transcription")
            return segments
        
        await self._report_progress("transcribing", 35, 
            f"Loading Whisper for {len(segments)} segments...")
        
        try:
            # Load Whisper model
            await self._load_whisper()
            
            total = len(segments)
            for i, ts in enumerate(segments):
                try:
                    ts.status = 'transcribing'
                    await self._log(f"Transcribing segment {ts.idx} "
                                   f"(source lang: {self.source_language})...")
                    
                    # Extract audio segment from master file
                    audio = self._extract_segment_audio(ts)
                    
                    # Build generate_kwargs based on source language setting
                    generate_kwargs = {}
                    if self.source_language and self.source_language != 'auto':
                        generate_kwargs["language"] = self.source_language
                    
                    # Run Whisper - CRITICAL FIX: Pass dict even if empty
                    result = self._whisper(
                        {"array": audio, "sampling_rate": 16000},
                        return_timestamps=False,
                        generate_kwargs=generate_kwargs if generate_kwargs else None
                    )
                    
                    ts.original_text = result.get('text', '').strip()
                    ts.status = 'translating'  # Ready for next step
                    
                    # CONSOLE OUTPUT: Show original transcript
                    print(f"\n{'='*60}")
                    print(f"[SEGMENT {ts.idx}] ORIGINAL TRANSCRIPT ({self.source_language}):")
                    print(f"  Time: {ts.start:.2f}s - {ts.end:.2f}s | "
                          f"Speaker: {ts.speaker_name}")
                    print(f"  Text: \"{ts.original_text}\"")
                    print(f"{'='*60}\n")
                    
                    await self._log(f"Segment {ts.idx}: "
                                   f"'{ts.original_text[:50]}...'")
                    
                    # Progress update
                    progress = 35 + int((i + 1) / total * 5)  # 35-40% range
                    await self._report_progress("transcribing", progress, 
                        f"Transcribed {i+1}/{total} segments")
                    
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"Transcription failed for segment {ts.idx} "
                                f"with traceback:\n{tb_str}")
                    ts.status = 'failed'
                    ts.error = f"Transcription: {str(e)}\nTraceback: {tb_str[:500]}"
                    # Continue with other segments, don't fail entire pipeline
            
            return segments
            
        finally:
            # ALWAYS unload Whisper to free VRAM for next subphase
            self._unload_whisper()
    
    async def _load_whisper(self):
        """Load Whisper model."""
        if self._loaded:
            return
        
        logger.info("Loading Whisper model for transcription...")
        self._whisper = manager.get_whisper()
        self._loaded = True
        logger.info("Whisper loaded")
    
    def _unload_whisper(self):
        """Unload Whisper to free VRAM."""
        if not self._loaded:
            return
        
        logger.info("Unloading Whisper to free VRAM...")
        self._whisper = None
        self._loaded = False
        manager.clear_cache(keep=['speaker_encoder'])  # Keep speaker encoder
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper unloaded")
    
    def _extract_segment_audio(self, ts: TranslationSegment) -> np.ndarray:
        """Extract audio for a specific time segment from master file."""
        # Load the relevant portion of the master audio
        with sf.SoundFile(self.master_audio_path) as f:
            sr = f.samplerate
            start_sample = int(ts.start * sr)
            end_sample = int(ts.end * sr)
            duration_samples = end_sample - start_sample
            
            f.seek(start_sample)
            audio = f.read(duration_samples)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        return audio
    
    async def _report_progress(self, phase: str, percent: int, message: str):
        """Report progress via callback."""
        if self.progress_callback:
            try:
                await self.progress_callback(phase, percent, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def _log(self, message: str):
        """Log message."""
        logger.info(message)


async def run_transcription_subphase(
    task_id: str,
    master_audio_path: str,
    segments: List[TranslationSegment],
    source_language: str = "auto",
    progress_callback: Optional[Callable[[str, int, str], None]] = None
) -> List[TranslationSegment]:
    """
    Convenience function to run transcription subphase.
    
    Args:
        task_id: The task ID
        master_audio_path: Path to extracted master audio
        segments: Segments needing transcription
        source_language: Source language code
        progress_callback: Optional callback for progress updates
    
    Returns:
        Updated segments with transcription results
    """
    subphase = TranscriptionSubphase(
        task_id=task_id,
        master_audio_path=master_audio_path,
        source_language=source_language,
        progress_callback=progress_callback
    )
    
    return await subphase.run(segments)
