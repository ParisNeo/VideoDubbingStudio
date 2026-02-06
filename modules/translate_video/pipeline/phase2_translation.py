"""
Phase 2: Translation Pipeline

This module handles the complete translation workflow:
1. Transcribe each speech segment using Whisper (with configurable source language) - SKIPPED if pre-transcribed
2. Translate text using Lollms (when source != target)
3. Generate TTS audio using F5-TTS or FishSpeech with voice cloning (configurable engine)

Design principles:
- Chunked processing to stay within 8GB VRAM
- Progress reporting via callbacks
- State persistence after each chunk for resumability
- Speaker-aware processing (use correct voice sample per speaker)
- Configurable source language, target language, and TTS engine
- OPTIMIZED: Translate BEFORE loading TTS model to avoid VRAM pressure during translation
"""

import asyncio
import numpy as np
import soundfile as sf
import torch
import logging
import gc
import os
import io
import base64
import tempfile
import requests
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from core.resources import manager
from core.database import db
from modules.translate_video.state import broadcast_to_task

logger = logging.getLogger("phase2_translation")


@dataclass
class TranslationSegment:
    """Represents a single segment's translation state."""
    idx: int
    start: float  # original timestamp in seconds
    end: float
    speaker_id: int
    speaker_name: str = ""  # User-defined name from validation
    original_text: str = ""
    translated_text: str = ""
    audio_path: Optional[str] = None  # Path to generated TTS audio
    status: str = "pending"  # pending, transcribing, translating, synthesizing, completed, failed
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class Phase2TranslationPipeline:
    """
    GPU-efficient translation pipeline with chunked processing.
    
    Processes segments in small batches to maintain 8GB VRAM budget
    while providing real-time progress updates.
    
    OPTIMIZED MODEL LOADING ORDER:
    1. Load Whisper only if needed (not pre-transcribed)
    2. Unload Whisper, load Lollms for translation
    3. Unload Lollms, load TTS model ONLY after all translation is done
    4. Synthesize speech
    """
    
    def __init__(
        self,
        task_id: str,
        speaker_config: Dict[str, Any],
        target_language: str = "en",
        source_language: str = "auto",  # NEW: source language for transcription
        tts_engine: str = "f5",  # NEW: TTS engine selection
        chunk_size: int = 3,  # segments per batch (conservative for 8GB)
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        translation_update_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,  # NEW: progressive translation updates
        pre_transcribed_segments: Optional[List[Dict[str, Any]]] = None  # NEW: pre-transcribed segments from Phase 1
    ):
        self.task_id = task_id
        self.speaker_config = speaker_config  # {speaker_id: {name, action, sample_path}}
        self.target_language = target_language
        self.source_language = source_language  # NEW
        self.tts_engine = tts_engine  # NEW
        self.chunk_size = chunk_size
        self.progress_callback = progress_callback
        self.translation_update_callback = translation_update_callback  # NEW
        self.pre_transcribed_segments = pre_transcribed_segments  # NEW
        
        # Load segments from database
        self.segments = self._load_segments()
        
        # Create translation segment objects
        self.translation_segments: List[TranslationSegment] = []
        
        # Track overall progress
        self.total_segments = len(self.segments)
        self.completed_segments = 0
        
        # Models (lazy loaded - order matters for VRAM!)
        self._whisper = None
        self._lollms = None
        self._f5_model = None
        self._f5_vocoder = None
        
        # Flags for model loading state
        self._translation_models_loaded = False
        self._tts_models_loaded = False
        
        logger.info(f"Phase2 initialized: {self.total_segments} segments, "
                   f"src_lang={source_language}, tgt_lang={target_language}, "
                   f"engine={tts_engine}, "
                   f"pre_transcribed={pre_transcribed_segments is not None and len(pre_transcribed_segments) > 0}")
    
    def _load_segments(self) -> List[Dict[str, Any]]:
        """Load segments from database."""
        task = db.get_task(self.task_id)
        if not task:
            raise ValueError(f"Task {self.task_id} not found")
        
        segments = task.get('segments', [])
        if not segments:
            raise ValueError("No segments found - cannot proceed with translation")
        
        # Get master audio path for extracting segment audio
        self.master_audio_path = task.get('master_audio')
        if not self.master_audio_path or not Path(self.master_audio_path).exists():
            raise ValueError("Master audio not found")
        
        return segments
    
    async def run(self) -> bool:
        """
        Execute the full translation pipeline with OPTIMIZED model loading order.
        
        NEW ORDER:
        1. Transcribe (if not pre-transcribed) → unload Whisper
        2. Translate all segments → unload Lollms
        3. Load TTS model → synthesize all
        
        Returns True if successful, False if any segments failed.
        """
        try:
            # Initialize translation segment objects
            self._initialize_translation_segments()
            
            # Calculate which segments to process (for resume support)
            start_idx = self._find_resume_point()
            if start_idx > 0:
                logger.info(f"Resuming from segment {start_idx}")
                await self._report_progress("resuming", start_idx, "Resuming translation...")
            
            # =================================================================
            # PHASE A: TRANSCRIPTION (only for non-pre-transcribed segments)
            # =================================================================
            transcription_needed = [
                ts for ts in self.translation_segments 
                if ts.status in ('pending', 'transcribing') and not ts.original_text
            ]
            
            if transcription_needed:
                await self._report_progress("transcribing", 35, f"Transcribing {len(transcription_needed)} segments...")
                await self._transcribe_all(transcription_needed)
                # Unload Whisper immediately to free VRAM
                self._unload_whisper()
            else:
                logger.info("Skipping transcription - all segments have pre-transcribed or existing text")
                # Mark any pending segments with text as ready for translation
                for ts in self.translation_segments:
                    if ts.status in ('pending', 'transcribing') and ts.original_text:
                        ts.status = 'translating'
            
            # =================================================================
            # PHASE B: TRANSLATION (load Lollms, translate all, unload Lollms)
            # =================================================================
            translation_needed = [
                ts for ts in self.translation_segments
                if ts.status == 'translating' or (ts.original_text and not ts.translated_text)
            ]
            
            if translation_needed:
                await self._report_progress("translating", 40, f"Translating {len(translation_needed)} segments...")
                # Load Lollms for translation
                await self._load_translation_models()
                await self._translate_all(translation_needed)
                # Unload Lollms to free VRAM for TTS
                self._unload_translation_models()
                
                # Broadcast translation progress
                if self.translation_update_callback:
                    segments_data = [
                        {
                            "idx": ts.idx,
                            "segment_idx": ts.idx,
                            "start": ts.start,
                            "end": ts.end,
                            "speaker_id": ts.speaker_id,
                            "original_text": ts.original_text,
                            "translated_text": ts.translated_text if ts.translated_text else "",
                            "status": "translated" if ts.translated_text else ts.status
                        }
                        for ts in self.translation_segments
                    ]
                    try:
                        await self.translation_update_callback(segments_data)
                    except Exception as e:
                        logger.warning(f"Translation update callback failed: {e}")
            else:
                logger.info("Skipping translation - no segments need translation")
            
            # =================================================================
            # PHASE C: TTS SYNTHESIS (load TTS model, synthesize all)
            # =================================================================
            synthesis_needed = [
                ts for ts in self.translation_segments
                if ts.status in ('synthesizing', 'translated') or 
                   (ts.translated_text and not ts.audio_path)
            ]
            
            if synthesis_needed:
                await self._report_progress("synthesizing", 60, f"Synthesizing {len(synthesis_needed)} segments with {self.tts_engine}...")
                # Load TTS model ONLY now, after all translation is done
                await self._load_tts_models()
                await self._synthesize_all(synthesis_needed)
            else:
                logger.info("Skipping synthesis - no segments need audio generation")
            
            # =================================================================
            # FINAL SUMMARY
            # =================================================================
            success_count = sum(1 for s in self.translation_segments if s.status == "completed")
            logger.info(f"Phase 2 complete: {success_count}/{self.total_segments} segments successful")
            
            # Update final task status
            if success_count == self.total_segments:
                db.update_task(
                    self.task_id,
                    phase="recomposing",
                    status="queued",
                    progress=80,
                    message="Translation complete, starting final assembly..."
                )
                return True
            else:
                db.update_task(
                    self.task_id,
                    status="failed",
                    error_message=f"Only {success_count}/{self.total_segments} segments translated successfully"
                )
                return False
                
        except Exception as e:
            logger.exception("Phase 2 failed")
            db.update_task(
                self.task_id,
                status="failed",
                error_message=f"Translation pipeline failed: {str(e)}"
            )
            return False
    
    def _initialize_translation_segments(self):
        """Create TranslationSegment objects from diarization results."""
        # Log pre-transcribed segments info for debugging
        if self.pre_transcribed_segments:
            logger.info(f"Received {len(self.pre_transcribed_segments)} pre-transcribed segments")
            for ps in self.pre_transcribed_segments[:3]:  # Log first 3 for debugging
                idx = ps.get('idx') if 'idx' in ps else ps.get('segment_idx', 'unknown')
                text_preview = (ps.get('original_text', '')[:30] + '...') if ps.get('original_text') else '(empty)'
                logger.info(f"  Pre-transcribed segment {idx}: {text_preview}")
        
        for seg_data in self.segments:
            speaker_id = seg_data.get('speaker_id', 0)
            speaker_info = self.speaker_config.get(str(speaker_id), {})
            
            ts = TranslationSegment(
                idx=seg_data.get('idx', 0),
                start=seg_data.get('start', 0),
                end=seg_data.get('end', 0),
                speaker_id=speaker_id,
                speaker_name=speaker_info.get('name', f"Speaker {speaker_id + 1}")
            )
            
            # Check if already processed (from previous run or pre-transcribed)
            self._check_existing_translation(ts)
            
            self.translation_segments.append(ts)
        
        # Sort by index to ensure correct order
        self.translation_segments.sort(key=lambda x: x.idx)
        
        # Log final status
        pending = sum(1 for ts in self.translation_segments if ts.status == 'pending')
        transcribing = sum(1 for ts in self.translation_segments if ts.status == 'transcribing')
        translating = sum(1 for ts in self.translation_segments if ts.status == 'translating')
        completed = sum(1 for ts in self.translation_segments if ts.status == 'completed')
        logger.info(f"Segment status after init: pending={pending}, transcribing={transcribing}, translating={translating}, completed={completed}")
    
    def _check_existing_translation(self, ts: TranslationSegment):
        """Check database and pre-transcribed data for existing translation of this segment."""
        # First check database for resumed tasks
        existing = db.get_translation_segments(self.task_id)
        for ex in existing:
            if ex.get('segment_idx') == ts.idx:
                ts.original_text = ex.get('original_text', '')
                ts.translated_text = ex.get('translated_text', '')
                ts.audio_path = ex.get('audio_path')
                # Restore timing info if available
                if 'start_time' in ex:
                    ts.start = ex.get('start_time', ts.start)
                if 'end_time' in ex:
                    ts.end = ex.get('end_time', ts.end)
                if 'speaker_id' in ex:
                    ts.speaker_id = ex.get('speaker_id', ts.speaker_id)
                if ts.audio_path and Path(ts.audio_path).exists():
                    ts.status = 'completed'
                    logger.info(f"Segment {ts.idx}: found completed in database")
                else:
                    # Has translation but no audio - needs synthesis
                    if ts.translated_text:
                        ts.status = 'synthesizing'
                        logger.info(f"Segment {ts.idx}: found translated in database, needs synthesis")
                    else:
                        ts.status = 'translating' if ts.original_text else 'pending'
                        logger.info(f"Segment {ts.idx}: found in database, status={ts.status}")
                return  # Found in DB, done
        
        # If no existing translation in DB, check pre-transcribed segments from Phase 1
        if self.pre_transcribed_segments:
            for pre_seg in self.pre_transcribed_segments:
                # Handle both 'idx' (from Phase 1) and 'segment_idx' (from DB) keys
                pre_idx = pre_seg.get('idx') if 'idx' in pre_seg else pre_seg.get('segment_idx')
                if pre_idx == ts.idx or pre_idx == str(ts.idx):
                    pre_text = pre_seg.get('original_text', '')
                    if pre_text:  # Only use if there's actual text
                        ts.original_text = pre_text
                        ts.status = 'translating'  # Skip transcription, go straight to translation
                        logger.info(f"Using pre-transcribed text for segment {ts.idx}: {ts.original_text[:50]}...")
                        return
        
        # If we get here, no existing data found - will need full pipeline
        logger.debug(f"Segment {ts.idx}: no existing data, will use full pipeline")
    
    def _find_resume_point(self) -> int:
        """Find first segment that needs processing."""
        for i, ts in enumerate(self.translation_segments):
            if ts.status != 'completed':
                return i
        return len(self.translation_segments)  # All done
    
    # =================================================================
    # MODEL LOADING (OPTIMIZED ORDER)
    # =================================================================
    
    async def _load_translation_models(self):
        """Load Lollms client for translation. Called only when translation is needed."""
        if self._translation_models_loaded:
            return
        
        logger.info("Loading translation model (Lollms)...")
        self._lollms = manager.get_lollms_client()
        self._translation_models_loaded = True
        logger.info("Translation model loaded")
    
    def _unload_translation_models(self):
        """Unload Lollms to free VRAM for TTS."""
        if not self._translation_models_loaded:
            return
        
        logger.info("Unloading translation model to free VRAM...")
        self._lollms = None
        self._translation_models_loaded = False
        manager.clear_cache(keep=['speaker_encoder'])  # Keep speaker encoder if needed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Translation model unloaded, VRAM freed")
    
    async def _load_tts_models(self):
        """Load TTS model ONLY after all translation is done. This is the VRAM-heavy model."""
        if self._tts_models_loaded:
            return
        
        if self.tts_engine == 'f5':
            logger.info("Loading TTS model (F5-TTS) - this is VRAM-intensive...")
            self._f5_model, self._f5_vocoder = manager.get_f5_tts()
            self._tts_models_loaded = True
            logger.info("TTS model loaded")
        else:
            # FishSpeech is API-based, no local model to load
            self._tts_models_loaded = True
            logger.info("Using FishSpeech API (no local model to load)")
    
    def _unload_tts_models(self):
        """Unload TTS models."""
        if not self._tts_models_loaded:
            return
        
        if self.tts_engine == 'f5':
            logger.info("Unloading TTS model...")
            self._f5_model = None
            self._f5_vocoder = None
        
        self._tts_models_loaded = False
        manager.clear_cache(keep=['speaker_encoder'])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TTS model unloaded")
    
    # =================================================================
    # PHASE A: TRANSCRIPTION
    # =================================================================
    
    async def _transcribe_all(self, segments: List[TranslationSegment]):
        """Transcribe all segments that need it."""
        if not segments:
            return
        
        if self._whisper is None:
            self._whisper = manager.get_whisper()
        
        total = len(segments)
        for i, ts in enumerate(segments):
            try:
                ts.status = 'transcribing'
                await self._log(f"Transcribing segment {ts.idx} (source lang: {self.source_language})...")
                
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
                
                await self._log(f"Segment {ts.idx}: '{ts.original_text[:50]}...'")
                
                # Progress update
                progress = 35 + int((i + 1) / total * 5)  # 35-40% range
                await self._report_progress("transcribing", progress, f"Transcribed {i+1}/{total} segments")
                
            except Exception as e:
                logger.error(f"Transcription failed for segment {ts.idx}: {e}")
                ts.status = 'failed'
                ts.error = f"Transcription: {str(e)}"
                # Continue with other segments, don't fail entire pipeline
    
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
    
    def _unload_whisper(self):
        """Unload Whisper to free VRAM for translation."""
        if self._whisper is None:
            return
        
        logger.info("Unloading Whisper to free VRAM...")
        self._whisper = None
        manager.clear_cache(keep=['speaker_encoder'])  # Keep speaker encoder if needed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper unloaded")
    
    # =================================================================
    # PHASE B: TRANSLATION
    # =================================================================
    
    async def _translate_all(self, segments: List[TranslationSegment]):
        """Translate all segments that need it."""
        if not segments:
            return
        
        if not self._lollms:
            logger.warning("Lollms not available, copying original text as translation")
            for ts in segments:
                if ts.status != 'failed':
                    ts.translated_text = ts.original_text
                    ts.status = 'synthesizing'
            return
        
        total = len(segments)
        for i, ts in enumerate(segments):
            if ts.status == 'failed':
                continue
            
            try:
                # Check if speaker action is 'remove'
                speaker_info = self.speaker_config.get(str(ts.speaker_id), {})
                if speaker_info.get('action') == 'remove':
                    ts.translated_text = ""  # Will result in silence
                    ts.status = 'synthesizing'
                    await self._log(f"Segment {ts.idx}: speaker set to remove, skipping translation")
                    continue
                
                # Skip translation if source and target are the same
                if self.source_language == self.target_language:
                    ts.translated_text = ts.original_text
                    ts.status = 'synthesizing'
                    await self._log(f"Segment {ts.idx}: source=target ({self.target_language}), skipping translation")
                    continue
                
                # Build translation prompt
                prompt = self._build_translation_prompt(ts.original_text)
                
                # Generate with timeout
                translated = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._lollms.generate_text,
                        prompt,
                        temperature=0.3
                    ),
                    timeout=30.0
                )
                
                ts.translated_text = translated.strip()
                ts.status = 'synthesizing'
                
                await self._log(f"Segment {ts.idx} translated: '{ts.translated_text[:50]}...'")
                
                # Progress update
                progress = 40 + int((i + 1) / total * 20)  # 40-60% range
                await self._report_progress("translating", progress, f"Translated {i+1}/{total} segments")
                
            except Exception as e:
                logger.error(f"Translation failed for segment {ts.idx}: {e}")
                # Fallback: use original text
                ts.translated_text = ts.original_text
                ts.status = 'synthesizing'
                await self._log(f"Segment {ts.idx}: translation failed, using original")
    
    def _build_translation_prompt(self, text: str) -> str:
        """Build translation prompt for Lollms."""
        return f"""Translate the following text to {self.target_language}. 
Preserve the meaning, tone, and style. Only output the translation, no explanations.

Text: {text}

Translation:"""
    
    # =================================================================
    # PHASE C: TTS SYNTHESIS
    # =================================================================
    
    async def _synthesize_all(self, segments: List[TranslationSegment]):
        """Generate TTS audio for all segments."""
        if not segments:
            return
        
        total = len(segments)
        for i, ts in enumerate(segments):
            if ts.status == 'failed':
                continue
            
            try:
                # Check if speaker action is 'remove'
                speaker_info = self.speaker_config.get(str(ts.speaker_id), {})
                if speaker_info.get('action') == 'remove':
                    # Generate silence for removed speakers
                    await self._generate_silence(ts)
                    continue
                
                # Get reference audio for this speaker
                sample_path = speaker_info.get('sample_path')
                if not sample_path:
                    raise ValueError(f"No voice sample for speaker {ts.speaker_id}")
                
                # Resolve sample path
                if sample_path.startswith('/'):
                    # URL path, convert to file path
                    sample_path = str(Path("temp_chunks") / Path(sample_path).name)
                
                if not Path(sample_path).exists():
                    # Try alternative locations
                    alt_path = Path("temp_chunks") / self.task_id / "speaker_samples" / f"speaker_{ts.speaker_id}_sample.wav"
                    if alt_path.exists():
                        sample_path = str(alt_path)
                    else:
                        raise ValueError(f"Voice sample not found: {sample_path}")
                
                await self._log(f"Synthesizing segment {ts.idx} with {self.tts_engine} engine, voice of {ts.speaker_name}...")
                
                # Generate TTS based on configured engine
                if self.tts_engine == 'fishspeech':
                    audio = await self._generate_tts_fishspeech(ts.translated_text, sample_path)
                else:  # default to f5
                    audio = await self._generate_tts_f5(ts.translated_text, sample_path)
                
                # Save to disk
                output_path = self._save_segment_audio(ts, audio)
                ts.audio_path = output_path
                ts.status = 'completed'
                
                # Save to database with timing info
                db.save_translation_segment(
                    self.task_id,
                    ts.idx,
                    ts.original_text,
                    ts.translated_text,
                    output_path,
                    'completed',
                    start_time=ts.start,
                    end_time=ts.end,
                    speaker_id=ts.speaker_id
                )
                
                await self._log(f"Segment {ts.idx} synthesized successfully with {self.tts_engine}")
                
                # Progress update
                progress = 60 + int((i + 1) / total * 20)  # 60-80% range
                await self._report_progress("synthesizing", progress, f"Synthesized {i+1}/{total} segments")
                
            except Exception as e:
                logger.error(f"TTS failed for segment {ts.idx} with {self.tts_engine}: {e}")
                ts.status = 'failed'
                ts.error = f"TTS ({self.tts_engine}): {str(e)}"
                # Don't fail entire pipeline, continue with other segments
    
    async def _generate_tts_f5(self, text: str, ref_path: str) -> np.ndarray:
        """Generate TTS using F5-TTS."""
        import librosa
        
        # Load reference audio
        ref_audio, sr = sf.read(ref_path)
        if sr != 24000:
            ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=24000)
        
        # Ensure mono
        if len(ref_audio.shape) > 1:
            ref_audio = ref_audio.mean(axis=1)
        
        # Limit reference length
        ref_audio = ref_audio[: 10 * 24000]  # Max 10 seconds
        
        # Write temp file for F5
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, ref_audio, 24000)
            ref_tmp = f.name
        
        try:
            # Import here to handle missing F5 gracefully
            from f5_tts.infer.utils_infer import infer_process
            
            # Run inference
            audio, sr_out, _ = infer_process(
                ref_audio=ref_tmp,
                ref_text="",  # F5 doesn't need reference text
                gen_text=text,
                model_obj=self._f5_model,
                vocoder=self._f5_vocoder,
                mel_spec_type="vocos",
                device=manager.device,
            )
            
            # Convert to numpy
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            audio = np.asarray(audio).flatten()
            
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            
            return audio
            
        finally:
            try:
                import os
                os.remove(ref_tmp)
            except:
                pass
    
    async def _generate_tts_fishspeech(self, text: str, ref_path: str) -> np.ndarray:
        """Generate TTS using FishSpeech API."""
        import librosa
        
        # Get FishSpeech URL from environment or resource manager
        fish_speech_url = os.getenv("FISH_SPEECH_API_URL", "http://127.0.0.1:8080/v1/tts")
        
        # Load reference audio
        ref_audio, sr = sf.read(ref_path)
        
        # Get reference text via STT if not cached
        ref_text = self._get_cached_ref_text(ref_path)
        
        # Encode reference audio as base64
        bio = io.BytesIO()
        sf.write(bio, ref_audio.astype(np.float32), sr, format='WAV')
        audio_b64 = base64.b64encode(bio.getvalue()).decode()
        
        # Call FishSpeech API
        payload = {
            "text": text,
            "reference_audio": audio_b64,
            "reference_text": ref_text,
        }
        
        try:
            response = requests.post(
                fish_speech_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            # Decode response
            content_type = response.headers.get('content-type', '').lower()
            if 'application/json' in content_type:
                result = response.json()
                audio_b64 = result.get('audio') or result.get('wav') or result.get('data') or result.get('audio_base64')
                if not audio_b64:
                    raise RuntimeError(f"FishSpeech returned JSON without audio field: keys={list(result.keys())}")
                audio_bytes = base64.b64decode(audio_b64)
            else:
                audio_bytes = response.content
            
            # Load audio from bytes
            bio = io.BytesIO(audio_bytes)
            audio, out_sr = sf.read(bio)
            
            # Resample to 24kHz if needed (standardize with F5 output)
            if out_sr != 24000:
                audio = librosa.resample(audio, orig_sr=out_sr, target_sr=24000)
            
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            
            return audio
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to FishSpeech API at {fish_speech_url}. Is the server running?")
        except requests.exceptions.Timeout:
            raise RuntimeError("FishSpeech API request timed out after 120 seconds")
    
    async def _generate_silence(self, ts: TranslationSegment):
        """Generate silent audio for removed speakers."""
        # Create appropriate duration of silence
        duration = ts.end - ts.start
        # F5-TTS and FishSpeech both output 24kHz
        sr = 24000
        samples = int(duration * sr)
        silence = np.zeros(samples, dtype=np.float32)
        
        output_path = self._save_segment_audio(ts, silence)
        ts.audio_path = output_path
        ts.status = 'completed'
        
        db.save_translation_segment(
            self.task_id,
            ts.idx,
            "",  # No original text
            "",  # No translated text
            output_path,
            'completed',
            start_time=ts.start,
            end_time=ts.end,
            speaker_id=ts.speaker_id
        )
        
        await self._log(f"Segment {ts.idx}: generated silence for removed speaker")
    
    def _get_cached_ref_text(self, ref_path: str) -> str:
        """Get or compute reference text for voice cloning."""
        cache_path = Path(ref_path).with_suffix('.txt')
        if cache_path.exists():
            return cache_path.read_text()
        
        # Compute with STT
        stt = manager.get_whisper()
        result = stt(ref_path)
        text = result.get('text', '')
        
        # Cache
        cache_path.write_text(text)
        return text
    
    def _save_segment_audio(self, ts: TranslationSegment, audio: np.ndarray) -> str:
        """Save synthesized audio to disk."""
        output_dir = Path("outputs") / self.task_id / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use 24kHz for consistent output
        sr = 24000
        
        # Time-stretch if needed to match original duration
        target_duration = ts.end - ts.start
        current_duration = len(audio) / sr
        duration_diff = abs(current_duration - target_duration)
        
        if duration_diff > 0.5:  # More than 0.5s difference
            try:
                import librosa
                rate = current_duration / target_duration
                audio = librosa.effects.time_stretch(audio, rate=rate)
                logger.info(f"Time-stretched segment {ts.idx}: {current_duration:.2f}s -> {target_duration:.2f}s")
            except Exception as e:
                logger.warning(f"Time-stretch failed for segment {ts.idx}: {e}")
        
        # Save
        path = output_dir / f"seg_{ts.idx:04d}_spk{ts.speaker_id}.wav"
        sf.write(path, audio, sr)
        
        return str(path)
    
    def _save_checkpoint(self):
        """Save progress to database for resume support."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'completed_segments': sum(1 for s in self.translation_segments if s.status == 'completed'),
            'total_segments': len(self.translation_segments),
            'segments': [ts.to_dict() for ts in self.translation_segments]
        }
        
        # FIX: Unpack the dictionary with ** instead of passing as positional argument
        db.update_task(self.task_id, checkpoint_data=checkpoint)
        
        # Also broadcast progress via WebSocket
        progress = int((checkpoint['completed_segments'] / checkpoint['total_segments']) * 100)
        # Map to overall progress (35-80%)
        overall_progress = 35 + int(progress * 0.45)
        
        # Don't block on broadcast
        asyncio.create_task(self._broadcast_progress(overall_progress))
    
    async def _broadcast_progress(self, progress: int):
        """Broadcast progress to connected WebSocket clients."""
        try:
            await broadcast_to_task(self.task_id, {
                'type': 'progress',
                'data': {
                    'phase': 'translating',
                    'percent': progress,
                    'message': f'Translation {progress}% complete'
                }
            })
        except Exception as e:
            logger.warning(f"Failed to broadcast progress: {e}")
    
    async def _report_progress(self, phase: str, percent: int, message: str):
        """Report progress via callback and log."""
        if self.progress_callback:
            try:
                await self.progress_callback(phase, percent, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Also update database
        db.update_task(
            self.task_id,
            progress=percent,
            message=message
        )
        
        await self._log(message)
    
    async def _log(self, message: str):
        """Log message to database and WebSocket."""
        logger.info(message)
        
        # Add to task logs
        try:
            from modules.translate_video.project_manager import append_log
            append_log(self.task_id, message, "info")
        except Exception:
            pass
        
        # Broadcast via WebSocket
        try:
            await broadcast_to_task(self.task_id, {
                'type': 'log',
                'data': {'message': message, 'style': 'info'}
            })
        except Exception:
            pass


async def run_phase2_translation(
    task_id: str,
    speaker_config: Dict[str, Any],
    target_language: str = "en",
    source_language: str = "auto",  # NEW parameter
    tts_engine: str = "f5",  # NEW parameter
    is_resume: bool = False,
    resume_from_idx: int = -1,
    progress_callback: Optional[Callable[[str, int, str], None]] = None,
    translation_update_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,  # NEW: progressive translation updates
    pre_transcribed_segments: Optional[List[Dict[str, Any]]] = None  # NEW: pre-transcribed segments from Phase 1 to avoid re-transcription
) -> bool:
    """
    Convenience function to run Phase 2 translation.
    
    Args:
        task_id: The task ID
        speaker_config: Speaker configuration from validation
        target_language: Target language code
        source_language: Source language code ('auto' for auto-detect)
        tts_engine: TTS engine to use ('f5' or 'fishspeech')
        is_resume: Whether this is a resume operation
        resume_from_idx: Segment index to resume from (if known)
        progress_callback: Optional callback for progress updates
        translation_update_callback: Optional callback for progressive translation updates to UI
        pre_transcribed_segments: Optional pre-transcribed segments from Phase 1 to avoid re-transcription
        
    Returns:
        True if successful, False otherwise
    """
    # Load task to get stored settings if not explicitly provided
    task = db.get_task(task_id)
    if task:
        # Use stored settings as defaults, allow overrides
        if target_language == "en" and task.get('tgt_lang'):
            target_language = task.get('tgt_lang')
        if source_language == "auto" and task.get('src_lang'):
            source_language = task.get('src_lang')
        if tts_engine == "f5" and task.get('tts_engine'):
            tts_engine = task.get('tts_engine')
        
        # CRITICAL: Also try to get pre_transcribed_segments from task if not provided
        if pre_transcribed_segments is None and task.get('transcribed_segments'):
            pre_transcribed_segments = task.get('transcribed_segments')
            logger.info(f"Loaded {len(pre_transcribed_segments)} pre-transcribed segments from task")
    
    # Ensure target_language is never None - default to 'en' if still None
    if target_language is None:
        logger.warning("target_language was None, defaulting to 'en'")
        target_language = "en"
    
    # Ensure pre_transcribed_segments is at least an empty list
    if pre_transcribed_segments is None:
        pre_transcribed_segments = []
    
    pipeline = Phase2TranslationPipeline(
        task_id=task_id,
        speaker_config=speaker_config,
        target_language=target_language,
        source_language=source_language,  # Pass to pipeline
        tts_engine=tts_engine,  # Pass to pipeline
        progress_callback=progress_callback,
        translation_update_callback=translation_update_callback,  # NEW: pass to pipeline
        pre_transcribed_segments=pre_transcribed_segments  # Pass pre-transcribed segments
    )
    
    # If resuming from specific index, we could modify the pipeline state here
    # The pipeline automatically detects resume point from database
    
    return await pipeline.run()
