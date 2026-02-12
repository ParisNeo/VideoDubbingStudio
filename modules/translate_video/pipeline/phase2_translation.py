"""
Phase 2: Translation Pipeline (Refactored)

This module is now a thin orchestrator that calls the three subphases:
1. Transcription (phase2_subphase1_transcription.py) - Whisper STT
2. Translation (phase2_subphase2_translation.py) - Lollms text translation  
3. TTS Synthesis (phase2_subphase3_tts.py) - Voice cloning

Each subphase is self-contained with its own model loading/unloading
to maintain 8GB VRAM budget throughout the pipeline.

Design principles:
- Sequential subphase execution with VRAM cleanup between each
- Progress reporting via callbacks
- State persistence after each chunk for resumability
- Progressive translation updates to UI
"""

import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional, Callable

from core.database import db
from modules.translate_video.state import broadcast_to_task

# Import subphases
from .phase2_models import TranslationSegment
from .phase2_subphase1_transcription import run_transcription_subphase
from .phase2_subphase2_translation import run_translation_subphase
from .phase2_subphase3_tts import run_tts_subphase

logger = logging.getLogger("phase2_translation")


class Phase2TranslationPipeline:
    """
    Orchestrates the three subphases of Phase 2.
    
    VRAM-efficient execution order:
    1. Transcription subphase (load Whisper, transcribe, unload)
    2. Translation subphase (load Lollms, translate all, unload)
    3. TTS subphase (load TTS model, synthesize all, unload)
    """
    
    def __init__(
        self,
        task_id: str,
        speaker_config: Dict[str, Any],
        target_language: str = "en",
        source_language: str = "auto",
        tts_engine: str = "f5",
        chunk_size: int = 3,  # Kept for API compatibility, not used in subphases
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        translation_update_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        pre_transcribed_segments: Optional[List[Dict[str, Any]]] = None
    ):
        self.task_id = task_id
        self.speaker_config = speaker_config
        self.target_language = target_language
        self.source_language = source_language
        self.tts_engine = tts_engine
        self.progress_callback = progress_callback
        self.translation_update_callback = translation_update_callback
        self.pre_transcribed_segments = pre_transcribed_segments or []
        
        # Load segments from database
        self.segments = self._load_segments()
        self.translation_segments: List[TranslationSegment] = []
        
        logger.info(f"Phase2TranslationPipeline initialized: "
                   f"{len(self.segments)} segments, "
                   f"src_lang={source_language}, tgt_lang={target_language}, "
                   f"engine={tts_engine}, "
                   f"pre_transcribed={len(self.pre_transcribed_segments)}")
    
    def _load_segments(self) -> List[Dict[str, Any]]:
        """Load segments from database."""
        task = db.get_task(self.task_id)
        if not task:
            raise ValueError(f"Task {self.task_id} not found")
        
        segments = task.get('segments', [])
        if not segments:
            raise ValueError("No segments found - cannot proceed with translation")
        
        # CRITICAL FIX: Always use database values as the source of truth
        # The database is the persistent store, so we must read from it
        task_tgt = task.get('tgt_lang') or task.get('target_language')
        task_src = task.get('src_lang')
        task_engine = task.get('tts_engine')
        
        logger.info(f"Task database values: tgt_lang={task_tgt}, src_lang={task_src}, tts_engine={task_engine}")
        
        # Only override if we have valid database values
        # This ensures we never lose the language settings between phases
        if task_tgt and str(task_tgt).strip() and str(task_tgt).strip().lower() != 'none':
            if self.target_language != task_tgt:
                logger.info(f"Updating target_language from database: {self.target_language} -> {task_tgt}")
                self.target_language = task_tgt
        
        if task_src and str(task_src).strip() and str(task_src).strip().lower() != 'none':
            if self.source_language != task_src:
                logger.info(f"Updating source_language from database: {self.source_language} -> {task_src}")
                self.source_language = task_src
        
        if task_engine and str(task_engine).strip() and str(task_engine).strip().lower() != 'none':
            if self.tts_engine != task_engine:
                logger.info(f"Updating tts_engine from database: {self.tts_engine} -> {task_engine}")
                self.tts_engine = task_engine
        
        # Get master audio path
        self.master_audio_path = task.get('master_audio')
        if not self.master_audio_path or not Path(self.master_audio_path).exists():
            raise ValueError("Master audio not found")
        
        return segments
    
    async def run(self) -> bool:
        """
        Execute all three subphases in sequence.
        
        Returns:
            True if successful, False if any critical failure
        """
        try:
            # Initialize segment objects
            self._initialize_translation_segments()
            
            # Calculate resume point
            start_idx = self._find_resume_point()
            if start_idx > 0:
                logger.info(f"Resuming from segment {start_idx}")
                await self._report_progress("resuming", start_idx, "Resuming translation...")
            
            # =================================================================
            # SUBPHASE 1: TRANSCRIPTION (if needed)
            # =================================================================
            transcription_needed = [
                ts for ts in self.translation_segments
                if ts.status in ('pending', 'transcribing') and not ts.original_text
            ]
            
            if transcription_needed:
                logger.info(f"Subphase 1: Transcribing {len(transcription_needed)} segments")
                await run_transcription_subphase(
                    task_id=self.task_id,
                    master_audio_path=self.master_audio_path,
                    segments=transcription_needed,
                    source_language=self.source_language,
                    progress_callback=self.progress_callback
                )
            else:
                logger.info("Subphase 1: Skipping transcription (all pre-transcribed)")
                # Mark any pending segments with text as ready for translation
                for ts in self.translation_segments:
                    if ts.status in ('pending', 'transcribing') and ts.original_text:
                        ts.status = 'translating'
            
            # =================================================================
            # SUBPHASE 2: TRANSLATION
            # =================================================================
            translation_needed = [
                ts for ts in self.translation_segments
                if ts.status == 'translating' or (ts.original_text and not ts.translated_text)
            ]
            
            if translation_needed:
                logger.info(f"Subphase 2: Translating {len(translation_needed)} segments to {self.target_language}")
                await run_translation_subphase(
                    task_id=self.task_id,
                    segments=translation_needed,
                    target_language=self.target_language,
                    source_language=self.source_language,
                    speaker_config=self.speaker_config,
                    progress_callback=self.progress_callback,
                    translation_update_callback=self.translation_update_callback
                )
            else:
                logger.info("Subphase 2: No segments need translation")
            
            # =================================================================
            # SUBPHASE 3: TTS SYNTHESIS
            # =================================================================
            synthesis_needed = [
                ts for ts in self.translation_segments
                if ts.status in ('synthesizing', 'translated') or 
                   (ts.translated_text and not ts.audio_path)
            ]
            
            if synthesis_needed:
                logger.info(f"Subphase 3: Synthesizing {len(synthesis_needed)} segments with {self.tts_engine}")
                await run_tts_subphase(
                    task_id=self.task_id,
                    segments=synthesis_needed,
                    tts_engine=self.tts_engine,
                    speaker_config=self.speaker_config,
                    progress_callback=self.progress_callback
                )
            else:
                logger.info("Subphase 3: No segments need synthesis")
            
            # =================================================================
            # FINAL SUMMARY
            # =================================================================
            success_count = sum(1 for s in self.translation_segments 
                              if s.status == "completed")
            logger.info(f"Phase 2 complete: {success_count}/{len(self.translation_segments)} "
                       f"segments successful")
            
            # Update final task status
            if success_count == len(self.translation_segments):
                db.update_task(
                    self.task_id,
                    phase="recomposing",
                    status="queued",
                    progress=80,
                    message="Translation complete, starting final assembly..."
                )
                return True
            else:
                failed_segments = [s for s in self.translation_segments 
                                  if s.status == "failed"]
                error_summary = "\n".join([
                    f"Segment {s.idx}: {s.error}" 
                    for s in failed_segments[:5]
                ])
                full_error = f"Only {success_count}/{len(self.translation_segments)} " \
                            f"segments translated successfully.\n\n" \
                            f"Failed segments:\n{error_summary}"
                
                db.update_task(
                    self.task_id,
                    status="failed",
                    error_message=full_error
                )
                return False
                
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.exception(f"Phase 2 failed:\n{tb_str}")
            
            db.update_task(
                self.task_id,
                status="failed",
                error_message=f"Translation pipeline failed: {str(e)}",
                error_traceback=tb_str
            )
            return False
    
    def _initialize_translation_segments(self):
        """Create TranslationSegment objects from diarization results."""
        # Log pre-transcribed segments info
        if self.pre_transcribed_segments:
            logger.info(f"Received {len(self.pre_transcribed_segments)} "
                       f"pre-transcribed segments")
        
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
            
            # Check for existing progress (from resume)
            self._check_existing_translation(ts)
            
            self.translation_segments.append(ts)
        
        # Sort by index
        self.translation_segments.sort(key=lambda x: x.idx)
        
        # Log status
        pending = sum(1 for ts in self.translation_segments if ts.status == 'pending')
        logger.info(f"Segment status after init: {pending} pending of "
                   f"{len(self.translation_segments)} total")
    
    def _check_existing_translation(self, ts: TranslationSegment):
        """Check database and pre-transcribed data for existing translation."""
        # First check database for resumed tasks
        existing = db.get_translation_segments(self.task_id)
        for ex in existing:
            if ex.get('segment_idx') == ts.idx:
                ts.original_text = ex.get('original_text', '')
                ts.translated_text = ex.get('translated_text', '')
                ts.audio_path = ex.get('audio_path')
                if 'start_time' in ex:
                    ts.start = ex.get('start_time', ts.start)
                if 'end_time' in ex:
                    ts.end = ex.get('end_time', ts.end)
                if 'speaker_id' in ex:
                    ts.speaker_id = ex.get('speaker_id', ts.speaker_id)
                if ts.audio_path and Path(ts.audio_path).exists():
                    ts.status = 'completed'
                    logger.info(f"Segment {ts.idx}: found completed in database")
                elif ts.translated_text:
                    ts.status = 'synthesizing'
                    logger.info(f"Segment {ts.idx}: translated, needs synthesis")
                else:
                    ts.status = 'translating' if ts.original_text else 'pending'
                    logger.info(f"Segment {ts.idx}: found in database, "
                               f"status={ts.status}")
                return
        
        # Check pre-transcribed segments from Phase 1
        if self.pre_transcribed_segments:
            for pre_seg in self.pre_transcribed_segments:
                pre_idx = pre_seg.get('idx') if 'idx' in pre_seg else \
                         pre_seg.get('segment_idx')
                if pre_idx == ts.idx or pre_idx == str(ts.idx):
                    pre_text = pre_seg.get('original_text', '')
                    if pre_text:
                        ts.original_text = pre_text
                        ts.status = 'translating'  # Skip transcription
                        logger.info(f"Using pre-transcribed text for segment "
                                   f"{ts.idx}: {ts.original_text[:50]}...")
                        return
        
        logger.debug(f"Segment {ts.idx}: no existing data, full pipeline")
    
    def _find_resume_point(self) -> int:
        """Find first segment that needs processing."""
        for i, ts in enumerate(self.translation_segments):
            if ts.status != 'completed':
                return i
        return len(self.translation_segments)
    
    async def _report_progress(self, phase: str, percent: int, message: str):
        """Report progress via callback."""
        if self.progress_callback:
            try:
                await self.progress_callback(phase, percent, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Also update database
        db.update_task(self.task_id, progress=percent, message=message)
        
        # Broadcast via WebSocket
        try:
            await broadcast_to_task(self.task_id, {
                'type': 'progress',
                'data': {'phase': phase, 'percent': percent, 'message': message}
            })
        except Exception:
            pass


async def run_phase2_translation(
    task_id: str,
    speaker_config: Dict[str, Any],
    target_language: str = "en",
    source_language: str = "auto",
    tts_engine: str = "f5",
    is_resume: bool = False,
    resume_from_idx: int = -1,
    progress_callback: Optional[Callable[[str, int, str], None]] = None,
    translation_update_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    pre_transcribed_segments: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Convenience function to run Phase 2 translation pipeline.
    
    Args:
        task_id: The task ID
        speaker_config: Speaker configuration from validation
        target_language: Target language code
        source_language: Source language code ('auto' for auto-detect)
        tts_engine: TTS engine to use ('f5' or 'fishspeech')
        is_resume: Whether this is a resume operation
        resume_from_idx: Segment index to resume from (if known)
        progress_callback: Optional callback for progress updates
        translation_update_callback: Optional callback for progressive UI updates
        pre_transcribed_segments: Optional pre-transcribed segments from Phase 1
    
    Returns:
        True if successful, False otherwise
    """
    # Load task to get stored settings - ALWAYS use database as source of truth
    task = db.get_task(task_id)
    
    logger.info(f"run_phase2_translation called with: "
               f"target={target_language}, source={source_language}, "
               f"engine={tts_engine}")
    
    if task:
        # CRITICAL FIX: Always use database values as the authoritative source
        # The database persists across phases, so it's the ground truth
        task_tgt = task.get('tgt_lang') or task.get('target_language')
        task_src = task.get('src_lang')
        task_engine = task.get('tts_engine')
        
        logger.info(f"Task database values: "
                   f"tgt={task_tgt}, src={task_src}, engine={task_engine}")
        
        # Override function parameters with database values if they exist
        # This ensures we never lose settings between phases
        if task_tgt and str(task_tgt).strip() and str(task_tgt).strip().lower() != 'none':
            if target_language != task_tgt:
                logger.info(f"OVERRIDING target_language from DB: {target_language} -> {task_tgt}")
                target_language = task_tgt
        
        if task_src and str(task_src).strip() and str(task_src).strip().lower() != 'none':
            if source_language != task_src:
                logger.info(f"OVERRIDING source_language from DB: {source_language} -> {task_src}")
                source_language = task_src
        
        if task_engine and str(task_engine).strip() and str(task_engine).strip().lower() != 'none':
            if tts_engine != task_engine:
                logger.info(f"OVERRIDING tts_engine from DB: {tts_engine} -> {task_engine}")
                tts_engine = task_engine
        
        # Get pre_transcribed_segments from task if not provided
        if pre_transcribed_segments is None and task.get('transcribed_segments'):
            pre_transcribed_segments = task.get('transcribed_segments')
            logger.info(f"Loaded {len(pre_transcribed_segments)} "
                       f"pre-transcribed segments from task")
    
    # Ensure defaults - but use what we got from database
    target_language = target_language or "en"
    source_language = source_language or "auto"
    pre_transcribed_segments = pre_transcribed_segments or []
    
    logger.info(f"Final values for Phase 2: target={target_language}, "
               f"source={source_language}, engine={tts_engine}")
    
    pipeline = Phase2TranslationPipeline(
        task_id=task_id,
        speaker_config=speaker_config,
        target_language=target_language,
        source_language=source_language,
        tts_engine=tts_engine,
        progress_callback=progress_callback,
        translation_update_callback=translation_update_callback,
        pre_transcribed_segments=pre_transcribed_segments
    )
    
    return await pipeline.run()


# Need Path for the _load_segments method
from pathlib import Path
