"""
Phase 1.5: Translation Review

Translates transcribed segments using LoLLMs and sends results to UI for review.
Users can edit translations before TTS synthesis begins.

This phase runs after speaker validation and before TTS synthesis.
"""

import asyncio
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from core.database import db
from .phase2_translation import TranslationEngine, TranslationSegment

logger = logging.getLogger("phase1_5_translation_review")


async def run_translation_review(
    task_id: str,
    edited_segments: Optional[List[Dict[str, Any]]] = None,  # User-edited transcriptions
    target_language: str = "en",
    source_language: str = "auto",
    progress_callback: Optional[Callable[[str, int, str], None]] = None
) -> List[TranslationSegment]:
    """
    Run translation review phase.
    
    Args:
        task_id: The task ID
        edited_segments: Optional user-edited transcriptions from UI
        target_language: Target language code
        source_language: Source language code
        progress_callback: Called with (phase, percent, message)
    
    Returns:
        List of TranslationSegment with original and translated text
    """
    
    async def report(phase: str, percent: int, message: str):
        if progress_callback:
            await progress_callback(phase, percent, message)
        logger.info(f"[Translation Review] {percent}%: {message}")
    
    try:
        # Load task and segments
        task = db.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Use edited segments if provided, otherwise use stored segments
        if edited_segments:
            segments = [TranslationSegment.from_dict(s) for s in edited_segments]
            # Update stored segments with user edits
            db.update_task(task_id, transcribed_segments=edited_segments)
            await report("loading", 10, f"Loaded {len(segments)} user-edited segments")
        else:
            seg_data = task.get('transcribed_segments', [])
            if not seg_data:
                raise ValueError("No transcribed segments found - run Phase 1 first")
            segments = [TranslationSegment.from_dict(s) for s in seg_data]
            await report("loading", 10, f"Loaded {len(segments)} segments")
        
        # Step 1: Translate all segments
        await report("translating", 20, "Starting LoLLMs translation...")
        
        translator = TranslationEngine(target_language, source_language)
        
        def trans_progress(current, total):
            pct = 20 + int((current / total) * 50)
            asyncio.create_task(report("translating", pct, 
                f"Translated {current}/{total} segments"))
        
        segments = translator.translate_batch(segments, trans_progress)
        
        translated_count = sum(1 for s in segments if s.translated_text)
        await report("translating", 70, 
            f"Translation complete: {translated_count}/{len(segments)} segments")
        
        # Save translations
        db.update_task(
            task_id,
            segments=[s.to_dict() for s in segments],
            phase="awaiting_translation_review",
            status="awaiting_validation",
            progress=50,
            message="Translation complete - review and edit before voice synthesis"
        )
        
        # Send translation review data to UI
        from ..state import broadcast_to_task
        await broadcast_to_task(task_id, {
            'type': 'translation_ready',
            'data': {
                'segments': [s.to_dict() for s in segments],
                'target_language': target_language,
                'source_language': source_language,
                'can_edit': True,
                'next_phase': 'tts_synthesis'
            }
        })
        
        await report("complete", 100, "Translation review ready - awaiting user confirmation")
        
        return segments
        
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Translation review failed:\n{tb_str}")
        
        db.update_task(
            task_id,
            status="failed",
            phase="translating",
            error_message=f"Translation review failed: {str(e)}",
            error_traceback=tb_str
        )
        
        raise RuntimeError(f"Translation review failed: {str(e)}") from e
