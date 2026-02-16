"""
Legacy logic module - now delegates to pipeline modules.

For new code, import directly from pipeline modules:
- from .pipeline.phase1_diarization import run_phase1
- from .pipeline.phase2_translation import run_phase2  
- from .pipeline.phase3_recomposition import run_phase3
"""

import asyncio
import warnings
from typing import Dict, Any, Callable, Optional

# Re-export main pipeline functions for backward compatibility
from .pipeline.phase1_diarization import run_phase1 as _run_phase1
from .pipeline.phase2_translation import run_phase2 as _run_phase2
from .pipeline.phase3_recomposition import run_phase3 as _run_phase3

# Re-export WebSocket utilities from state module
from .state import active_connections


async def start_identification_task(task_id: str, video_path: str, 
                                   progress_callback: Optional[Callable] = None):
    """
    Phase 1: Speaker identification and transcription.
    Delegates to pipeline.phase1_diarization.run_phase1
    """
    return await _run_phase1(task_id, video_path, 
                            source_language="auto",
                            progress_callback=progress_callback)


async def start_dubbing_task(task_id: str, speaker_config: Dict[str, Any], 
                            is_resume: bool = False, 
                            resume_from_idx: int = -1,
                            progress_callback: Optional[Callable] = None):
    """
    Phase 2: Translation and TTS synthesis.
    Delegates to pipeline.phase2_translation.run_phase2
    """
    from core.database import db
    
    task = db.get_task(task_id)
    target_language = task.get('tgt_lang', 'en') if task else 'en'
    source_language = task.get('src_lang', 'auto') if task else 'auto'
    tts_engine = task.get('tts_engine', 'f5') if task else 'f5'
    
    return await _run_phase2(task_id, speaker_config, 
                            target_language=target_language,
                            source_language=source_language,
                            tts_engine=tts_engine)


async def start_assembly_task(task_id: str, original_video_path: str,
                              use_demucs: bool = False,
                              progress_callback: Optional[Callable] = None):
    """
    Phase 3: Final video assembly.
    Delegates to pipeline.phase3_recomposition.run_phase3
    """
    return await _run_phase3(task_id, original_video_path,
                            use_demucs=use_demucs,
                            progress_callback=progress_callback)


# Re-export data models for convenience
from .pipeline.phase1_diarization import DiarizationResult, SpeechSegment
from .pipeline.phase2_translation import TranslationResult, TranslationSegment, TranslationEngine, TTSEngine
from .pipeline.phase3_recomposition import AudioSegment
