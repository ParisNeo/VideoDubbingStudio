"""
Workflow package - simplified to direct pipeline calls.

This package now just re-exports from the pipeline modules.
"""

# Import directly from pipeline
from ..pipeline.phase1_diarization import (
    run_phase1,
    DiarizationResult,
    SpeechSegment,
)

from ..pipeline.phase2_translation import (
    run_phase2,
    TranslationResult,
    TranslationSegment,
    TranslationEngine,
    TTSEngine,
)

from ..pipeline.phase3_recomposition import (
    run_phase3,
    AudioSegment,
)

__all__ = [
    # Phase 1
    'run_phase1',
    'DiarizationResult',
    'SpeechSegment',
    
    # Phase 2
    'run_phase2',
    'TranslationResult',
    'TranslationSegment',
    'TranslationEngine',
    'TTSEngine',
    
    # Phase 3
    'run_phase3',
    'AudioSegment',
]

