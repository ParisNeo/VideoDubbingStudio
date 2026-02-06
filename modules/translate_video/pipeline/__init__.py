"""
Video Translation Pipeline - Modular Phase Implementation

This package contains the phase-by-phase implementation of the video translation pipeline:
- Phase 1: Identification (diarization) - in logic.py
- Phase 2: Translation (transcription + translation + TTS) - in phase2_translation.py
- Phase 3: Recomposition (final video assembly) - in phase3_recomposition.py

Each phase is self-contained with clear inputs/outputs for resumability.
"""

from .phase2_translation import run_phase2_translation, Phase2TranslationPipeline
from .phase3_recomposition import run_phase3_recomposition, Phase3Recomposer

__all__ = [
    'run_phase2_translation',
    'Phase2TranslationPipeline',
    'run_phase3_recomposition',
    'Phase3Recomposer',
]
