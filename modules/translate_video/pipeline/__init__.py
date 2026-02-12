"""
Video Translation Pipeline - Modular Phase Implementation

This package contains the phase-by-phase implementation of the video translation pipeline:
- Phase 1: Speaker Identification (diarization) - now in pipeline folder
  - phase1_diarization.py: Main orchestrator
  - phase1_models.py: Shared data classes
- Phase 2: Translation - split into 3 subphases:
  - phase2_subphase1_transcription.py: Whisper STT
  - phase2_subphase2_translation.py: Lollms text translation  
  - phase2_subphase3_tts.py: F5-TTS/FishSpeech voice synthesis
  - phase2_translation.py: Orchestrator that calls all 3 subphases
  - phase2_models.py: Shared TranslationSegment dataclass
- Phase 3: Recomposition (final video assembly) - in phase3_recomposition.py

Each subphase is self-contained with clear inputs/outputs for resumability
and VRAM-efficient model loading/unloading.
"""

# Phase 1 exports
from .phase1_diarization import run_phase1_diarization, Phase1Diarizer
from .phase1_models import DiarizationResult, SpeechSegment

# Phase 2 exports
from .phase2_translation import run_phase2_translation, Phase2TranslationPipeline
from .phase2_models import TranslationSegment

# Phase 3 exports
from .phase3_recomposition import run_phase3_recomposition, Phase3Recomposer

# Subphase exports (for advanced use)
from .phase2_subphase1_transcription import run_transcription_subphase, TranscriptionSubphase
from .phase2_subphase2_translation import run_translation_subphase, TranslationSubphase
from .phase2_subphase3_tts import run_tts_subphase, TTSSubphase

__all__ = [
    # Phase 1
    'run_phase1_diarization',
    'Phase1Diarizer',
    'DiarizationResult',
    'SpeechSegment',
    # Phase 2
    'run_phase2_translation',
    'Phase2TranslationPipeline',
    'TranslationSegment',
    # Phase 3
    'run_phase3_recomposition',
    'Phase3Recomposer',
    # Subphases
    'run_transcription_subphase',
    'TranscriptionSubphase',
    'run_translation_subphase',
    'TranslationSubphase',
    'run_tts_subphase',
    'TTSSubphase',
]
