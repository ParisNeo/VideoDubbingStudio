"""
Video Translation Pipeline - Consolidated Phase Implementation

This package contains the three main phases of video translation:
- Phase 1: Audio Extraction & Speaker Identification (phase1_diarization.py)
- Phase 2: Translation & TTS Synthesis (phase2_translation.py)
- Phase 3: Final Video Assembly (phase3_recomposition.py)

Each phase is self-contained with clear checkpoint boundaries:
- Phase 1 checkpoints: after diarization, after transcription
- Phase 2 checkpoints: after translation, after each TTS batch
- Phase 3 checkpoints: after speech track, after audio mix
"""

from .phase1_diarization import (
    run_phase1,
    DiarizationResult,
    SpeechSegment,
    extract_audio,
    run_vad,
    SpeakerIdentifier,
    extract_speaker_samples,
    transcribe_segments
)

from .phase2_translation import (
    run_phase2,
    TranslationResult,
    TranslationSegment,
    TranslationEngine,
    TTSEngine
)

from .phase3_recomposition import (
    run_phase3,
    AudioSegment,
    build_speech_track,
    mix_audio_tracks,
    merge_with_video,
    separate_background_demucs
)

__all__ = [
    # Phase 1
    'run_phase1',
    'DiarizationResult',
    'SpeechSegment',
    'extract_audio',
    'run_vad',
    'SpeakerIdentifier',
    'extract_speaker_samples',
    'transcribe_segments',
    
    # Phase 2
    'run_phase2',
    'TranslationResult',
    'TranslationSegment',
    'TranslationEngine',
    'TTSEngine',
    
    # Phase 3
    'run_phase3',
    'AudioSegment',
    'build_speech_track',
    'mix_audio_tracks',
    'merge_with_video',
    'separate_background_demucs',
]
