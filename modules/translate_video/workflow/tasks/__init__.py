"""
Task implementations for the granular workflow system.

Each task is a single-responsibility unit that can be tracked and resumed independently.
"""

from .phase1_audio_tasks import (
    ExtractAudioTask,
    RunVADTask,
    ExtractEmbeddingsTask,
    ClusterSpeakersTask,
    ExtractSpeakerSamplesTask,
    TranscribeSegmentsTask
)

from .phase2_translation_tasks import (
    TranslateSegmentTask,
    SynthesizeSegmentTask
)

from .phase3_assembly_tasks import (
    SeparateBackgroundAudioTask,
    BuildSpeechTrackTask,
    MixAudioTracksTask,
    MergeVideoTask
)

__all__ = [
    # Phase 1
    'ExtractAudioTask',
    'RunVADTask',
    'ExtractEmbeddingsTask',
    'ClusterSpeakersTask',
    'ExtractSpeakerSamplesTask',
    'TranscribeSegmentsTask',
    
    # Phase 2
    'TranslateSegmentTask',
    'SynthesizeSegmentTask',
    
    # Phase 3
    'SeparateBackgroundAudioTask',
    'BuildSpeechTrackTask',
    'MixAudioTracksTask',
    'MergeVideoTask'
]
