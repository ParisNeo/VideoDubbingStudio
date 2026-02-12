"""
Granular Workflow Task System

This package provides a task-based workflow engine where each phase is
decomposed into discrete, single-responsibility tasks that can be
tracked, resumed, and debugged individually.
"""

from .task_definitions import (
    TaskDefinition,
    TaskRegistry,
    WorkflowBuilder,
    TaskType,
    TaskResult,
    TaskContext
)

from .task_executor import TaskExecutor, TaskExecutionError

from .tasks import (
    # Phase 1 tasks
    ExtractAudioTask,
    RunVADTask,
    ExtractEmbeddingsTask,
    ClusterSpeakersTask,
    ExtractSpeakerSamplesTask,
    TranscribeSegmentsTask,
    
    # Phase 2 tasks
    TranslateSegmentTask,
    SynthesizeSegmentTask,
    
    # Phase 3 tasks
    SeparateBackgroundAudioTask,
    BuildSpeechTrackTask,
    MixAudioTracksTask,
    MergeVideoTask
)

__all__ = [
    # Core classes
    'TaskDefinition',
    'TaskRegistry',
    'WorkflowBuilder',
    'TaskType',
    'TaskResult',
    'TaskContext',
    'TaskExecutor',
    'TaskExecutionError',
    
    # Task implementations
    'ExtractAudioTask',
    'RunVADTask',
    'ExtractEmbeddingsTask',
    'ClusterSpeakersTask',
    'ExtractSpeakerSamplesTask',
    'TranscribeSegmentsTask',
    'TranslateSegmentTask',
    'SynthesizeSegmentTask',
    'SeparateBackgroundAudioTask',
    'BuildSpeechTrackTask',
    'MixAudioTracksTask',
    'MergeVideoTask'
]
