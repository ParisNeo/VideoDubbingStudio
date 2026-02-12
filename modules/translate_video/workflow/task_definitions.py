"""
Core task definition framework for granular workflow management.

Provides the infrastructure for defining, registering, and executing
individual workflow tasks with full lifecycle tracking.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Any, List, Optional, Callable, Awaitable, Type
from enum import Enum, auto
from abc import ABC, abstractmethod
import asyncio
import traceback
from datetime import datetime
from pathlib import Path


class TaskStatus(str, Enum):
    """Status of an individual task execution."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Classification of task by computational characteristics."""
    CPU_BOUND = "cpu_bound"           # Heavy CPU, no special hardware
    GPU_BOUND = "gpu_bound"           # Requires GPU, high VRAM
    IO_BOUND = "io_bound"             # File/network operations
    EXTERNAL_API = "external_api"     # Calls to external services
    USER_INTERACTION = "user_interaction"  # Requires user input


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    checkpoint_data: Optional[Dict[str, Any]] = None  # For resume within task
    metrics: Dict[str, Any] = field(default_factory=dict)  # Timing, memory, etc.
    
    @classmethod
    def success_result(cls, **output_data) -> "TaskResult":
        """Create a successful result with output data."""
        return cls(success=True, output_data=output_data)
    
    @classmethod
    def failure_result(cls, error: Exception, checkpoint: Optional[Dict] = None) -> "TaskResult":
        """Create a failed result with error details."""
        return cls(
            success=False,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            checkpoint_data=checkpoint
        )
    
    @classmethod
    def skipped_result(cls, reason: str = "") -> "TaskResult":
        """Create a skipped result."""
        return cls(success=True, output_data={"skipped": True, "reason": reason})


@dataclass
class TaskContext:
    """
    Runtime context passed to each task execution.
    Contains all state needed for the task to run and checkpoint.
    """
    task_id: str                      # Project/task identifier
    task_name: str                    # Name of this specific task
    workflow_task_id: Optional[int] = None  # Database ID for this task instance
    
    # Input data
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Accumulated results from previous tasks
    previous_results: Dict[str, TaskResult] = field(default_factory=dict)
    
    # Project-level persistent state
    project_state: Dict[str, Any] = field(default_factory=dict)
    
    # Paths
    work_dir: Path = field(default_factory=lambda: Path("temp_chunks"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    uploads_dir: Path = field(default_factory=lambda: Path("uploads"))
    
    # Execution control
    cancellation_event: Optional[asyncio.Event] = None
    
    # Progress reporting
    progress_callback: Optional[Callable[[str, int, str], Awaitable[None]]] = None
    log_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
    
    def get_previous_output(self, task_name: str, key: str, default=None):
        """Get a specific output from a previous task's result."""
        if task_name not in self.previous_results:
            return default
        return self.previous_results[task_name].output_data.get(key, default)
    
    def get_all_previous_outputs(self) -> Dict[str, Any]:
        """Merge all previous task outputs into a single dict."""
        merged = {}
        for result in self.previous_results.values():
            merged.update(result.output_data)
        return merged
    
    async def report_progress(self, percent: int, message: str):
        """Report progress via callback if available."""
        if self.progress_callback:
            await self.progress_callback(self.task_name, percent, message)
    
    async def log(self, message: str, style: str = "info"):
        """Log message via callback if available."""
        if self.log_callback:
            await self.log_callback(message, style)


class TaskDefinition(ABC):
    """
    Abstract base class for workflow task definitions.
    
    Each task has a single responsibility and produces a TaskResult.
    Tasks can checkpoint their progress for resumption within long operations.
    """
    
    # Task metadata - must be defined by subclasses
    name: str = ""                    # Unique task identifier
    description: str = ""             # Human-readable description
    phase: str = "init"               # Which macro phase this belongs to
    task_group: Optional[str] = None  # Sub-group within phase (e.g., "diarization")
    task_type: TaskType = TaskType.CPU_BOUND
    
    # Execution constraints
    max_attempts: int = 3
    timeout_seconds: Optional[float] = None
    
    # GPU requirements (for GPU_BOUND tasks)
    required_vram_gb: Optional[float] = None  # Minimum VRAM required
    gpu_exclusive: bool = False       # Whether to unload other models
    
    # Dependencies - use simple class variable instead of field() to avoid Field object issues
    depends_on: List[str] = []        # Names of tasks that must complete first
    
    # Whether this task can be resumed from a checkpoint
    supports_checkpoints: bool = False
    
    @classmethod
    @abstractmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        """
        Execute the task. Must be implemented by subclasses.
        
        For long-running tasks, implementations should periodically:
        1. Check context.cancellation_event.is_set()
        2. Call context.report_progress()
        3. Return checkpoint data if interrupted
        """
        raise NotImplementedError
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        """Get list of task names this task depends on."""
        # Handle both direct list and field() wrapper for backward compatibility
        deps = cls.depends_on
        if hasattr(deps, 'default_factory'):
            # It's a Field object, get the default
            return deps.default_factory() if deps.default_factory else []
        elif hasattr(deps, 'default'):
            # It's a Field with default value
            return deps.default if deps.default is not None else []
        return list(deps) if deps else []
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Serialize task definition to dictionary."""
        # Get dependencies properly
        deps = cls.get_dependencies()
        
        return {
            'name': cls.name,
            'description': cls.description,
            'phase': cls.phase,
            'task_group': cls.task_group,
            'task_type': cls.task_type.value,
            'max_attempts': cls.max_attempts,
            'timeout_seconds': cls.timeout_seconds,
            'required_vram_gb': cls.required_vram_gb,
            'gpu_exclusive': cls.gpu_exclusive,
            'depends_on': deps,
            'supports_checkpoints': cls.supports_checkpoints
        }


class TaskRegistry:
    """Registry of all available task definitions."""
    
    _tasks: Dict[str, Type[TaskDefinition]] = {}
    
    @classmethod
    def register(cls, task_class: Type[TaskDefinition]) -> Type[TaskDefinition]:
        """Decorator to register a task class."""
        if not task_class.name:
            raise ValueError(f"Task class {task_class.__name__} must have a name")
        
        cls._tasks[task_class.name] = task_class
        return task_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[TaskDefinition]]:
        """Get a task class by name."""
        return cls._tasks.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, Type[TaskDefinition]]:
        """Get all registered tasks."""
        return dict(cls._tasks)
    
    @classmethod
    def get_by_phase(cls, phase: str) -> List[Type[TaskDefinition]]:
        """Get all tasks belonging to a phase."""
        return [t for t in cls._tasks.values() if t.phase == phase]
    
    @classmethod
    def validate_dependencies(cls) -> List[str]:
        """Validate that all task dependencies exist."""
        errors = []
        for name, task_class in cls._tasks.items():
            for dep in task_class.get_dependencies():
                if dep not in cls._tasks:
                    errors.append(f"Task '{name}' depends on unknown task '{dep}'")
        return errors


class WorkflowBuilder:
    """
    Builder for constructing linear or branching task workflows.
    """
    
    def __init__(self):
        self.tasks: List[Dict[str, Any]] = []
        self._task_names: set = set()
    
    def add_task(self, task_class: Type[TaskDefinition], 
                 inputs: Optional[Dict[str, Any]] = None,
                 condition: Optional[Callable[[TaskContext], bool]] = None) -> "WorkflowBuilder":
        """Add a task to the workflow."""
        if task_class.name in self._task_names:
            raise ValueError(f"Task '{task_class.name}' already added to workflow")
        
        task_def = {
            'name': task_class.name,
            'phase': task_class.phase,
            'task_group': task_class.task_group,
            'task_type': task_class.task_type.value,
            'max_attempts': task_class.max_attempts,
            'inputs': inputs or {},
            'depends_on': task_class.get_dependencies(),  # Use getter method
            'condition': condition
        }
        
        self.tasks.append(task_def)
        self._task_names.add(task_class.name)
        return self
    
    def add_conditional(self, condition: Callable[[TaskContext], bool],
                       if_true: List[Type[TaskDefinition]],
                       if_false: List[Type[TaskDefinition]] = None) -> "WorkflowBuilder":
        """Add a conditional branch (evaluated at runtime)."""
        # For now, we add all tasks but mark them with conditions
        # The executor will evaluate conditions
        for task_class in if_true:
            self.add_task(task_class, condition=condition)
        
        if if_false:
            neg_condition = lambda ctx: not condition(ctx)
            for task_class in if_false:
                self.add_task(task_class, condition=neg_condition)
        
        return self
    
    def build(self) -> List[Dict[str, Any]]:
        """Build and return the task definitions list."""
        # Validate dependencies
        for i, task in enumerate(self.tasks):
            for dep in task.get('depends_on', []):
                dep_index = next((j for j, t in enumerate(self.tasks) 
                                if t['name'] == dep), None)
                if dep_index is None:
                    raise ValueError(f"Task '{task['name']}' depends on '{dep}' which is not in workflow")
                if dep_index >= i:
                    raise ValueError(f"Task '{task['name']}' depends on '{dep}' which comes after it")
        
        return self.tasks.copy()
    
    @classmethod
    def create_default_workflow(cls, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create the default video dubbing workflow."""
        from .tasks import (
            # Phase 1: Audio & Diarization
            ExtractAudioTask, RunVADTask, ExtractEmbeddingsTask,
            ClusterSpeakersTask, ExtractSpeakerSamplesTask, TranscribeSegmentsTask,
            
            # Phase 2: Translation (after validation)
            TranslateSegmentTask, SynthesizeSegmentTask,
            
            # Phase 3: Assembly
            SeparateBackgroundAudioTask, BuildSpeechTrackTask,
            MixAudioTracksTask, MergeVideoTask
        )
        
        use_demucs = config.get('separate_audio', False)
        
        builder = cls()
        
        # ========== PHASE 1: AUDIO EXTRACTION & SPEAKER IDENTIFICATION ==========
        
        # 1.1 Audio Extraction
        builder.add_task(ExtractAudioTask, inputs={
            'video_path': config.get('video_path'),
            'output_sample_rate': 16000
        })
        
        # 1.2 Voice Activity Detection
        builder.add_task(RunVADTask, inputs={
            'min_speech_duration': 0.5,
            'threshold': 0.5
        })
        
        # 1.3 Speaker Embedding Extraction
        builder.add_task(ExtractEmbeddingsTask, inputs={
            'batch_size': 32
        })
        
        # 1.4 Speaker Clustering
        builder.add_task(ClusterSpeakersTask, inputs={
            'min_speakers': 1,
            'max_speakers': 10
        })
        
        # 1.5 Extract Reference Samples for Each Speaker
        builder.add_task(ExtractSpeakerSamplesTask, inputs={
            'sample_duration': 10.0
        })
        
        # 1.6 Transcribe All Segments
        builder.add_task(TranscribeSegmentsTask, inputs={
            'source_language': config.get('src_lang', 'auto'),
            'batch_size': 4
        })
        
        # ========== PHASE 2: TRANSLATION & SYNTHESIS (POST-VALIDATION) ==========
        
        # Note: Validation is handled separately via user interaction
        
        # 2.1 Translate Segments (runs after user validates speakers)
        builder.add_task(TranslateSegmentTask, inputs={
            'target_language': config.get('tgt_lang', 'en'),
            'source_language': config.get('src_lang', 'auto')
        })
        
        # 2.2 Synthesize Speech with Voice Cloning
        builder.add_task(SynthesizeSegmentTask, inputs={
            'tts_engine': config.get('tts_engine', 'f5'),
            'target_sample_rate': 24000
        })
        
        # ========== PHASE 3: FINAL ASSEMBLY ==========
        
        # 3.1 Optional: Separate Background Audio
        if use_demucs:
            builder.add_task(SeparateBackgroundAudioTask, inputs={
                'model_name': 'htdemucs'
            })
        
        # 3.2 Build Continuous Speech Track
        builder.add_task(BuildSpeechTrackTask, inputs={
            'apply_crossfade': True,
            'crossfade_ms': 20
        })
        
        # 3.3 Mix with Background
        builder.add_task(MixAudioTracksTask, inputs={
            'speech_gain_db': -6,
            'background_gain_db': -20,
            'ducking_threshold': 0.01
        })
        
        # 3.4 Final Video Merge
        builder.add_task(MergeVideoTask, inputs={
            'video_codec': 'copy',
            'audio_codec': 'aac',
            'audio_bitrate': '192k'
        })
        
        return builder.build()
