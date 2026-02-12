"""
Phase 1: Shared Data Models

Contains data classes used by the Phase 1 diarization pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from numpy import ndarray


@dataclass
class SpeechSegment:
    """Represents a detected speech segment with timing and speaker info."""
    idx: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker_id: Optional[int] = None  # Assigned speaker ID after clustering
    embedding: Optional[ndarray] = None  # Speaker embedding vector


@dataclass
class DiarizationResult:
    """Complete result from Phase 1 diarization."""
    segments: List[SpeechSegment]
    speaker_count: int
    speaker_samples: Dict[int, str]  # Map speaker_id -> sample audio path
    assignments: Dict[int, int]  # Map segment_idx -> speaker_id
    transcribed_segments: List[Dict[str, Any]]  # For UI preview
    master_audio: str  # Path to extracted master audio file
    speaker_config: Dict[str, Any]  # Config for validation UI
