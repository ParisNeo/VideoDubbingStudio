"""
Phase 2: Shared Data Models

Contains the TranslationSegment dataclass used across all Phase 2 subphases.
Centralized here to avoid circular imports between subphase modules.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class TranslationSegment:
    """Represents a single segment's translation state."""
    idx: int
    start: float  # original timestamp in seconds
    end: float
    speaker_id: int
    speaker_name: str = ""  # User-defined name from validation
    original_text: str = ""
    translated_text: str = ""
    audio_path: Optional[str] = None  # Path to generated TTS audio
    status: str = "pending"  # pending, transcribing, translating, synthesizing, completed, failed
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
