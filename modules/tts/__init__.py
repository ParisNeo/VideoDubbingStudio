"""
TTS module with SVML-safe exports.
"""

from .logic import (
    generate_speech,
    generate_speech_f5_robust,
    generate_speech_fishspeech,
    get_default_tts_engine,
)

__all__ = [
    'generate_speech',
    'generate_speech_f5_robust',
    'generate_speech_fishspeech',
    'get_default_tts_engine',
]
