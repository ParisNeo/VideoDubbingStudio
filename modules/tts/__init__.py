"""
TTS module with SVML-safe exports.
"""

from .logic import (
    generate_speech,
    generate_speech_f5_robust,
    generate_speech_fishspeech,
    generate_speech_lollms,
    get_default_tts_engine,
    get_available_voices_lollms,
)

__all__ = [
    'generate_speech',
    'generate_speech_f5_robust',
    'generate_speech_fishspeech',
    'generate_speech_lollms',
    'get_default_tts_engine',
    'get_available_voices_lollms',
]
