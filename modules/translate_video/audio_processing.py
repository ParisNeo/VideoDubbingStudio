import subprocess
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import torch
import tempfile
import logging

from core.resources import manager

logger = logging.getLogger("audio_processing")


def extract_audio(video_path: str, output_wav: str, sample_rate: int = 16000) -> str:
    """Extract audio from video to WAV format."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        output_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    return output_wav


def separate_background_foreground(audio_path: str, output_dir: Path) -> Dict[str, str]:
    """
    Separate audio into vocals (speech) and background (music/noise) using Demucs.
    Returns paths to separated stems.
    """
    if not DEMUX_AVAILABLE:
        # Fallback: just return original as vocals, no background
        return {
            'vocals': audio_path,
            'background': None,
            'other': None,
            'drums': None,
            'bass': None
        }
    
    import torchaudio
    
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    
    # Ensure stereo for Demucs
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    
    # Pad to avoid edge artifacts
    wav = torch.nn.functional.pad(wav, (0, 44100))  # 1 second padding
    
    # Apply Demucs
    model = manager.get_demucs()
    device = next(model.parameters()).device
    
    wav = wav.to(device)
    with torch.no_grad():
        sources = apply_model(model, wav[None], split=True, overlap=0.25)[0]
    
    # sources: [n_sources, n_channels, time]
    # Order: drums, bass, other, vocals
    
    sources = sources[:, :, :-44100]  # Remove padding
    
    # Reconstruct: vocals = vocals, background = drums + bass + other
    vocals = sources[3]  # vocals
    background = sources[0] + sources[1] + sources[2]  # drums + bass + other
    
    # Convert to mono and save
    def save_stem(tensor, name):
        mono = tensor.mean(dim=0).cpu().numpy()
        path = output_dir / f"{name}.wav"
        sf.write(path, mono, sr)
        return str(path)
    
    return {
        'vocals': save_stem(vocals, 'vocals'),
        'background': save_stem(background, 'background'),
        'drums': save_stem(sources[0], 'drums'),
        'bass': save_stem(sources[1], 'bass'),
        'other': save_stem(sources[2], 'other')
    }


def load_audio_segment(audio_path: str, start_sec: float, end_sec: float, 
                       target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load a specific segment of audio."""
    # Use soundfile for precise seeking
    info = sf.info(audio_path)
    sr = info.samplerate
    
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    duration_samples = end_sample - start_sample
    
    with sf.SoundFile(audio_path) as f:
        f.seek(start_sample)
        audio = f.read(duration_samples)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    return audio, sr


def normalize_audio(audio: np.ndarray, target_db: float = -14) -> np.ndarray:
    """Normalize audio to target LUFS-like level (simplified)."""
    if len(audio) == 0:
        return audio
    
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-10:
        return audio
    
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / current_rms
    
    # Soft limit to prevent clipping
    audio = audio * gain
    audio = np.tanh(audio * 0.8) / 0.8  # Soft clip
    
    return audio


def mix_audio_tracks(tracks: List[Tuple[np.ndarray, float]], 
                     sample_rate: int = 16000) -> np.ndarray:
    """
    Mix multiple audio tracks with individual gains.
    tracks: list of (audio_array, gain_db)
    """
    # Find max length
    max_len = max(len(t[0]) for t in tracks)
    
    # Mix
    mixed = np.zeros(max_len, dtype=np.float64)
    
    for audio, gain_db in tracks:
        gain = 10 ** (gain_db / 20)
        # Pad if shorter
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        mixed += audio * gain
    
    # Final normalization
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak * 0.95
    
    return mixed.astype(np.float32)


def get_audio_duration(audio_path: str) -> float:
    """Get duration in seconds."""
    info = sf.info(audio_path)
    return info.duration


def slice_audio_chunks(audio_path: str, chunk_duration: float = 30.0, 
                       overlap: float = 1.0) -> List[Tuple[float, float, np.ndarray]]:
    """
    Slice audio into overlapping chunks for processing.
    Returns list of (start_sec, end_sec, audio_array)
    """
    info = sf.info(audio_path)
    duration = info.duration
    sr = info.samplerate
    
    chunks = []
    start = 0.0
    
    while start < duration:
        end = min(start + chunk_duration, duration)
        
        # Load chunk
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        with sf.SoundFile(audio_path) as f:
            f.seek(start_sample)
            audio = f.read(end_sample - start_sample)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16k if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        chunks.append((start, end, audio))
        
        # Move start, accounting for overlap (except last chunk)
        if end >= duration:
            break
        start = end - overlap
    
    return chunks
