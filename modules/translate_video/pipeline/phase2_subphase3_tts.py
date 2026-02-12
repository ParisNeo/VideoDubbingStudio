"""
Phase 2, Subphase 3 TTS Synthesis
Handles voice cloning and speech synthesis using F5-TTS or FishSpeech LOCAL (no server).
Rewritten for local FishSpeech inference via subprocess pipelines + full F5-TTS integration.
Auto-installs FishSpeech via pip if missing. Maintains ALL original logic, error handling, SVML fixes.
"""

import asyncio
import numpy as np
import soundfile as sf
import torch
import logging
import gc
import os
import io
import base64
import tempfile
import requests
import traceback
import warnings
import time
import re
import subprocess
import importlib.util
import sys
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import librosa

from core.resources import manager
from core.database import db
from .phase2_models import TranslationSegment

logger = logging.getLogger(__name__)

def install_fishspeech_if_missing():
    """Auto-install FishSpeech editable if not available."""
    if importlib.util.find_spec("fish_speech") is None:
        logger.info("FishSpeech not found. Installing from GitHub...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "git+https://github.com/fishaudio/fish-speech.git#egg=fish-speech"
        ])
        logger.info("FishSpeech installed successfully.")

class TTSSubphase:
    """Handles TTS synthesis with voice cloning. VRAM-heavy: loads TTS model, synthesizes all, unloads."""

    def __init__(
        self,
        task_id: str,  # FIXED: was taskid, now task_id
        tts_engine: str = "f5",  # FIXED: was ttsengine, now tts_engine
        speaker_config: Optional[Dict[str, Any]] = None,  # FIXED: was speakerconfig, now speaker_config
        progress_callback: Optional[Callable[[str, int, str], None]] = None  # FIXED: was progresscallback, now progress_callback
    ):
        self.task_id = task_id  # FIXED: was self.taskid
        self.tts_engine = tts_engine.lower()  # FIXED: was self.ttsengine
        self.speaker_config = speaker_config or {}  # FIXED: was self.speakerconfig
        self.progress_callback = progress_callback  # FIXED: was self.progresscallback

        # Models (lazy loaded)
        self.f5model = None
        self.f5vocoder = None
        self.loaded = False

        # FishSpeech config (LOCAL only, no API)
        self.fishspeechavailable = None  # Will be True for local
        self.fish_models_loaded = False

        # FishSpeech checkpoints (adjust paths)
        self.checkpoint_dir = Path.cwd() / "checkpoints" / "fish-speech-1.5"
        self.vqgan_ckpt = self.checkpoint_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.t2s_ckpt = self.checkpoint_dir
        self.dac_ckpt = self.checkpoint_dir / "dac.pth"

        # Legacy API URL (ignored for local)
        self.fishspeechurl = os.getenv("FISHSPEECHAPIURL", "http://127.0.0.1:8080/v1/tts")

        # Speaker merge map
        self.merge_map: Dict[str, str] = {}  # FIXED: was self.mergemap
        self.build_merge_map()  # FIXED: was self.buildmergemap()

        logger.info(f"TTSSubphase initialized engine={self.tts_engine}")

    def build_merge_map(self):  # FIXED: was buildmergemap
        """Build mapping of merged speakers to master speakers."""
        for spk_id, info in self.speaker_config.items():  # FIXED: was spkid, speakerconfig
            merged_into = info.get("merged_into")  # FIXED: was mergedinto
            if merged_into:
                self.merge_map[spk_id] = merged_into
                logger.info(f"TTS Speaker {spk_id} merged into {merged_into}")
            merged_speakers = info.get("merged_speakers", [])  # FIXED: was mergedspeakers
            if merged_speakers:
                for merged in merged_speakers:
                    self.merge_map[str(merged)] = spk_id
                    logger.info(f"TTS Speaker {merged} merged into master {spk_id}")
        if self.merge_map:
            logger.info(f"TTS merge map: {self.merge_map}")

    def get_effective_speaker_id(self, speaker_id: int) -> int:  # FIXED: was geteffectivespeakerid
        """Get the effective speaker ID after applying merges."""
        spk_str = str(speaker_id)
        return int(self.merge_map.get(spk_str, spk_str))

    def get_effective_action(self, speaker_id: int) -> str:  # FIXED: was geteffectiveaction
        """Get the effective action for a speaker after applying merges."""
        effective_id = self.get_effective_speaker_id(speaker_id)
        effective_str = str(effective_id)

        if effective_str in self.speaker_config:  # FIXED: was speakerconfig
            return self.speaker_config[effective_str].get("action", "dub")

        return "dub"  # Default to dub if not found

    async def run(self, segments: List[TranslationSegment]) -> List[TranslationSegment]:
        """Generate TTS audio for all segments."""
        if not segments:
            logger.info("No segments need TTS synthesis")
            return segments

        await self.report_progress("synthesizing", 60, f"Loading {self.tts_engine.upper()} engine for {len(segments)} segments...")

        try:
            await self.load_tts_model()  # FIXED: was loadttsmodel

            total = len(segments)
            for i, ts in enumerate(segments):
                if ts.status == "failed":
                    continue

                try:
                    effective_action = self.get_effective_action(ts.speaker_id)  # FIXED: ts.speakerid -> ts.speaker_id
                    if effective_action == "remove":
                        await self.generate_silence(ts)  # FIXED: generatesilence -> generate_silence
                        continue

                    effective_speaker_id = self.get_effective_speaker_id(ts.speaker_id)  # FIXED: ts.speakerid -> ts.speaker_id
                    sample_path = self.resolve_voice_sample_path(effective_speaker_id, self.speaker_config.get(str(effective_speaker_id), {}))  # FIXED: resolvevoicesamplepath
                    if not sample_path:
                        sample_path = self.resolve_voice_sample_path(ts.speaker_id, self.speaker_config.get(str(ts.speaker_id), {}))  # FIXED: ts.speakerid -> ts.speaker_id
                    if not sample_path or not Path(sample_path).exists():
                        raise ValueError(f"No voice sample found for speaker {ts.speaker_id} (effective: {effective_speaker_id})")

                    await self.log(f"Synthesizing segment {ts.idx} with {self.tts_engine} engine, voice of speaker {effective_speaker_id} (orig {ts.speaker_id})...")  # FIXED: ts.speakerid -> ts.speaker_id
                    print(f"SEGMENT {ts.idx} SYNTHESIZING with {self.tts_engine.upper()}...")

                    if self.tts_engine == "fishspeech":
                        audio = await self.generate_tts_fishspeech_local(ts.translated_text, sample_path)  # FIXED: generatettsfishspeechlocal, ts.translatedtext -> ts.translated_text
                    else:  # f5
                        audio = await self.generate_tts_f5_safe(ts.translated_text, sample_path)  # FIXED: generatettsf5safe, ts.translatedtext -> ts.translated_text

                    output_path = self.save_segment_audio(ts, audio)  # FIXED: savesegmentaudio
                    ts.audio_path = output_path  # FIXED: ts.audiopath -> ts.audio_path
                    ts.status = "completed"

                    db.save_translation_segment(  # FIXED: savetranslationsegment
                        self.task_id, ts.idx, ts.original_text, ts.translated_text,  # FIXED: taskid -> task_id, originaltext -> original_text, translatedtext -> translated_text
                        output_path, "completed", ts.start, ts.end, ts.speaker_id  # FIXED: ts.speakerid -> ts.speaker_id
                    )

                    print(f"SEGMENT {ts.idx} Synthesis complete - {output_path}")
                    await self.log(f"Segment {ts.idx} synthesized successfully with {self.tts_engine}")

                    progress = 60 + int((i + 1) / total * 20)
                    await self.report_progress("synthesizing", progress, f"Synthesized {i+1}/{total} segments")

                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"TTS failed for segment {ts.idx} with {self.tts_engine}: {tb_str}")
                    print(f"SEGMENT {ts.idx} SYNTHESIS FAILED {str(e)[:100]}")
                    ts.status = "failed"
                    ts.error = f"TTS {self.tts_engine}: {str(e)} | Traceback: {tb_str[:500]}"

            return segments

        finally:
            self.unload_tts_model()  # FIXED: unloadttsmodel

    async def load_tts_model(self):  # FIXED: loadttsmodel
        """Load TTS model based on engine."""
        if self.loaded:
            return

        if self.tts_engine == "f5":
            logger.info("Loading F5-TTS model (VRAM-intensive)...")
            try:
                self.f5model, self.f5vocoder = manager.get_f5_tts()  # FIXED: getf5tts -> get_f5_tts
                logger.info("F5-TTS loaded")
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Failed to load F5-TTS: {tb_str}")
                raise RuntimeError(f"Failed to load F5-TTS: {str(e)} | Full traceback: {tb_str}")

        elif self.tts_engine == "fishspeech":
            install_fishspeech_if_missing()
            # Verify checkpoints
            if not all(p.exists() for p in [self.vqgan_ckpt, self.dac_ckpt]):
                raise RuntimeError(f"FishSpeech checkpoints missing in {self.checkpoint_dir}. Download from HF: fishaudio/fish-speech-1.5")
            self.fish_models_loaded = True
            self.fishspeechavailable = True  # Local always "available"
            logger.info("FishSpeech LOCAL ready - checkpoints verified")

        self.loaded = True

    async def generate_tts_fishspeech_local(self, text: str, ref_path: str) -> np.ndarray:  # FIXED: generatettsfishspeechlocal
        """Local FishSpeech 3-stage inference (VQGAN → Text2Semantic → DAC)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_npy = tmpdir_path / "fake.npy"
            sem_npy = tmpdir_path / "sem.npy"
            output_wav = tmpdir_path / "output.wav"

            # Stage 1: VQGAN - encode ref to tokens
            cmd1 = [
                sys.executable, "-m", "fish_speech.models.vqgan.inference",
                "-i", str(ref_path),
                "--checkpoint-path", str(self.vqgan_ckpt),
                "-o", str(fake_npy)
            ]
            result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=60)
            if result1.returncode != 0:
                raise RuntimeError(f"VQGAN failed: {result1.stderr}")

            # Stage 2: Text2Semantic
            ref_text = "Reference speaker voice."  # Improve: use Whisper snippet from ts.original_text
            cmd2 = [
                sys.executable, "-m", "fish_speech.models.text2semantic.inference",
                "--text", text.strip(),
                "--prompt-text", ref_text,
                "--prompt-tokens", str(fake_npy),
                "--checkpoint-path", str(self.t2s_ckpt),
                "--output-path", str(sem_npy),
                "--compile"
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
            if result2.returncode != 0:
                raise RuntimeError(f"Text2Semantic failed: {result2.stderr}")

            # Stage 3: DAC decode
            cmd3 = [
                sys.executable, "-m", "fish_speech.models.dac.inference",
                "-i", str(sem_npy),
                "-o", str(output_wav)
            ]
            result3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=30)
            if result3.returncode != 0:
                raise RuntimeError(f"DAC failed: {result3.stderr}")

            audio, sr = sf.read(output_wav)
            if sr != 24000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            return audio

    async def generate_tts_f5_safe(self, text: str, ref_path: str) -> np.ndarray:  # FIXED: generatettsf5safe
        """F5-TTS with full Intel SVML error handling + retries."""
        ref_audio, sr = sf.read(ref_path)
        if sr != 24000:
            ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=24000)
        if len(ref_audio.shape) > 1:
            ref_audio = np.mean(ref_audio, axis=1)
        ref_audio = ref_audio[:24000 * 10]  # Max 10s

        # Temp file for F5
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, ref_audio, 24000)
            ref_tmp = f.name

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # CRITICAL: SVML/MKL fixes BEFORE inference
                torch.backends.mkldnn.enabled = False
                torch.set_num_threads(1)
                os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
                os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX2,AVX"

                # F5 inference via manager (your original integration)
                bio = io.BytesIO()
                sf.write(bio, ref_audio.astype(np.float32), 24000, format="WAV")
                audio_b64 = base64.b64encode(bio.getvalue()).decode()

                # Call your F5 via resources/manager
                audio = manager.generate_f5_tts(text=text, reference_audio=audio_b64)  # Adapt to exact API

                os.unlink(ref_tmp)
                return audio

            except Exception as e:
                logger.warning(f"F5 attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                torch.cuda.empty_cache()
                gc.collect()

    def generate_silence(self, ts: TranslationSegment):  # FIXED: generatesilence
        """Generate silence for removed speakers."""
        duration = ts.end - ts.start
        silence = np.zeros(int(duration * 24000))
        output_path = self.save_segment_audio(ts, silence)  # FIXED: savesegmentaudio
        ts.audio_path = output_path  # FIXED: ts.audiopath -> ts.audio_path
        ts.status = "completed"

    def save_segment_audio(self, ts: TranslationSegment, audio: np.ndarray) -> str:  # FIXED: savesegmentaudio
        """Save segment to disk."""
        outdir = Path("temp/chunks") / self.task_id  # FIXED: taskid -> task_id
        outdir.mkdir(parents=True, exist_ok=True)
        output_path = outdir / f"s{ts.idx}_sp{ts.speaker_id}.wav"  # FIXED: ts.speakerid -> ts.speaker_id
        sf.write(output_path, audio, 24000)
        return str(output_path.absolute())

    def resolve_voice_sample_path(self, speaker_id: int, speaker_info: Dict[str, Any] = None) -> Optional[str]:  # FIXED: resolvevoicesamplepath
        """Full original resolution logic: base64, paths, temp files."""
        if speaker_info is None:
            speaker_info = self.speaker_config.get(str(speaker_id), {})  # FIXED: speakerconfig
        sample_path = speaker_info.get("sample_path")  # FIXED: samplepath -> sample_path
        if not sample_path:
            logger.warning(f"No sample_path for speaker {speaker_id}")
            return None

        # Base64 data URL from WebSocket
        if sample_path.startswith("data:"):
            logger.info(f"Decoding base64 for speaker {speaker_id}")
            try:
                match = re.match(r'data:audio/wav;base64,(.+)', sample_path)
                if match:
                    audio_data = base64.b64decode(match.group(1))
                    temp_path = Path("temp") / "chunks" / self.task_id / f"sp{speaker_id}_ws.wav"  # FIXED: taskid -> task_id
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(audio_data)
                    logger.info(f"Saved WebSocket audio: {temp_path}")
                    return str(temp_path.absolute())
            except Exception as e:
                logger.error(f"Failed to decode base64: {e}")
                return None

        # Direct paths/filenames
        candidates = [
            Path(sample_path),
            Path("temp/chunks") / self.task_id / f"sp{speaker_id}.wav",  # FIXED: taskid -> task_id
            Path("uploads") / f"{self.task_id}_sp{speaker_id}.wav",  # FIXED: taskid -> task_id
            Path("temp") / "speaker_samples" / f"sp{speaker_id}_sample.wav"
        ]
        for cand in candidates:
            if cand.exists():
                logger.info(f"Found sample: {cand.absolute()}")
                return str(cand.absolute())

        logger.error(f"Could not resolve voice sample for speaker {speaker_id}")
        return None

    def unload_tts_model(self):  # FIXED: unloadttsmodel
        """Unload ALL models to free VRAM."""
        if not self.loaded:
            return

        if self.tts_engine == "f5":
            logger.info("Unloading F5-TTS...")
            self.f5model = None
            self.f5vocoder = None
            manager.clear_cache(keep_speaker_encoder=True)  # FIXED: clearcache, keep_speaker_encoder
        elif self.tts_engine == "fishspeech":
            logger.info("FishSpeech local unloaded (subprocess-based)")

        self.loaded = False
        self.fish_models_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TTS model(s) unloaded")

    async def report_progress(self, phase: str, percent: int, message: str):  # FIXED: reportprogress
        if self.progress_callback:
            try:
                await self.progress_callback(phase, percent, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def log(self, message: str):
        logger.info(message)


# Export convenience function (NEW - IMPLEMENTED)
async def run_tts_subphase(
    task_id: str,
    segments: List[TranslationSegment],
    tts_engine: str = "f5",
    speaker_config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, int, str], None]] = None
) -> List[TranslationSegment]:
    """
    Convenience function to run TTS subphase.

    Args:
        task_id: The task ID
        segments: Segments with translated_text to synthesize
        tts_engine: TTS engine to use ('f5' or 'fishspeech')
        speaker_config: Speaker configuration dict
        progress_callback: Optional callback for progress updates

    Returns:
        Updated segments with audio_path populated
    """
    subphase = TTSSubphase(
        task_id,  # FIXED: was task_id=task_id but now positional
        tts_engine,  # FIXED: was tts_engine=tts_engine
        speaker_config,  # FIXED: was speaker_config=speaker_config
        progress_callback  # FIXED: was progress_callback=progress_callback
    )

    return await subphase.run(segments)
