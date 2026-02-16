"""
Phase 2: Translation & TTS Synthesis

Complete pipeline: Load segments → Translate text → Synthesize speech with voice cloning

This module handles the core translation and voice synthesis work after
speaker validation. It processes in batches to maintain VRAM efficiency.

Checkpoints:
- After translation complete (before TTS) - allows changing TTS engine/settings
- After each batch of TTS - can resume from partial synthesis
"""

import asyncio
import json
import soundfile as sf
import numpy as np
import torch
import logging
import gc
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import base64
import io
import os
import sys
import re
import subprocess
import importlib.util

from core.resources import manager
from core.database import db

logger = logging.getLogger("phase2_translation")


# =============================================================================
# DATA MODELS
# =============================================================================

class TranslationSegment:
    """Represents a segment through Phase 2 processing."""
    def __init__(
        self,
        idx: int,
        start: float,
        end: float,
        speaker_id: int,
        original_text: str = "",
        translated_text: str = "",
        audio_path: Optional[str] = None,
        status: str = "pending",
        error: Optional[str] = None
    ):
        self.idx = idx
        self.start = start
        self.end = end
        self.speaker_id = speaker_id
        self.original_text = original_text
        self.translated_text = translated_text
        self.audio_path = audio_path
        self.status = status
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'idx': self.idx,
            'start': self.start,
            'end': self.end,
            'speaker_id': self.speaker_id,
            'original_text': self.original_text,
            'translated_text': self.translated_text,
            'audio_path': self.audio_path,
            'status': self.status,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslationSegment":
        return cls(
            idx=data['idx'],
            start=data['start'],
            end=data['end'],
            speaker_id=data['speaker_id'],
            original_text=data.get('original_text', ''),
            translated_text=data.get('translated_text', ''),
            audio_path=data.get('audio_path'),
            status=data.get('status', 'pending'),
            error=data.get('error')
        )


class TranslationResult:
    """Result from Phase 2."""
    def __init__(
        self,
        segments: List[TranslationSegment],
        translated_count: int,
        synthesized_count: int,
        failed_count: int
    ):
        self.segments = segments
        self.translated_count = translated_count
        self.synthesized_count = synthesized_count
        self.failed_count = failed_count
    
    @property
    def success(self) -> bool:
        return self.failed_count == 0 and self.synthesized_count > 0


# =============================================================================
# TRANSLATION
# =============================================================================

class TranslationEngine:
    """Handles text translation using Lollms."""
    
    def __init__(self, target_language: str = "en", source_language: str = "auto"):
        self.target_language = target_language
        self.source_language = source_language
        self.lollms = None
    
    def _load(self):
        """Load Lollms client."""
        self.lollms = manager.get_lollms_client()
    
    def _unload(self):
        """Free resources."""
        self.lollms = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _should_translate(self) -> bool:
        """Check if translation is needed or if source == target."""
        if self.source_language == self.target_language:
            return False
        
        # Language code aliases
        aliases = {
            'en': ['english', 'eng'],
            'es': ['spanish', 'spa'],
            'fr': ['french', 'fra'],
            'de': ['german', 'deu'],
            'it': ['italian', 'ita'],
            'pt': ['portuguese', 'por'],
            'zh': ['chinese', 'chi', 'zho'],
            'ja': ['japanese', 'jpn'],
            'ko': ['korean', 'kor'],
            'ar': ['arabic', 'ara'],
            'ru': ['russian', 'rus'],
        }
        
        src = self.source_language.lower()
        tgt = self.target_language.lower()
        
        # Direct match
        if src == tgt:
            return False
        
        # Check aliases
        for code, names in aliases.items():
            all_names = [code] + names
            if src in all_names and tgt in all_names:
                return False
        
        return True
    
    def _build_prompt(self, text: str) -> str:
        """Build translation prompt."""
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'zh': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'ru': 'Russian',
            'hi': 'Hindi', 'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish',
            'vi': 'Vietnamese', 'th': 'Thai'
        }
        
        lang_name = lang_names.get(self.target_language, self.target_language)
        
        return f"""Translate the following text to {lang_name}. 
Preserve the meaning, tone, and style. Only output the translation, no explanations.

Text: {text}

Translation to {lang_name}:"""
    
    def translate(self, text: str) -> str:
        """Translate a single text segment."""
        if not self.lollms:
            self._load()
        
        if not self.lollms:
            # Fallback: return original
            return text
        
        try:
            prompt = self._build_prompt(text)
            result = self.lollms.generate_text(prompt, temperature=0.3)
            return result.strip()
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text  # Fallback
    
    def translate_batch(self, 
                       segments: List[TranslationSegment],
                       progress_callback: Optional[Callable[[int, int], None]] = None
                       ) -> List[TranslationSegment]:
        """Translate a batch of segments."""
        if not self._should_translate():
            # Copy original text
            for seg in segments:
                seg.translated_text = seg.original_text
                seg.status = "translated"
            return segments
        
        self._load()
        
        try:
            total = len(segments)
            for i, seg in enumerate(segments):
                if not seg.original_text or seg.status == "failed":
                    continue
                
                try:
                    seg.translated_text = self.translate(seg.original_text)
                    seg.status = "translated"
                    
                    if progress_callback:
                        progress_callback(i + 1, total)
                        
                except Exception as e:
                    logger.error(f"Failed to translate segment {seg.idx}: {e}")
                    seg.translated_text = seg.original_text
                    seg.status = "translated"  # Continue with original
            
            return segments
            
        finally:
            self._unload()


# =============================================================================
# TTS SYNTHESIS
# =============================================================================

def _is_svml_error(error_msg: str) -> bool:
    """Detect Intel SVML/CPU instruction errors."""
    keywords = ['svml', '__svml', 'llvm', 'intel', 'mkl', 
                'illegal instruction', 'symbol not found']
    return any(kw in error_msg.lower() for kw in keywords)


def install_fishspeech_if_missing():
    """Auto-install FishSpeech if not available."""
    if importlib.util.find_spec("fish_speech") is None:
        logger.info("Installing FishSpeech...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/fishaudio/fish-speech.git#egg=fish-speech"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("FishSpeech installed")
        except Exception as e:
            logger.warning(f"Failed to install FishSpeech: {e}")


class TTSEngine:
    """Handles TTS synthesis with multiple backends."""
    
    def __init__(self, 
                 engine: str = "f5",
                 speaker_config: Optional[Dict[str, Any]] = None):
        self.engine = engine.lower()
        self.speaker_config = speaker_config or {}
        
        # Models
        self.f5_model = None
        self.f5_vocoder = None
        
        # FishSpeech paths
        self.checkpoint_dir = Path.cwd() / "checkpoints" / "fish-speech-1.5"
        self.vqgan_ckpt = self.checkpoint_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.t2s_ckpt = self.checkpoint_dir
        self.dac_ckpt = self.checkpoint_dir / "dac.pth"
        
        # Merge map for speaker handling
        self.merge_map = self._build_merge_map()
    
    def _build_merge_map(self) -> Dict[str, str]:
        """Build speaker merge mappings."""
        merge_map = {}
        for spk_id, info in self.speaker_config.items():
            merged_into = info.get('merged_into')
            if merged_into:
                merge_map[spk_id] = merged_into
            
            merged_speakers = info.get('merged_speakers', [])
            for merged in merged_speakers:
                merge_map[str(merged)] = spk_id
        
        return merge_map
    
    def _get_effective_speaker(self, speaker_id: int) -> Tuple[int, str]:
        """Get effective speaker ID and action."""
        spk_str = str(speaker_id)
        
        # Check if merged into another
        if spk_str in self.merge_map:
            effective_id = int(self.merge_map[spk_str])
        else:
            effective_id = speaker_id
        
        effective_str = str(effective_id)
        action = self.speaker_config.get(effective_str, {}).get('action', 'dub')
        
        return effective_id, action
    
    def _get_voice_sample(self, speaker_id: int) -> Optional[str]:
        """Get voice sample path for a speaker."""
        # Try effective speaker first
        effective_id, _ = self._get_effective_speaker(speaker_id)
        
        for sid in [effective_id, speaker_id]:
            spk_str = str(sid)
            if spk_str in self.speaker_config:
                sample_path = self.speaker_config[spk_str].get('sample_path')
                if sample_path:
                    # Convert URL path to filesystem path
                    if sample_path.startswith('/temp_chunks/'):
                        path = Path.cwd() / sample_path[1:]  # Remove leading /
                        if path.exists():
                            return str(path)
                    elif Path(sample_path).exists():
                        return sample_path
        
        return None
    
    def _load_f5(self):
        """Load F5-TTS model."""
        self.f5_model, self.f5_vocoder = manager.get_f5_tts()
    
    def _unload_f5(self):
        """Unload F5-TTS."""
        self.f5_model = None
        self.f5_vocoder = None
        manager.clear_cache(keep=['speaker_encoder'])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_fishspeech(self):
        """Verify FishSpeech is available."""
        install_fishspeech_if_missing()
        
        if not all(p.exists() for p in [self.vqgan_ckpt, self.dac_ckpt]):
            raise RuntimeError(f"FishSpeech checkpoints not found in {self.checkpoint_dir}")
    
    def _synthesize_f5(self, text: str, ref_path: str) -> np.ndarray:
        """Synthesize with F5-TTS."""
        from f5_tts.infer.utils_infer import infer_process
        
        # Load reference
        ref_audio, sr = sf.read(ref_path)
        if sr != 24000:
            import librosa
            ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=24000)
        
        if len(ref_audio.shape) > 1:
            ref_audio = ref_audio.mean(axis=1)
        
        ref_audio = ref_audio[:240_000]  # Max 10s
        
        # Save temp
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, ref_audio, 24000)
            ref_tmp = f.name
        
        try:
            # Apply SVML fixes
            os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"
            os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX2,AVX"
            torch.backends.mkldnn.enabled = False
            torch.set_num_threads(1)
            
            # Generate
            audio, sr_out, _ = infer_process(
                ref_audio=ref_tmp,
                ref_text="",
                gen_text=text,
                model_obj=self.f5_model,
                vocoder=self.f5_vocoder,
                mel_spec_type="vocos",
                device=manager.device
            )
            
            # Convert to numpy
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            audio = np.asarray(audio).flatten()
            
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            
            return audio
            
        finally:
            try:
                os.unlink(ref_tmp)
            except:
                pass
    
    def _synthesize_fishspeech(self, text: str, ref_path: str) -> np.ndarray:
        """Synthesize with FishSpeech (local 3-stage)."""
        import librosa
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_npy = tmpdir_path / "fake.npy"
            sem_npy = tmpdir_path / "sem.npy"
            output_wav = tmpdir_path / "output.wav"
            
            # Stage 1: VQGAN encode
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
            cmd2 = [
                sys.executable, "-m", "fish_speech.models.text2semantic.inference",
                "--text", text.strip(),
                "--prompt-text", "Reference speaker voice.",
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
            
            # Load result
            audio, sr = sf.read(output_wav)
            if sr != 24000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            
            return audio
    
    def _synthesize_lollms(self, text: str, speaker_id: int) -> np.ndarray:
        """Synthesize with LoLLMs TTS with voice cloning support."""
        from modules.tts.logic import generate_speech_lollms
        
        # Map speakers to voices
        voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        voice = voices[speaker_id % len(voices)]
        
        # Get voice sample for cloning
        sample_path = self._get_voice_sample(speaker_id)
        
        audio, sr = generate_speech_lollms(
            text=text,
            voice=voice,
            model="tts-1",
            response_format="wav",  # Use WAV for direct loading
            audio_sample_path=sample_path,  # Enable voice cloning if available
            max_retries=2
        )
        
        # Resample if needed
        if sr != 24000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        
        return audio
    
    def synthesize(self, 
                   segment: TranslationSegment,
                   output_dir: Path) -> bool:
        """
        Synthesize a single segment.
        
        On failure, generates silent audio instead of failing completely.
        Returns True if audio was generated (including silence), False only on critical errors.
        """
        try:
            # Check if speaker should be removed
            effective_id, action = self._get_effective_speaker(segment.speaker_id)
            
            # Calculate target duration
            target_duration = segment.end - segment.start
            
            if action == 'remove':
                # Generate silence
                audio = np.zeros(int(target_duration * 24000))
                is_silence = True
            else:
                # Get voice sample
                sample_path = self._get_voice_sample(segment.speaker_id)
                if not sample_path:
                    logger.warning(f"No voice sample for speaker {segment.speaker_id}, using silence")
                    audio = np.zeros(int(target_duration * 24000))
                    is_silence = True
                else:
                    # Synthesize based on engine
                    try:
                        if self.engine == 'f5':
                            audio = self._synthesize_f5(segment.translated_text, sample_path)
                        elif self.engine == 'fishspeech':
                            audio = self._synthesize_fishspeech(segment.translated_text, sample_path)
                        else:  # lollms
                            audio = self._synthesize_lollms(segment.translated_text, effective_id)
                        
                        is_silence = False
                        
                    except Exception as synth_error:
                        logger.error(f"TTS engine failed for segment {segment.idx}: {synth_error}")
                        # Fallback: generate silence with warning
                        audio = np.zeros(int(target_duration * 24000))
                        is_silence = True
                        segment.error = f"TTS failed, using silence: {str(synth_error)[:100]}"
                
                # Time-stretch to match duration if needed (only for non-silence)
                if not is_silence:
                    current_duration = len(audio) / 24000
                    
                    if abs(current_duration - target_duration) > 0.1:
                        try:
                            import librosa
                            rate = current_duration / target_duration
                            audio = librosa.effects.time_stretch(audio, rate=rate)
                        except Exception as e:
                            logger.warning(f"Time-stretch failed for segment {segment.idx}: {e}")
            
            # Save
            output_path = output_dir / f"segment_{segment.idx:04d}.wav"
            sf.write(output_path, audio, 24000)
            
            segment.audio_path = str(output_path)
            segment.status = "completed"
            
            if is_silence and not segment.error:
                segment.error = "Silent segment (speaker removed or TTS unavailable)"
            
            return True
            
        except Exception as e:
            # Critical failure - even silence generation failed
            logger.error(f"Critical TTS failure for segment {segment.idx}: {e}")
            segment.error = f"Critical failure: {str(e)[:200]}"
            segment.status = "failed"
            return False
    
    def synthesize_batch(self,
                        segments: List[TranslationSegment],
                        output_dir: Path,
                        progress_callback: Optional[Callable[[int, int], None]] = None
                        ) -> List[TranslationSegment]:
        """Synthesize a batch of segments."""
        # Load appropriate model
        if self.engine == 'f5':
            self._load_f5()
        elif self.engine == 'fishspeech':
            self._load_fishspeech()
        
        try:
            total = len(segments)
            for i, seg in enumerate(segments):
                if seg.status == "failed":
                    continue
                
                seg.status = "synthesizing"
                self.synthesize(seg, output_dir)
                
                if progress_callback:
                    progress_callback(i + 1, total)
            
            return segments
            
        finally:
            if self.engine == 'f5':
                self._unload_f5()


# =============================================================================
# MAIN PHASE 2 ENTRY POINT
# =============================================================================

async def run_phase2(
    task_id: str,
    speaker_config: Dict[str, Any],
    target_language: str = "en",
    source_language: str = "auto",
    tts_engine: str = "f5",
    batch_size: int = 10,
    progress_callback: Optional[Callable[[str, int, str], None]] = None,
    checkpoint_after_translation: bool = True
) -> TranslationResult:
    """
    Run complete Phase 2 pipeline.
    
    Args:
        task_id: The task ID
        speaker_config: Speaker configuration from validation
        target_language: Target language code
        source_language: Source language code
        tts_engine: TTS engine to use ('f5', 'fishspeech', 'lollms')
        batch_size: Number of segments to process in each TTS batch
        progress_callback: Called with (phase, percent, message)
        checkpoint_after_translation: Save checkpoint after translation (before TTS)
    
    Returns:
        TranslationResult with completion statistics
    
    Raises:
        RuntimeError: If critical failure occurs
    """
    
    async def report(phase: str, percent: int, message: str):
        if progress_callback:
            await progress_callback(phase, percent, message)
        logger.info(f"[Phase 2] {percent}%: {message}")
    
    try:
        # Load task data
        task = db.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Load segments
        seg_data = task.get('transcribed_segments', [])
        if not seg_data:
            raise ValueError("No transcribed segments found - run Phase 1 first")
        
        segments = [TranslationSegment.from_dict(s) for s in seg_data]
        master_audio = task.get('master_audio')
        
        await report("loading", 35, f"Loaded {len(segments)} segments")
        
        # Step 1: Translation
        await report("translating", 40, "Starting translation...")
        
        translator = TranslationEngine(target_language, source_language)
        
        def trans_progress(current, total):
            pct = 40 + int((current / total) * 20)
            asyncio.create_task(report("translating", pct, 
                f"Translated {current}/{total} segments"))
        
        segments = translator.translate_batch(segments, trans_progress)
        
        translated_count = sum(1 for s in segments if s.translated_text)
        await report("translating", 60, 
            f"Translation complete: {translated_count}/{len(segments)} segments")
        
        # CHECKPOINT: Save after translation
        db.update_task(
            task_id,
            segments=[s.to_dict() for s in segments],
            phase="translating",
            status="processing",
            progress=60,
            message="Translation complete - starting synthesis..."
        )
        
        if checkpoint_after_translation:
            logger.info(f"Checkpoint saved after translation for task {task_id}")
        
        # Step 2: TTS Synthesis (in batches)
        await report("synthesizing", 65, "Loading TTS model...")
        
        output_dir = Path("temp_chunks") / task_id / "synthesized"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tts = TTSEngine(tts_engine, speaker_config)
        
        # Process in batches
        total = len(segments)
        processed = 0
        completed = 0
        failed = 0
        
        for i in range(0, total, batch_size):
            batch = segments[i:i + batch_size]
            
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            await report("synthesizing", 
                65 + int((i / total) * 15),
                f"Synthesizing batch {batch_num}/{total_batches}")
            
            def synth_progress(current, total_in_batch):
                overall = i + current
                pct = 65 + int((overall / total) * 15)
                asyncio.create_task(report("synthesizing", pct,
                    f"Synthesized {overall}/{total} segments"))
            
            tts.synthesize_batch(batch, output_dir, synth_progress)
            
            # Update counts
            for seg in batch:
                if seg.status == "completed":
                    completed += 1
                elif seg.status == "failed":
                    failed += 1
            
            processed += len(batch)
            
            # INTERMEDIATE CHECKPOINT: Save progress every batch
            db.update_task(
                task_id,
                segments=[s.to_dict() for s in segments],
                progress=65 + int((processed / total) * 15)
            )
        
        await report("synthesizing", 80, 
            f"Synthesis complete: {completed} successful, {failed} failed")
        
        # Final checkpoint
        db.update_task(
            task_id,
            segments=[s.to_dict() for s in segments],
            translation_segments=[s.to_dict() for s in segments],
            phase="recomposing",
            status="queued",
            progress=80,
            message="Phase 2 complete - starting final assembly..."
        )
        
        return TranslationResult(
            segments=segments,
            translated_count=translated_count,
            synthesized_count=completed,
            failed_count=failed
        )
        
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Phase 2 failed:\n{tb_str}")
        
        db.update_task(
            task_id,
            status="failed",
            phase="translating",
            error_message=f"Phase 2 failed: {str(e)}",
            error_traceback=tb_str
        )
        
        raise RuntimeError(f"Phase 2 translation failed: {str(e)}") from e


# Export for workflow tasks
__all__ = [
    'run_phase2',
    'TranslationResult',
    'TranslationSegment',
    'TranslationEngine',
    'TTSEngine'
]
