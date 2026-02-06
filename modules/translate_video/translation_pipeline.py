import asyncio
import numpy as np
import soundfile as sf
import torch
import logging
import gc
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from core.resources import manager
from core.database import db

logger = logging.getLogger("translation_pipeline")


@dataclass
class TranslationSegment:
    idx: int
    start: float  # original timestamp
    end: float
    speaker_id: int
    original_text: str
    translated_text: Optional[str] = None
    synthesized_audio: Optional[np.ndarray] = None
    status: str = "pending"  # pending, translating, synthesizing, completed, failed


class ChunkedTranslationPipeline:
    """
    GPU-efficient translation pipeline that processes in small chunks
    to stay within 8GB VRAM budget.
    """
    
    def __init__(self, task_id: str, 
                 tts_engine: str = "f5",
                 chunk_size: int = 5,  # segments per GPU batch
                 target_language: str = "en"):
        self.task_id = task_id
        self.tts_engine = tts_engine
        self.chunk_size = chunk_size
        self.target_language = target_language
        
        self.lollms = None
        self.tts_model = None
        self.tts_vocoder = None
        
        self.progress_callback: Optional[Callable] = None
        
    async def initialize(self):
        """Load models with careful memory management."""
        # Load Lollms client (CPU-based, keep loaded)
        self.lollms = manager.get_lollms_client()
        
        # Pre-load TTS model if using F5
        if self.tts_engine == "f5" and F5_AVAILABLE:
            self.tts_model, self.tts_vocoder = manager.get_f5_tts()
        
        # Whisper is loaded on-demand and cleared after each batch
    
    async def translate_segments(self, 
                                  segments: List[Dict[str, Any]],
                                  speaker_voice_samples: Dict[int, str],
                                  progress_callback: Optional[Callable] = None) -> List[TranslationSegment]:
        """
        Translate all segments in GPU-friendly chunks.
        
        segments: List of {start, end, speaker_id, audio_path} from diarization
        speaker_voice_samples: Dict mapping speaker_id to reference audio path
        """
        self.progress_callback = progress_callback
        
        # Create translation objects
        trans_segments = [
            TranslationSegment(
                idx=i,
                start=seg['start'],
                end=seg['end'],
                speaker_id=seg['speaker_id'],
                original_text=""  # Will fill from STT
            )
            for i, seg in enumerate(segments)
        ]
        
        total = len(trans_segments)
        
        # Process in chunks
        for chunk_start in range(0, total, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total)
            chunk = trans_segments[chunk_start:chunk_end]
            
            await self._log(f"Processing segments {chunk_start}-{chunk_end-1}/{total}")
            
            # Phase 1: STT for chunk
            await self._run_stt_chunk(chunk)
            
            # Clear Whisper from GPU memory
            manager.clear_cache(keep=['speaker_encoder', 'f5_tts'] if self.tts_engine == 'f5' else ['speaker_encoder'])
            
            # Phase 2: Translation for chunk
            await self._run_translation_chunk(chunk)
            
            # Phase 3: TTS for chunk
            await self._run_tts_chunk(chunk, speaker_voice_samples)
            
            # Save progress to database after each chunk
            self._save_chunk_progress(chunk)
            
            # Report progress
            progress = int((chunk_end / total) * 100)
            if self.progress_callback:
                await self.progress_callback({
                    'phase': 'translating',
                    'progress': progress,
                    'current_segment': chunk_end,
                    'total_segments': total
                })
            
            # Aggressive cleanup between chunks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return trans_segments
    
    async def _run_stt_chunk(self, segments: List[TranslationSegment]):
        """Run Whisper STT on a chunk of segments."""
        stt = manager.get_whisper()
        
        for seg in segments:
            try:
                # Load audio segment
                audio, sr = sf.read(seg.audio_path if hasattr(seg, 'audio_path') else 
                                   self._get_segment_audio_path(seg))
                
                # Run STT
                result = stt(
                    {"array": audio, "sampling_rate": sr},
                    return_timestamps=False
                )
                
                seg.original_text = result.get('text', '').strip()
                seg.status = 'translated'  # Ready for translation
                
                await self._log(f"STT seg {seg.idx}: {seg.original_text[:50]}...")
                
            except Exception as e:
                logger.error(f"STT failed for segment {seg.idx}: {e}")
                seg.original_text = ""
                seg.status = 'failed'
    
    async def _run_translation_chunk(self, segments: List[TranslationSegment]):
        """Translate text using Lollms."""
        if not self.lollms:
            # Skip translation if Lollms unavailable
            for seg in segments:
                seg.translated_text = seg.original_text
            return
        
        for seg in segments:
            if seg.status == 'failed' or not seg.original_text:
                continue
            
            try:
                # Build translation prompt
                prompt = self._build_translation_prompt(seg.original_text)
                
                # Generate with timeout
                translated = await asyncio.wait_for(
                    asyncio.to_thread(self.lollms.generate_text, prompt, n_predict=512),
                    timeout=30.0
                )
                
                seg.translated_text = translated.strip()
                seg.status = 'synthesizing'
                
                await self._log(f"Translated seg {seg.idx}: {seg.translated_text[:50]}...")
                
            except Exception as e:
                logger.error(f"Translation failed for segment {seg.idx}: {e}")
                # Fallback: keep original
                seg.translated_text = seg.original_text
                seg.status = 'synthesizing'  # Still try to synthesize
    
    async def _run_tts_chunk(self, segments: List[TranslationSegment],
                            speaker_samples: Dict[int, str]):
        """Synthesize speech using F5-TTS or FishSpeech."""
        for seg in segments:
            if seg.status == 'failed' or not seg.translated_text:
                continue
            
            try:
                # Get reference audio for this speaker
                ref_path = speaker_samples.get(seg.speaker_id)
                if not ref_path:
                    logger.warning(f"No voice sample for speaker {seg.speaker_id}")
                    seg.status = 'failed'
                    continue
                
                # Generate audio
                if self.tts_engine == 'f5':
                    audio = await self._synthesize_f5(seg.translated_text, ref_path)
                else:
                    audio = await self._synthesize_fishspeech(seg.translated_text, ref_path)
                
                seg.synthesized_audio = audio
                seg.status = 'completed'
                
                # Save to disk
                output_path = self._save_segment_audio(seg, audio)
                
                # Update database
                db.save_translation_segment(
                    self.task_id,
                    seg.idx,
                    seg.original_text,
                    seg.translated_text,
                    output_path,
                    'completed'
                )
                
                await self._log(f"TTS completed for seg {seg.idx}")
                
            except Exception as e:
                logger.error(f"TTS failed for segment {seg.idx}: {e}")
                seg.status = 'failed'
    
    async def _synthesize_f5(self, text: str, ref_path: str) -> np.ndarray:
        """Synthesize with F5-TTS."""
        import tempfile
        from f5_tts.infer.utils_infer import infer_process
        
        # Load reference
        ref_audio, sr = sf.read(ref_path)
        if sr != 24000:
            ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=24000)
        
        # Ensure correct shape
        if len(ref_audio.shape) > 1:
            ref_audio = ref_audio.mean(axis=1)
        
        # Write temp file for F5
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, ref_audio, 24000)
            ref_tmp = f.name
        
        try:
            # Run inference
            audio, sr_out, _ = infer_process(
                ref_audio=ref_tmp,
                ref_text="",  # F5 doesn't need reference text
                gen_text=text,
                model_obj=self.tts_model,
                vocoder=self.tts_vocoder,
                mel_spec_type="vocos",
                device=manager.device,
            )
            
            # Convert to numpy
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            audio = np.asarray(audio).flatten()
            
            # Normalize
            audio = self._normalize_audio(audio)
            
            return audio
            
        finally:
            import os
            try:
                os.remove(ref_tmp)
            except:
                pass
    
    async def _synthesize_fishspeech(self, text: str, ref_path: str) -> np.ndarray:
        """Synthesize with FishSpeech API."""
        import requests
        import base64
        import io
        
        # Load and encode reference
        ref_audio, sr = sf.read(ref_path)
        
        # Get reference text via STT if not cached
        ref_text = self._get_cached_ref_text(ref_path)
        
        # Encode audio
        bio = io.BytesIO()
        sf.write(bio, ref_audio.astype(np.float32), sr, format='WAV')
        audio_b64 = base64.b64encode(bio.getvalue()).decode()
        
        # Call API
        payload = {
            "text": text,
            "reference_audio": audio_b64,
            "reference_text": ref_text,
        }
        
        response = requests.post(
            manager.fish_speech_url,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        # Decode response
        result = response.json()
        audio_bytes = base64.b64decode(result['audio'])
        
        # Load with soundfile
        bio = io.BytesIO(audio_bytes)
        audio, sr = sf.read(bio)
        
        return self._normalize_audio(audio)
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize to prevent clipping."""
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio / peak * 0.95
        return audio
    
    def _build_translation_prompt(self, text: str) -> str:
        """Build translation prompt for Lollms."""
        return f"""Translate the following text to {self.target_language}. 
Preserve the meaning, tone, and style. Only output the translation, no explanations.

Text: {text}

Translation:"""
    
    def _save_segment_audio(self, seg: TranslationSegment, audio: np.ndarray) -> str:
        """Save synthesized audio to disk."""
        output_dir = Path("outputs") / self.task_id / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        path = output_dir / f"seg_{seg.idx:04d}_{seg.speaker_id}.wav"
        
        # Match original duration approximately (speed up/slow down if needed)
        target_duration = seg.end - seg.start
        current_duration = len(audio) / 24000  # F5 outputs 24kHz
        
        if abs(current_duration - target_duration) > 0.5:
            # Use time-stretch to match duration
            rate = current_duration / target_duration
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        sf.write(path, audio, 24000)
        return str(path)
    
    def _get_segment_audio_path(self, seg: TranslationSegment) -> str:
        """Reconstruct path to original segment audio."""
        # This should be stored during diarization phase
        task = db.get_task(self.task_id)
        segments = task.get('segments', [])
        if seg.idx < len(segments):
            return segments[seg.idx].get('audio_path', '')
        return ''
    
    def _get_cached_ref_text(self, ref_path: str) -> str:
        """Get or compute reference text for voice cloning."""
        cache_path = Path(ref_path).with_suffix('.txt')
        if cache_path.exists():
            return cache_path.read_text()
        
        # Compute with STT
        stt = manager.get_whisper()
        result = stt(ref_path)
        text = result.get('text', '')
        
        # Cache
        cache_path.write_text(text)
        return text
    
    def _save_chunk_progress(self, segments: List[TranslationSegment]):
        """Save checkpoint after chunk completion."""
        checkpoint = {
            'last_completed_idx': max(s.idx for s in segments),
            'timestamp': datetime.now().isoformat(),
            'segments': [
                {
                    'idx': s.idx,
                    'status': s.status,
                    'original': s.original_text[:100] if s.original_text else '',
                    'translated': s.translated_text[:100] if s.translated_text else ''
                }
                for s in segments
            ]
        }
        db.update_task(self.task_id, {'checkpoint_data': checkpoint})
    
    async def _log(self, msg: str, style: str = "info"):
        """Log to database and console."""
        logger.info(msg)
        db.append_log(self.task_id, msg, style)


class VideoRecomposer:
    """Recompose final video with translated audio mixed with background."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        
    async def recompose(self, 
                       original_video: str,
                       background_audio: Optional[str],
                       translated_segments: List[TranslationSegment],
                       output_path: str,
                       progress_callback: Optional[Callable] = None):
        """
        Build final video:
        1. Assemble translated speech in timeline
        2. Mix with background audio
        3. Merge with original video
        """
        # Step 1: Build speech track
        speech_track = await self._build_speech_track(translated_segments)
        
        # Step 2: Mix with background
        final_audio = await self._mix_with_background(
            speech_track, 
            background_audio,
            self._get_video_duration(original_video)
        )
        
        # Step 3: Merge with video
        await self._merge_audio_video(original_video, final_audio, output_path, progress_callback)
        
        return output_path
    
    async def _build_speech_track(self, segments: List[TranslationSegment]) -> np.ndarray:
        """Assemble all speech segments into continuous audio."""
        # Find total duration
        max_end = max(s.end for s in segments if s.end)
        sr = 24000  # F5 output rate
        
        # Create silent track
        total_samples = int(max_end * sr)
        track = np.zeros(total_samples, dtype=np.float32)
        
        for seg in segments:
            if seg.status != 'completed' or seg.synthesized_audio is None:
                continue
            
            # Load saved audio if not in memory
            audio = seg.synthesized_audio
            if audio is None:
                path = db.get_translation_segments(self.task_id)[seg.idx].get('audio_path')
                if path:
                    audio, _ = sf.read(path)
            
            if audio is None:
                continue
            
            # Resample if needed
            if len(audio) != int((seg.end - seg.start) * sr):
                # Stretch to fit exactly
                target_len = int((seg.end - seg.start) * sr)
                audio = librosa.effects.time_stretch(audio, len(audio) / target_len)
            
            # Place in track
            start_sample = int(seg.start * sr)
            end_sample = start_sample + len(audio)
            
            if end_sample <= len(track):
                # Crossfade for smooth transitions
                fade_samples = min(1000, len(audio) // 4)
                if start_sample > 0:
                    # Fade in
                    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                if end_sample < len(track):
                    # Fade out
                    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                track[start_sample:end_sample] = audio
        
        return track
    
    async def _mix_with_background(self,
                                    speech: np.ndarray,
                                    background_path: Optional[str],
                                    duration: float) -> str:
        """Mix speech with background, save to file."""
        sr = 24000
        
        # Load and resample background
        if background_path and Path(background_path).exists():
            bg, bg_sr = sf.read(background_path)
            if bg_sr != sr:
                bg = librosa.resample(bg, orig_sr=bg_sr, target_sr=sr)
        else:
            # Generate silence
            bg = np.zeros(len(speech))
        
        # Match lengths
        target_len = len(speech)
        if len(bg) < target_len:
            # Loop background
            repeats = int(np.ceil(target_len / len(bg)))
            bg = np.tile(bg, repeats)[:target_len]
        else:
            bg = bg[:target_len]
        
        # Mix with ducking (background lowers when speech present)
        # Simple approach: speech -6dB, background -20dB when speech active
        speech_gain = 0.5  # -6dB
        bg_gain = 0.1      # -20dB
        
        # Detect speech presence for dynamic ducking
        speech_rms = np.sqrt(np.convolve(speech**2, np.ones(1000)/1000, mode='same'))
        is_speech = speech_rms > 0.01
        
        # Apply ducking envelope
        bg_envelope = np.where(is_speech, bg_gain, bg_gain * 2)  # 2x louder when no speech
        
        mixed = speech * speech_gain + bg * bg_envelope
        
        # Final normalize
        peak = np.max(np.abs(mixed))
        if peak > 0.99:
            mixed = mixed / peak * 0.99
        
        # Save
        output_path = Path("outputs") / self.task_id / "final_audio.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, mixed, sr)
        
        return str(output_path)
    
    async def _merge_audio_video(self,
                                  video_path: str,
                                  audio_path: str,
                                  output_path: str,
                                  progress_callback: Optional[Callable] = None):
        """Use FFmpeg to merge audio with original video."""
        import subprocess
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,      # Video input
            "-i", audio_path,      # Audio input
            "-c:v", "copy",        # Copy video codec
            "-c:a", "aac",         # AAC audio
            "-b:a", "192k",        # Audio bitrate
            "-map", "0:v:0",       # Take video from first input
            "-map", "1:a:0",       # Take audio from second input
            "-shortest",           # Match shortest duration
            output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
        
        return output_path
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using FFprobe."""
        import subprocess
        
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
