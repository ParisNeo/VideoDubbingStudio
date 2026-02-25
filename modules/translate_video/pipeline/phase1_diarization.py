"""
Phase 1: Audio Extraction & Speaker Identification

Complete pipeline: Audio Extraction → VAD → Embedding Extraction → 
Clustering → Sample Extraction → Transcription

This module contains everything needed for Phase 1 in a single,
focused file with clear checkpoint boundaries.

Checkpoints:
- After speaker identification (before transcription) - allows re-clustering
- After transcription complete - ready for validation
"""

import asyncio
import json
import soundfile as sf
import subprocess
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import concurrent.futures
import numpy as np
import torch
import torch.nn.functional as F
import gc
import tempfile

from core.resources import manager
from core.database import db

logger = logging.getLogger("phase1_diarization")

# Thread pool for CPU-bound diarization
_diarization_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# -------------------------------------------------------------------------------------------------------------------------------
# DATA MODELS
# -------------------------------------------------------------------------------------------------------------------------------

class SpeechSegment:
    """Represents a detected speech segment."""
    def __init__(
        self,
        idx: int,
        start: float,
        end: float,
        speaker_id: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        original_text: str = "",
        translated_text: str = ""
    ):
        self.idx = idx
        self.start = start
        self.end = end
        self.speaker_id = speaker_id
        self.embedding = embedding
        self.original_text = original_text
        self.translated_text = translated_text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'idx': self.idx,
            'start': self.start,
            'end': self.end,
            'speaker_id': self.speaker_id,
            'original_text': self.original_text,
            'translated_text': self.translated_text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeechSegment":
        return cls(
            idx=data['idx'],
            start=data['start'],
            end=data['end'],
            speaker_id=data.get('speaker_id'),
            original_text=data.get('original_text', ''),
            translated_text=data.get('translated_text', '')
        )


class DiarizationResult:
    """Complete result from Phase 1."""
    def __init__(
        self,
        segments: List[SpeechSegment],
        speaker_count: int,
        speaker_samples: Dict[int, str],
        assignments: Dict[int, int],
        master_audio: str,
        speaker_config: Dict[str, Any]
    ):
        self.segments = segments
        self.speaker_count = speaker_count
        self.speaker_samples = speaker_samples
        self.assignments = assignments
        self.master_audio = master_audio
        self.speaker_config = speaker_config
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segments': [s.to_dict() for s in self.segments],
            'speaker_count': self.speaker_count,
            'speaker_samples': self.speaker_samples,
            'assignments': self.assignments,
            'master_audio': self.master_audio,
            'speaker_config': self.speaker_config
        }


# -------------------------------------------------------------------------------------------------------------------------------
# AUDIO PROCESSING
# -------------------------------------------------------------------------------------------------------------------------------

def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """Extract audio from video using FFmpeg."""
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    return output_path


# -------------------------------------------------------------------------------------------------------------------------------
# VAD (VOICE ACTIVITY DETECTION)
# -------------------------------------------------------------------------------------------------------------------------------
def run_vad(audio_path: str, 
            threshold: float = 0.15, # Increased sensitivity (Lower = more inclusive)
            min_speech_duration_ms: int = 100) -> List[Dict[str, float]]: 
    """
    Run Silero VAD with silence-padding and greedy merging to prevent audio stripping.
    """
    import torch
    import numpy as np
    
    # Load model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps, _, read_audio, _, _ = utils
    
    # 1. Load audio and PREPEND 1 second of silence to capture immediate speech at 0.0s
    wav = read_audio(audio_path)
    padding_len = 16000 # 1 second at 16kHz
    padding = torch.zeros(padding_len)
    padded_wav = torch.cat([padding, wav])
    
    # 2. Get timestamps from padded audio
    # Increased min_silence_duration_ms to 1000ms to merge words into sentences
    speech_timestamps = get_speech_timestamps(
        padded_wav, 
        model, 
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=1000 
    )
    
    # 3. Adjust timestamps back and apply greedy safety buffers
    refined_segments = []
    for ts in speech_timestamps:
        # Subtract the 1.0s padding
        start = (ts['start'] / 16000) - 1.0
        end = (ts['end'] / 16000) - 1.0
        
        # Greedy buffers: 300ms room to prevent word clipping
        start = max(0, start - 0.3) 
        end = end + 0.3             
        
        duration = end - start
        if duration < 0.1: continue # Only ignore absolute micro-blips
        
        # Split long segments for Whisper reliability (max 30s)
        if duration > 30:
            curr = start
            while curr < end:
                chunk_end = min(curr + 25, end) # Use 25s for safe overlap
                if (end - chunk_end) < 5: chunk_end = end
                refined_segments.append({'start': curr, 'end': chunk_end})
                curr = chunk_end
        else:
            refined_segments.append({'start': start, 'end': end})

    # Fallback: If nothing detected, return the whole file as one segment to prevent total loss
    if not refined_segments and len(wav) > 0:
        logger.warning("VAD detected no speech. Falling back to full-file segment.")
        refined_segments.append({'start': 0.0, 'end': len(wav) / 16000})

    return refined_segments


# -------------------------------------------------------------------------------------------------------------------------------
# SPEAKER EMBEDDING & CLUSTERING
# -------------------------------------------------------------------------------------------------------------------------------

class SpeakerIdentifier:
    """
    Identifies speakers using WavLM embeddings and clustering.
    """
    
    def __init__(self, 
                 min_speakers: int = 1, 
                 max_speakers: int = 10,
                 similarity_threshold: float = 0.85):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        
        self.feature_extractor = None
        self.embedding_model = None
    
    def _load_models(self):
        """Load WavLM models."""
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        
        model_id = "microsoft/wavlm-base-plus-sv"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.embedding_model = WavLMForXVector.from_pretrained(model_id)
        
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.cuda()
        
        self.embedding_model.eval()
    
    def _unload_models(self):
        """Free GPU memory."""
        self.feature_extractor = None
        self.embedding_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_embeddings(self, 
                         audio_path: str,
                         segments: List[Dict[str, float]],
                         progress_callback: Optional[Callable[[int, int], None]] = None
                         ) -> Tuple[List[SpeechSegment], List[np.ndarray]]:
        """
        Extract embeddings for each speech segment.
        
        Returns: (list of SpeechSegment objects, list of embeddings)
        """
        self._load_models()
        
        try:
            # Load audio
            audio_np, sr = sf.read(audio_path)
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)
            if sr != 16000:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
            
            segments_list = []
            embeddings = []
            
            for i, seg in enumerate(segments):
                # Extract segment audio
                start_sample = int(seg['start'] * 16000)
                end_sample = int(seg['end'] * 16000)
                chunk = audio_np[start_sample:end_sample]
                
                # Skip very short segments
                if len(chunk) < 1600:  # 0.1s
                    continue
                
                # Extract embedding
                try:
                    inputs = self.feature_extractor(
                        chunk, 
                        sampling_rate=16000, 
                        return_tensors="pt",
                        padding=True
                    )
                    
                    input_values = inputs.input_values
                    if torch.cuda.is_available():
                        input_values = input_values.cuda()
                    
                    with torch.no_grad():
                        outputs = self.embedding_model(input_values)
                        emb = outputs.embeddings
                        emb = F.normalize(emb, p=2, dim=1)
                    
                    embeddings.append(emb.cpu().numpy()[0])
                    segments_list.append(SpeechSegment(
                        idx=i,
                        start=seg['start'],
                        end=seg['end']
                    ))
                    
                    if progress_callback and i % 10 == 0:
                        progress_callback(i, len(segments))
                        
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for segment {i}: {e}")
            
            return segments_list, embeddings
            
        finally:
            self._unload_models()
    
    def cluster_speakers(self, 
                        embeddings: List[np.ndarray],
                        progress_callback: Optional[Callable[[str], None]] = None
                        ) -> Dict[int, int]:
        """
        Cluster embeddings to identify speakers. Favoring single-speaker consistency.
        """
        if len(embeddings) == 0:
            return {}
        
        if len(embeddings) < 3: # Too few samples for meaningful clustering
            return {i: 0 for i in range(len(embeddings))}
        
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        X = np.stack(embeddings)
        n_samples = len(X)
        
        # Determine optimal number of speakers
        max_k = min(n_samples - 1, self.max_speakers)
        
        best_n = 1
        best_score = -1
        labels = np.zeros(n_samples, dtype=int)
        
        # Evaluation loop
        for k in range(2, max_k + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=k, 
                    metric='cosine', 
                    linkage='average' # Average linkage is more stable for speech
                )
                lbls = clustering.fit_predict(X)
                
                score = silhouette_score(X, lbls, metric='cosine')
                
                # Bias: To avoid splitting a single speaker into many, 
                # we require a significant improvement in silhouette score to increase K
                if score > (best_score + 0.15): 
                    best_score = score
                    best_n = k
                    labels = lbls
            except:
                continue
        
        # Final safety: If best score is very low, it's likely just one speaker
        if best_score < 0.4:
            best_n = 1
            labels = np.zeros(n_samples, dtype=int)
        
        if progress_callback:
            progress_callback(f"Identified {best_n} distinct speaker identities")
        
        return {i: int(label) for i, label in enumerate(labels)}
    
    def run_with_vad(self, 
                    audio_path: str,
                    speech_segments: List[Dict[str, float]],
                    progress_callback: Optional[Callable[[str, int, int], None]] = None
                    ) -> Tuple[List[SpeechSegment], Dict[int, int], int]:
        """
        Run identification using pre-calculated VAD segments (Respects sensitivity settings).
        """
        if progress_callback:
            progress_callback(f"Analyzing {len(speech_segments)} detected speech segments...", 20, 100)
        
        # Step 1: Extract embeddings
        def emb_progress(current, total):
            if progress_callback:
                pct = 20 + int((current / total) * 40) # Mapping 20% -> 60%
                progress_callback(f"Extracting speaker features: {current}/{total}", pct, 100)
        
        segments, embeddings = self.extract_embeddings(audio_path, speech_segments, emb_progress)
        
        # Step 2: Cluster
        if progress_callback:
            progress_callback("Grouping speakers...", 70, 100)
        
        def cluster_progress(msg):
            if progress_callback:
                progress_callback(msg, 85, 100)
        
        assignments = self.cluster_speakers(embeddings, cluster_progress)
        
        # Step 3: Assign IDs
        for i, seg in enumerate(segments):
            seg.speaker_id = assignments.get(i, 0)
        
        num_speakers = len(set(assignments.values())) if assignments else 0
        
        if progress_callback:
            progress_callback(f"Diarization complete: {num_speakers} speakers", 100, 100)
        
        return segments, assignments, num_speakers

    def run(self, 
            audio_path: str,
            progress_callback: Optional[Callable[[str, int, int], None]] = None
            ) -> Tuple[List[SpeechSegment], Dict[int, int], int]:
        """
        Legacy entry point: Runs VAD internally with default settings.
        """
        if progress_callback:
            progress_callback("Detecting speech...", 0, 100)
        
        speech_segments = run_vad(audio_path)
        return self.run_with_vad(audio_path, speech_segments, progress_callback)


# -------------------------------------------------------------------------------------------------------------------------------
# SAMPLE EXTRACTION & CONFIG GENERATION
# -------------------------------------------------------------------------------------------------------------------------------

def extract_speaker_samples(
    audio_path: str,
    segments: List[SpeechSegment],
    task_id: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Extract reference audio samples for each speaker.
    
    Returns: (speaker_id -> sample_path mapping, speaker_config for UI)
    """
    # Group segments by speaker
    speakers: Dict[int, List[SpeechSegment]] = {}
    for seg in segments:
        sid = seg.speaker_id
        if sid not in speakers:
            speakers[sid] = []
        speakers[sid].append(seg)
    
    # Load master audio
    audio_np, sr = sf.read(audio_path)
    
    # Create samples directory
    samples_dir = Path("temp_chunks") / task_id / "speaker_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    speaker_samples = {}
    speaker_config = {}
    
    for sid, segs in speakers.items():
        # Calculate Statistics
        total_duration = sum(s.end - s.start for s in segs)
        intervention_count = len(segs)

        # Sort segments by duration (longest first)
        sorted_segs = sorted(segs, key=lambda s: s.end - s.start, reverse=True)
        
        selected_audio = []
        accumulated_time = 0
        
        # Combine the longest segments to make a solid reference (max 15s)
        for s in sorted_segs:
            if accumulated_time >= 15.0:
                break
            
            start_sample = max(0, int(s.start * sr))
            end_sample = min(len(audio_np), int(s.end * sr))
            chunk = audio_np[start_sample:end_sample]
            
            selected_audio.append(chunk)
            accumulated_time += (s.end - s.start)
            
            # Add 0.5s silence between jumps to prevent jarring cuts
            silence = np.zeros(int(0.5 * sr), dtype=audio_np.dtype)
            selected_audio.append(silence)
            
        if selected_audio:
            sample_audio = np.concatenate(selected_audio)
        else:
            sample_audio = np.zeros(sr, dtype=audio_np.dtype)
        
        # Save sample
        sample_path = samples_dir / f"speaker_{sid}_sample.wav"
        sf.write(sample_path, sample_audio, sr)
        
        speaker_samples[sid] = str(sample_path)
        
        # Create config entry
        speaker_config[str(sid)] = {
            "name": f"Speaker {sid + 1}",
            "action": "dub",
            "sample_path": f"/temp_chunks/{task_id}/speaker_samples/speaker_{sid}_sample.wav",
            "total_duration": round(total_duration, 2),
            "intervention_count": intervention_count
        }
        
        if progress_callback:
            progress_callback(f"Extracted sample for speaker {sid + 1} ({round(total_duration, 1)}s total)")
    
    return speaker_samples, speaker_config


# -------------------------------------------------------------------------------------------------------------------------------
# TRANSCRIPTION
# -------------------------------------------------------------------------------------------------------------------------------

def transcribe_segments(
    audio_path: str,
    segments: List[SpeechSegment],
    source_language: str = "auto",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    whisper_model: Optional[str] = None
) -> List[SpeechSegment]:
    """
    Transcribe all segments using mutualized robust Whisper logic.
    """
    from modules.transcribe.logic import whisper_inference
    import librosa
    
    # Load master audio
    audio_np, sr = sf.read(audio_path)
    if len(audio_np.shape) > 1: audio_np = audio_np.mean(axis=1)
    
    total = len(segments)
    
    for i, seg in enumerate(segments):
        try:
            # Extract segment audio
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            chunk = audio_np[start_sample:end_sample]
            
            # Resample to 16kHz
            if sr != 16000:
                chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
            
            # Use mutualized inference (handles VRAM cleanup and language)
            seg.original_text = whisper_inference(
                chunk.astype(np.float32), 
                src_lang=source_language, 
                whisper_model=whisper_model
            )
            
            if progress_callback and (i % 5 == 0 or i == total - 1):
                progress_callback(i + 1, total)
                
        except Exception as e:
            logger.error(f"Failed to transcribe segment {seg.idx}: {e}")
            seg.original_text = f"[Transcription error: {str(e)[:50]}]"
    
    return segments


# -------------------------------------------------------------------------------------------------------------------------------
# MAIN PHASE 1 ENTRY POINT
# -------------------------------------------------------------------------------------------------------------------------------

async def run_diarization_phase(
    task_id: str,
    video_path: str,
    progress_callback: Optional[Callable] = None
):
    task_data = db.get_task(task_id)
    # Default to 0.15 for better inclusivity if not set
    user_threshold = float(task_data.get('vad_threshold', 0.15))
    """Stage 1: Extract audio and identify speakers."""
    import functools
    loop = asyncio.get_running_loop()

    async def report(phase: str, pct: int, msg: str):
        if progress_callback: await progress_callback(phase, pct, msg)
        logger.info(f"[Diarization] {pct}%: {msg}")

    try:
        # 1. Extraction
        await report("identifying", 5, "Extracting audio...")
        master_wav = str(Path("temp_chunks") / task_id / "master.wav")
        extract_audio(video_path, master_wav)

        # 2. Identification
        await report("identifying", 15, "Running diarization...")
        identifier = SpeakerIdentifier(max_speakers=10)
        
        # 1. Run VAD with user threshold
        await report("identifying", 10, f"Detecting speech (Sensitivity: {user_threshold})...")
        speech_segments = run_vad(master_wav, threshold=user_threshold)

        # 2. Run Identification
        await report("identifying", 20, "Extracting speaker features...")
        identifier = SpeakerIdentifier(max_speakers=10)

        def diar_progress_sync(msg, pct, total):
            current_pct = 20 + int(pct * 0.4)
            asyncio.run_coroutine_threadsafe(report("identifying", current_pct, msg), loop)

        # Modify SpeakerIdentifier to accept pre-defined segments
        segments, assignments, num_speakers = await loop.run_in_executor(
            None, functools.partial(identifier.run_with_vad, master_wav, speech_segments, diar_progress_sync)
        )

        # 3. Samples
        await report("identifying", 70, "Extracting speaker samples...")
        speaker_samples, speaker_config = extract_speaker_samples(master_wav, segments, task_id)

        # Update DB and stop for validation - CRITICAL: Do not report "complete" here
        # to prevent overwriting the 'awaiting_speaker_validation' phase.
        db.update_task(
            task_id,
            segments=[s.to_dict() for s in segments],
            speaker_config=speaker_config,
            speaker_samples=speaker_samples,
            master_audio=master_wav,
            phase="awaiting_speaker_validation",
            status="awaiting_validation",
            progress=35,
            message="Please confirm detected speakers"
        )
        
        from ..state import broadcast_to_task
        await broadcast_to_task(task_id, {
            'type': 'speaker_validation_ready',
            'data': {
                'speaker_config': speaker_config,
                'phase': 'awaiting_speaker_validation',
                'status': 'awaiting_validation'
            }
        })
        
        logger.info(f"Diarization finished for {task_id}. Phase set to awaiting_speaker_validation")
    except Exception as e:
        logger.error(f"Diarization phase failed: {traceback.format_exc()}")
        db.update_task(task_id, status="failed", error_message=str(e))

async def run_transcription_phase(
    task_id: str,
    source_language: str = "auto",
    progress_callback: Optional[Callable] = None
):
    """Stage 2: Transcribe segments after speaker confirmation."""
    import functools
    loop = asyncio.get_running_loop()

    async def report(phase: str, pct: int, msg: str):
        if progress_callback: await progress_callback(phase, pct, msg)
        logger.info(f"[Transcription] {pct}%: {msg}")

    try:
        task = db.get_task(task_id)
        raw_segments = task.get('segments', [])
        speaker_config = task.get('speaker_config', {})
        master_audio = task.get('master_audio')

        # FILTER: Only keep segments for speakers marked as 'dub'
        filtered_segments = []
        for s_data in raw_segments:
            sid = str(s_data.get('speaker_id'))
            action = speaker_config.get(sid, {}).get('action', 'dub')
            
            if action != 'remove':
                filtered_segments.append(SpeechSegment.from_dict(s_data))
        
        segments = filtered_segments
        await report("transcribing", 35, f"Starting transcription for {len(segments)} valid segments...")
        
        def trans_progress_sync(cur, tot):
            # Map 35% -> 60% to bridge the gap to Translation phase
            current_pct = 35 + int((cur / tot) * 25)
            asyncio.run_coroutine_threadsafe(report("transcribing", current_pct, f"Transcribed {cur}/{tot}"), loop)

        # Get whisper model preference from task
        whisper_model = task.get('whisper_model')
        
        # Run transcription in executor
        segments = await loop.run_in_executor(
            None, functools.partial(transcribe_segments, master_audio, segments, source_language, trans_progress_sync, whisper_model)
        )

        # Stop for Transcription Review
        db.update_task(
            task_id,
            segments=[s.to_dict() for s in segments],
            transcribed_segments=[s.to_dict() for s in segments],
            phase="awaiting_transcription_review",
            status="awaiting_validation",
            progress=60,
            message="Please review the transcription text"
        )
        
        from ..state import broadcast_to_task
        await broadcast_to_task(task_id, {
            'type': 'transcription_ready',
            'data': {
                'segments': [s.to_dict() for s in segments],
                'phase': 'awaiting_transcription_review',
                'status': 'awaiting_validation'
            }
        })
    except Exception as e:
        logger.error(f"Transcription phase failed: {traceback.format_exc()}")
        db.update_task(task_id, status="failed", error_message=str(e))

# Export for workflow tasks
__all__ = [
    'run_diarization_phase',
    'run_transcription_phase',
    'DiarizationResult',
    'SpeechSegment',
    'extract_audio',
    'run_vad',
    'SpeakerIdentifier',
    'extract_speaker_samples',
    'transcribe_segments'
]
