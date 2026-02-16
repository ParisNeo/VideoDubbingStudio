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


# =============================================================================
# DATA MODELS
# =============================================================================

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


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

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


# =============================================================================
# VAD (VOICE ACTIVITY DETECTION)
# =============================================================================

def run_vad(audio_path: str, 
            threshold: float = 0.5,
            min_speech_duration_ms: int = 250) -> List[Dict[str, float]]:
    """
    Run Silero VAD on audio file.
    
    Returns list of {start, end} dicts in seconds.
    """
    import torch
    
    # Load model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps, _, read_audio, _, _ = utils
    
    # Load audio
    wav = read_audio(audio_path)
    
    # Get timestamps
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=500
    )
    
    # Convert to seconds
    return [
        {'start': ts['start'] / 16000, 'end': ts['end'] / 16000}
        for ts in speech_timestamps
    ]


# =============================================================================
# SPEAKER EMBEDDING & CLUSTERING
# =============================================================================

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
        Cluster embeddings to identify speakers.
        
        Returns: mapping from segment index to speaker ID
        """
        if len(embeddings) == 0:
            return {}
        
        if len(embeddings) == 1:
            return {0: 0}
        
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        X = np.stack(embeddings)
        n_samples = len(X)
        
        # Determine optimal number of speakers
        max_k = min(n_samples - 1, self.max_speakers) if n_samples > 2 else 2
        min_k = 1 if n_samples == 1 else 2
        
        best_n = 1 if n_samples == 1 else 2
        best_score = -1
        labels = [0] * n_samples
        
        if n_samples >= 2:
            for k in range(min_k, max_k + 1):
                try:
                    if k == 1:
                        clustering = AgglomerativeClustering(n_clusters=1)
                        lbls = clustering.fit_predict(X)
                        score = -0.5
                    else:
                        clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
                        lbls = clustering.fit_predict(X)
                        
                        n_labels = len(set(lbls))
                        if 1 < n_labels < n_samples:
                            score = silhouette_score(X, lbls, metric='cosine')
                        elif n_labels == 1:
                            score = -0.5
                        else:
                            score = -1
                    
                    if score > best_score:
                        best_score = score
                        best_n = k
                        labels = lbls
                        
                except Exception as e:
                    logger.debug(f"Clustering with {k} speakers failed: {e}")
        
        if progress_callback:
            progress_callback(f"Identified {best_n} speakers")
        
        return {i: int(label) for i, label in enumerate(labels)}
    
    def run(self, 
            audio_path: str,
            progress_callback: Optional[Callable[[str, int, int], None]] = None
            ) -> Tuple[List[SpeechSegment], Dict[int, int], int]:
        """
        Run complete speaker identification.
        
        Returns: (segments with speaker IDs, assignments mapping, speaker count)
        """
        # Step 1: VAD
        if progress_callback:
            progress_callback("Running voice activity detection...", 0, 100)
        
        speech_segments = run_vad(audio_path)
        
        if progress_callback:
            progress_callback(f"Found {len(speech_segments)} speech segments", 20, 100)
        
        # Step 2: Extract embeddings
        if progress_callback:
            progress_callback("Extracting speaker embeddings...", 25, 100)
        
        def emb_progress(current, total):
            if progress_callback:
                pct = 25 + int((current / total) * 35)
                progress_callback(f"Extracted {current}/{total} embeddings...", pct, 100)
        
        segments, embeddings = self.extract_embeddings(audio_path, speech_segments, emb_progress)
        
        if progress_callback:
            progress_callback(f"Embeddings extracted: {len(embeddings)}", 60, 100)
        
        # Step 3: Cluster
        if progress_callback:
            progress_callback("Clustering speakers...", 65, 100)
        
        def cluster_progress(msg):
            if progress_callback:
                progress_callback(msg, 80, 100)
        
        assignments = self.cluster_speakers(embeddings, cluster_progress)
        
        # Update segments with speaker IDs
        for i, seg in enumerate(segments):
            seg.speaker_id = assignments.get(i, 0)
        
        num_speakers = len(set(assignments.values())) if assignments else 0
        
        if progress_callback:
            progress_callback(f"Diarization complete: {num_speakers} speakers", 100, 100)
        
        return segments, assignments, num_speakers


# =============================================================================
# SAMPLE EXTRACTION & CONFIG GENERATION
# =============================================================================

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
        # Find longest segment for this speaker
        best_seg = max(segs, key=lambda s: s.end - s.start)
        
        # Extract audio
        start_sample = int(best_seg.start * sr)
        end_sample = int(best_seg.end * sr)
        sample_audio = audio_np[start_sample:end_sample]
        
        # Save sample
        sample_path = samples_dir / f"speaker_{sid}_sample.wav"
        sf.write(sample_path, sample_audio, sr)
        
        speaker_samples[sid] = str(sample_path)
        
        # Create config entry
        speaker_config[str(sid)] = {
            "name": f"Speaker {sid + 1}",
            "action": "dub",
            "sample_path": f"/temp_chunks/{task_id}/speaker_samples/speaker_{sid}_sample.wav"
        }
        
        if progress_callback:
            progress_callback(f"Extracted sample for speaker {sid + 1}")
    
    return speaker_samples, speaker_config


# =============================================================================
# TRANSCRIPTION
# =============================================================================

def transcribe_segments(
    audio_path: str,
    segments: List[SpeechSegment],
    source_language: str = "auto",
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[SpeechSegment]:
    """
    Transcribe all segments using Whisper.
    
    Modifies segments in-place with original_text.
    """
    from transformers import AutoFeatureExtractor
    
    # Load Whisper
    whisper_pipe = manager.get_whisper()
    model = whisper_pipe.model
    tokenizer = whisper_pipe.tokenizer
    
    # Get feature extractor
    if hasattr(whisper_pipe, 'feature_extractor'):
        feature_extractor = whisper_pipe.feature_extractor
    else:
        from transformers import AutoFeatureExtractor
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v2")
    
    device = manager.device
    model_dtype = next(model.parameters()).dtype
    
    # Load master audio
    audio_np, sr = sf.read(audio_path)
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)
    
    total = len(segments)
    
    for i, seg in enumerate(segments):
        try:
            # Extract segment audio
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            chunk = audio_np[start_sample:end_sample]
            
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
            
            chunk = chunk.astype(np.float32)
            
            # Process with feature extractor
            inputs = feature_extractor(chunk, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device).to(model_dtype)
            
            # Generate kwargs
            generate_kwargs = {}
            if source_language != 'auto':
                generate_kwargs["language"] = source_language
            
            # Transcribe
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    max_length=448,
                    num_beams=1,
                    condition_on_prev_tokens=False,
                    **generate_kwargs
                )
            
            transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            seg.original_text = transcription.strip()
            
            if progress_callback and (i % 5 == 0 or i == total - 1):
                progress_callback(i + 1, total)
                
        except Exception as e:
            logger.error(f"Failed to transcribe segment {seg.idx}: {e}")
            seg.original_text = f"[Transcription error: {str(e)[:50]}]"
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return segments


# =============================================================================
# MAIN PHASE 1 ENTRY POINT
# =============================================================================

async def run_phase1(
    task_id: str,
    video_path: str,
    source_language: str = "auto",
    progress_callback: Optional[Callable[[str, int, str], None]] = None,
    checkpoint_after_diarization: bool = True
) -> DiarizationResult:
    """
    Run complete Phase 1 pipeline.
    
    Args:
        task_id: The task ID
        video_path: Path to input video
        source_language: Source language code
        progress_callback: Called with (phase, percent, message)
        checkpoint_after_diarization: If True, save checkpoint after speaker ID 
                                      (allows re-running with different clustering params)
    
    Returns:
        DiarizationResult with all Phase 1 outputs
    
    Raises:
        RuntimeError: If any step fails
    """
    
    async def report(phase: str, percent: int, message: str):
        if progress_callback:
            await progress_callback(phase, percent, message)
        logger.info(f"[Phase 1] {percent}%: {message}")
    
    try:
        # Step 1: Audio Extraction
        await report("extraction", 5, "Extracting audio with FFmpeg...")
        
        master_wav = str(Path("temp_chunks") / task_id / "master.wav")
        extract_audio(video_path, master_wav)
        
        await report("extraction", 15, "Audio extracted successfully")
        
        # Step 2: Speaker Identification
        await report("diarization", 20, "Running speaker diarization...")
        
        identifier = SpeakerIdentifier(max_speakers=10)
        
        def diar_progress(msg, pct, total):
            asyncio.create_task(report("diarization", pct, msg))
        
        segments, assignments, num_speakers = identifier.run(
            master_wav,
            progress_callback=diar_progress
        )
        
        await report("diarization", 70, f"Diarization complete: {num_speakers} speakers")
        
        # Step 3: Extract speaker samples
        await report("samples", 75, "Generating reference samples...")
        
        def sample_progress(msg):
            asyncio.create_task(report("samples", 80, msg))
        
        speaker_samples, speaker_config = extract_speaker_samples(
            master_wav, segments, task_id, sample_progress
        )
        
        await report("samples", 85, "Samples extracted")
        
        # CHECKPOINT: Save state after diarization (before transcription)
        # This allows resuming from transcription if needed
        db.update_task(
            task_id,
            segments=[s.to_dict() for s in segments],
            assignments=assignments,
            speaker_count=num_speakers,
            speaker_samples=speaker_samples,
            speaker_config=speaker_config,
            master_audio=master_wav,
            phase="identifying",
            status="processing",
            progress=85,
            message="Speaker identification complete - starting transcription..."
        )
        
        if checkpoint_after_diarization:
            logger.info(f"Checkpoint saved after diarization for task {task_id}")
        
        # Step 4: Transcription
        await report("transcription", 85, "Transcribing speech segments...")
        
        def transcribe_progress(current, total):
            pct = 85 + int((current / total) * 10)
            asyncio.create_task(report("transcription", pct, 
                f"Transcribed {current}/{total} segments"))
        
        segments = transcribe_segments(
            master_wav, segments, source_language, transcribe_progress
        )
        
        await report("transcription", 95, "Transcription complete")
        
        # Final checkpoint: Ready for validation - NOW WITH TRANSCRIPTION REVIEW
        db.update_task(
            task_id,
            segments=[s.to_dict() for s in segments],
            transcribed_segments=[s.to_dict() for s in segments],
            phase="awaiting_validation",
            status="awaiting_validation",
            progress=35,
            message="Phase 1 complete - review and edit transcriptions below"
        )
        
        # Send transcription data to UI via WebSocket
        from ..state import broadcast_to_task
        await broadcast_to_task(task_id, {
            'type': 'transcription_ready',
            'data': {
                'segments': [s.to_dict() for s in segments],
                'speaker_config': speaker_config,
                'speaker_samples': speaker_samples,
                'can_edit': True,
                'next_phase': 'translation_review'  # Will go to translation review next
            }
        })
        
        await report("complete", 100, "Phase 1 complete - awaiting transcription review")
        
        return DiarizationResult(
            segments=segments,
            speaker_count=num_speakers,
            speaker_samples=speaker_samples,
            assignments=assignments,
            master_audio=master_wav,
            speaker_config=speaker_config
        )
        
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Phase 1 failed:\n{tb_str}")
        
        db.update_task(
            task_id,
            status="failed",
            phase="identifying",
            error_message=f"Phase 1 failed: {str(e)}",
            error_traceback=tb_str
        )
        
        raise RuntimeError(f"Phase 1 diarization failed: {str(e)}") from e


# Convenience function for non-async callers
def run_phase1_sync(*args, **kwargs) -> DiarizationResult:
    """Synchronous wrapper for run_phase1."""
    return asyncio.run(run_phase1(*args, **kwargs))


# Export for workflow tasks
__all__ = [
    'run_phase1',
    'run_phase1_sync',
    'DiarizationResult',
    'SpeechSegment',
    'extract_audio',
    'run_vad',
    'SpeakerIdentifier',
    'extract_speaker_samples',
    'transcribe_segments'
]
