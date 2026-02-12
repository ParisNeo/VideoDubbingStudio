"""
Phase 1: Speaker Identification and Diarization

Complete diarization pipeline: Audio Extraction → VAD → Embedding Extraction → Clustering.
Also includes immediate transcription for UI preview.

This phase extracts audio from video, identifies speakers, and provides
reference voice samples for validation.

Design principles:
- Sequential processing with progress reporting
- Immediate transcription for real-time UI feedback
- State persistence for resume support
- Sample extraction for each detected speaker
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
import tempfile
import warnings
import gc
import torch.nn.functional as F

from core.resources import manager
from core.database import db
from modules.translate_video.state import broadcast_to_task
from modules.translate_video.project_manager import ProjectManager, append_log

from .phase1_models import SpeechSegment, DiarizationResult

# Thread pool for CPU-bound diarization
_diarization_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

logger = logging.getLogger("phase1_diarization")


# =============================================================================
# DIARIZATION IMPLEMENTATION (moved from missing diarization.py module)
# =============================================================================

class AudioDiarizer:
    """
    Speaker diarization using WavLM embeddings and agglomerative clustering.
    Optimized for 8GB VRAM with sequential model loading.
    """
    
    def __init__(self, device: Optional[torch.device] = None, 
                 min_speakers: int = 1, 
                 max_speakers: int = 5,  # Reduced default max to avoid over-splitting
                 similarity_threshold: float = 0.85):  # Higher threshold = fewer speakers
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.vad_model = None
        self.embedding_model = None
        self.feature_extractor = None
        self.sampling_rate = 16000
        
        # Clustering parameters
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold  # Cosine similarity threshold for same speaker
        
    def _load_vad(self):
        """Load Silero VAD model."""
        if self.vad_model is None:
            try:
                import torch
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.vad_model = model.to(self.device)
                self.vad_get_speech_timestamps = utils[0]
                self.vad_read_audio = utils[1]
                self.vad_collect_chunks = utils[2]
            except Exception as e:
                logger.error(f"Failed to load VAD model: {e}")
                raise RuntimeError(f"Failed to load VAD model: {e}")
    
    def _load_embedding_model(self):
        """Load WavLM speaker embedding model."""
        if self.embedding_model is None:
            try:
                from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
                
                model_id = "microsoft/wavlm-base-plus-sv"
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
                self.embedding_model = WavLMForXVector.from_pretrained(model_id).to(self.device)
                self.embedding_model.eval()
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def _unload_models(self):
        """Unload models to free VRAM."""
        self.vad_model = None
        self.embedding_model = None
        self.feature_extractor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run(self, audio_path: str, min_speech_duration: float = 0.5,
            progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
        """
        Run complete diarization pipeline.
        
        Args:
            audio_path: Path to audio file
            min_speech_duration: Minimum speech segment duration in seconds
            progress_callback: Called with (message, current_percent, total_percent)
        
        Returns:
            Dict with 'segments', 'assignments', 'speaker_samples', 'timing'
        """
        import time
        start_time = time.time()
        
        # Load audio
        audio_np, sr = sf.read(audio_path)
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != self.sampling_rate:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=self.sampling_rate)
        
        # Step 1: Voice Activity Detection
        vad_start = time.time()
        if progress_callback:
            progress_callback("Running voice activity detection...", 0, 100)
        
        self._load_vad()
        speech_timestamps = self._detect_speech_vad(audio_np)
        
        # Filter short segments
        speech_timestamps = [
            ts for ts in speech_timestamps 
            if (ts['end'] - ts['start']) >= min_speech_duration
        ]
        
        vad_time = time.time() - vad_start
        
        if progress_callback:
            progress_callback(f"VAD complete: {len(speech_timestamps)} speech segments", 20, 100)
        
        # Step 2: Extract embeddings for each segment
        embed_start = time.time()
        if progress_callback:
            progress_callback("Extracting speaker embeddings...", 25, 100)
        
        self._load_embedding_model()
        
        segments = []
        embeddings = []
        
        for i, ts in enumerate(speech_timestamps):
            start_sample = int(ts['start'] * self.sampling_rate)
            end_sample = int(ts['end'] * self.sampling_rate)
            segment_audio = audio_np[start_sample:end_sample]
            
            # Extract embedding
            try:
                embedding = self._extract_embedding(segment_audio)
                embeddings.append(embedding)
                segments.append({
                    'idx': i,
                    'start': ts['start'],
                    'end': ts['end'],
                    'speaker_id': None  # Will be assigned by clustering
                })
                
                if progress_callback and i % 10 == 0:
                    progress_pct = 25 + int((i / len(speech_timestamps)) * 35)
                    progress_callback(f"Extracted {i+1}/{len(speech_timestamps)} embeddings...", progress_pct, 100)
                    
            except Exception as e:
                logger.warning(f"Failed to extract embedding for segment {i}: {e}")
        
        embed_time = time.time() - embed_start
        
        if progress_callback:
            progress_callback(f"Embeddings extracted: {len(embeddings)} segments", 60, 100)
        
        # Step 3: Cluster embeddings to identify speakers
        cluster_start = time.time()
        if progress_callback:
            progress_callback("Clustering speakers...", 65, 100)
        
        assignments = self._cluster_embeddings(embeddings)
        
        # Update segments with speaker assignments
        for i, seg in enumerate(segments):
            seg['speaker_id'] = assignments.get(i, 0)
        
        cluster_time = time.time() - cluster_start
        
        # Step 4: Extract speaker samples
        if progress_callback:
            progress_callback("Extracting speaker samples...", 85, 100)
        
        speaker_samples = self._extract_speaker_samples(audio_np, segments, assignments)
        
        total_time = time.time() - start_time
        
        # Unload models to free VRAM
        self._unload_models()
        
        if progress_callback:
            progress_callback("Diarization complete!", 100, 100)
        
        return {
            'segments': segments,
            'assignments': assignments,
            'speaker_samples': speaker_samples,
            'timing': {
                'vad_seconds': vad_time,
                'embed_seconds': embed_time,
                'cluster_seconds': cluster_time,
                'total_seconds': total_time
            }
        }
    
    def _detect_speech_vad(self, audio: np.ndarray) -> List[Dict[str, float]]:
        """Detect speech timestamps using Silero VAD."""
        import torch
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).to(self.device)
        
        # Get speech timestamps
        with torch.no_grad():
            speech_timestamps = self.vad_get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=self.sampling_rate,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                window_size_samples=512
            )
        
        # Convert to seconds
        return [
            {
                'start': ts['start'] / self.sampling_rate,
                'end': ts['end'] / self.sampling_rate
            }
            for ts in speech_timestamps
        ]
    
    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding using WavLM."""
        import torch
        
        # Prepare input
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            outputs = self.embedding_model(input_values)
            embeddings = outputs.embeddings
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()[0]
    
    def _cluster_embeddings(self, embeddings: List[np.ndarray]) -> Dict[int, int]:
        """
        Cluster embeddings using agglomerative clustering with similarity threshold.
        Returns mapping from segment index to speaker ID.
        """
        if len(embeddings) == 0:
            return {}
        
        if len(embeddings) == 1:
            return {0: 0}
        
        # Stack embeddings
        X = np.stack(embeddings)
        
        # First, try to determine optimal number of speakers using similarity analysis
        # High similarity between embeddings suggests same speaker
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute pairwise similarities
        similarities = cosine_similarity(X)
        
        # Use hierarchical clustering with distance threshold
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert similarity to distance (1 - similarity)
        # Use a more conservative threshold to avoid over-splitting
        distance_threshold = 1.0 - self.similarity_threshold  # 0.15 for 0.85 threshold
        
        # Try clustering with distance threshold first (more natural)
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,  # Let threshold decide
                metric='cosine',
                linkage='average',
                distance_threshold=distance_threshold
            )
            labels = clustering.fit_predict(X)
            n_found = len(set(labels))
            
            logger.info(f"Distance-based clustering found {n_found} speakers "
                       f"(threshold={distance_threshold:.3f})")
            
            # Validate against min/max constraints
            if n_found < self.min_speakers:
                # Force more clusters
                logger.info(f"Too few speakers ({n_found}), forcing {self.min_speakers}")
                clustering = AgglomerativeClustering(
                    n_clusters=self.min_speakers,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(X)
            elif n_found > self.max_speakers:
                # Force fewer clusters
                logger.info(f"Too many speakers ({n_found}), forcing {self.max_speakers}")
                clustering = AgglomerativeClustering(
                    n_clusters=self.max_speakers,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(X)
                
        except Exception as e:
            logger.warning(f"Distance-based clustering failed: {e}, falling back to n_clusters")
            # Fallback: use silhouette score to find optimal k
            from sklearn.metrics import silhouette_score
            
            best_n = self.min_speakers
            best_score = -1
            
            # Try different numbers of speakers within bounds
            max_k = min(len(embeddings), self.max_speakers)
            min_k = max(1, self.min_speakers)
            
            for n_speakers in range(min_k, max_k + 1):
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_speakers,
                        metric='cosine',
                        linkage='average'
                    )
                    test_labels = clustering.fit_predict(X)
                    
                    if len(set(test_labels)) > 1:  # Need at least 2 clusters for silhouette
                        score = silhouette_score(X, test_labels, metric='cosine')
                        if score > best_score:
                            best_score = score
                            best_n = n_speakers
                except Exception as e:
                    logger.debug(f"Clustering with {n_speakers} speakers failed: {e}")
            
            # Final clustering with best n
            clustering = AgglomerativeClustering(
                n_clusters=best_n,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(X)
        
        # Create assignment mapping
        assignments = {i: int(label) for i, label in enumerate(labels)}
        
        # Log speaker distribution
        unique_speakers = set(assignments.values())
        logger.info(f"Final clustering: {len(unique_speakers)} speakers "
                   f"(IDs: {sorted(unique_speakers)})")
        
        return assignments
    
    def _extract_speaker_samples(self, audio: np.ndarray, segments: List[Dict], 
                                  assignments: Dict[int, int]) -> Dict[int, str]:
        """
        Extract representative audio samples for each speaker.
        Returns mapping from speaker_id to sample audio path.
        """
        speaker_samples = {}
        
        # Group segments by speaker
        speaker_segments: Dict[int, List[Dict]] = {}
        for seg in segments:
            spk_id = seg['speaker_id']
            if spk_id not in speaker_segments:
                speaker_segments[spk_id] = []
            speaker_segments[spk_id].append(seg)
        
        # For each speaker, find the longest segment as sample
        for spk_id, segs in speaker_segments.items():
            best_seg = max(segs, key=lambda s: s['end'] - s['start'])
            
            # Extract audio
            start_sample = int(best_seg['start'] * self.sampling_rate)
            end_sample = int(best_seg['end'] * self.sampling_rate)
            sample_audio = audio[start_sample:end_sample]
            
            speaker_samples[spk_id] = sample_audio
        
        return speaker_samples


# =============================================================================
# PHASE 1 DIARIZER CLASS
# =============================================================================

class Phase1Diarizer:
    """
    Complete Phase 1 pipeline for speaker identification.
    
    VRAM-efficient: loads models sequentially, unloads after each step.
    """
    
    def __init__(
        self,
        task_id: str,
        video_path: str,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        transcription_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        max_speakers: int = 5,  # Conservative default
        similarity_threshold: float = 0.85  # Higher = fewer speakers
    ):
        self.task_id = task_id
        self.video_path = Path(video_path)
        self.progress_callback = progress_callback
        self.transcription_callback = transcription_callback
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        
        # Paths
        self.master_wav: Optional[Path] = None
        self.segments: List[SpeechSegment] = []
        self.speaker_samples: Dict[int, str] = {}
        
        logger.info(f"Phase1Diarizer initialized for task {task_id} "
                   f"(max_speakers={max_speakers}, threshold={similarity_threshold})")
    
    async def run(self) -> DiarizationResult:
        """
        Execute full Phase 1 pipeline.
        
        Returns:
            DiarizationResult with segments, samples, assignments, and transcripts
        """
        try:
            # Step 1: Audio Extraction
            await self._report_progress("audio_extraction", 5, 
                "Extracting audio with FFmpeg...")
            self.master_wav = await self._extract_audio()
            await self._report_progress("audio_extraction", 10, 
                "Audio extracted successfully")
            
            # Step 2: Voice Activity Detection and Diarization
            await self._report_progress("diarization", 15, 
                "Running speaker diarization...")
            diarization_data = await self._run_diarization()
            
            # Create SpeechSegment objects
            self.segments = [
                SpeechSegment(
                    idx=i,
                    start=seg['start'],
                    end=seg['end'],
                    speaker_id=seg.get('speaker_id'),
                    embedding=None  # Will be populated if needed
                )
                for i, seg in enumerate(diarization_data.get('segments', []))
            ]
            
            # Get speaker assignments
            assignments = diarization_data.get('assignments', {})
            
            # Update segments with speaker IDs from assignments
            for idx, speaker_id in assignments.items():
                if int(idx) < len(self.segments):
                    self.segments[int(idx)].speaker_id = speaker_id
            
            num_speakers = len(set(assignments.values())) if assignments else 0
            await self._report_progress("diarization", 30, 
                f"Diarization complete: {num_speakers} speaker{'s' if num_speakers != 1 else ''}")
            
            # Step 3: Immediate Transcription for UI
            await self._report_progress("transcription", 30, 
                "Transcribing speech segments for preview...")
            transcribed_segments = await self._transcribe_segments()
            
            # Broadcast transcription to UI
            if self.transcription_callback:
                await self.transcription_callback(transcribed_segments)
            
            await self._report_progress("transcription", 35, 
                "Transcription complete - review below")
            
            # Step 4: Extract Speaker Samples
            await self._report_progress("samples", 35, 
                "Generating reference samples...")
            self.speaker_samples = await self._extract_speaker_samples(
                diarization_data.get('speaker_samples', {})
            )
            
            # Build speaker config
            speaker_config = self._build_speaker_config(assignments)
            
            # Save state to database
            self._save_state(assignments, transcribed_segments, speaker_config)
            
            await self._report_progress("complete", 35, 
                f"Phase 1 complete: {len(self.segments)} segments, "
                f"{len(self.speaker_samples)} speaker{'s' if len(self.speaker_samples) != 1 else ''}")
            
            return DiarizationResult(
                segments=self.segments,
                speaker_count=num_speakers,
                speaker_samples=self.speaker_samples,
                assignments=assignments,
                transcribed_segments=transcribed_segments,
                master_audio=str(self.master_wav),
                speaker_config=speaker_config
            )
            
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Phase 1 failed with traceback:\n{tb_str}")
            raise RuntimeError(f"Phase 1 diarization failed: {str(e)}")
    
    async def _extract_audio(self) -> Path:
        """Extract audio from video to WAV format."""
        output_wav = Path("temp_chunks") / f"{self.task_id}_master.wav"
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(output_wav)
        ]
        
        # Run FFmpeg asynchronously
        loop = asyncio.get_event_loop()
        
        def _run_ffmpeg():
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            return output_wav
        
        result_path = await loop.run_in_executor(None, _run_ffmpeg)
        return result_path
    
    async def _run_diarization(self) -> Dict[str, Any]:
        """Run diarization using the embedded AudioDiarizer."""
        # Check GPU
        if torch.cuda.is_available():
            await self._log(f"GPU detected: {torch.cuda.get_device_name(0)}", "info")
        else:
            await self._log("WARNING: No GPU detected, using CPU (slow)", "warning")
        
        # Read audio in main thread
        audio_np, sr = sf.read(str(self.master_wav))
        
        # Create a queue to receive progress from the worker thread
        import queue
        progress_queue = queue.Queue()
        
        def run_diarization_with_progress():
            """Run diarization and put progress updates in queue."""
            def on_progress(msg, pct, total):
                progress_queue.put(("progress", msg, pct))
            
            diarizer = AudioDiarizer(
                device=None,
                max_speakers=self.max_speakers,
                similarity_threshold=self.similarity_threshold
            )
            result = diarizer.run(str(self.master_wav), min_speech_duration=0.5,
                                progress_callback=on_progress)
            
            progress_queue.put(("done", result))
            return result
        
        # Start diarization in executor
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_diarization_executor, run_diarization_with_progress)
        
        # Poll for progress while waiting
        last_progress = 15  # Start from 15%
        while True:
            # Check if future is done
            if future.done():
                try:
                    result = future.result()
                    break
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"Diarization failed with traceback:\n{tb_str}")
                    await self._log(f"Diarization failed: {str(e)}", "error")
                    raise e
            
            # Drain progress queue
            try:
                while True:
                    item = progress_queue.get_nowait()
                    if item[0] == "progress":
                        _, msg, pct = item
                        # Map 0-100 diarization progress to 15-30 overall
                        overall_pct = 15 + int(pct * 0.15)
                        if overall_pct > last_progress:
                            last_progress = overall_pct
                            # Don't block on progress reporting
                            asyncio.create_task(self._report_progress(
                                "diarization", overall_pct, msg
                            ))
            except queue.Empty:
                pass
            
            # Small sleep to prevent busy-waiting
            await asyncio.sleep(0.1)
        
        # Log timing info
        if 'timing' in result:
            t = result['timing']
            await self._log(f"Diarization timing: "
                           f"VAD={t['vad_seconds']:.1f}s, "
                           f"Embeds={t['embed_seconds']:.1f}s, "
                           f"Cluster={t['cluster_seconds']:.1f}s, "
                           f"Total={t['total_seconds']:.1f}s", "info")
        
        return result
    
    async def _transcribe_segments(self) -> List[Dict[str, Any]]:
        """
        Transcribe all segments using Whisper for immediate UI display.
        Uses direct model inference to avoid pipeline issues on Windows.
        """
        from transformers import Wav2Vec2FeatureExtractor
        import torch.nn.functional as F
        
        segments_with_text = []
        total = len(self.segments)
        
        if not self.segments:
            return segments_with_text
        
        try:
            # Get the raw model and processor from the manager
            whisper_pipe = manager.get_whisper()
            model = whisper_pipe.model
            processor = whisper_pipe.tokenizer
            
            # We need the feature extractor from the processor
            if hasattr(whisper_pipe, 'feature_extractor'):
                feature_extractor = whisper_pipe.feature_extractor
            else:
                # Fallback: load feature extractor directly
                from transformers import AutoFeatureExtractor
                feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v2")
            
            device = manager.device
            
            # Get the model's dtype and ensure inputs match
            model_dtype = next(model.parameters()).dtype
            
            # Load master audio
            audio_np, sr = sf.read(str(self.master_wav))
            
            for i, seg in enumerate(self.segments):
                try:
                    # Extract audio segment
                    start_sample = int(seg.start * sr)
                    end_sample = int(seg.end * sr)
                    chunk = audio_np[start_sample:end_sample]
                    
                    # Ensure mono and correct shape
                    if len(chunk.shape) > 1:
                        chunk = chunk.mean(axis=1)
                    
                    # Resample to 16kHz if needed
                    if sr != 16000:
                        import librosa
                        chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
                    
                    # Convert audio to float32 for feature extractor
                    chunk = chunk.astype(np.float32)
                    
                    # Process audio with feature extractor
                    inputs = feature_extractor(
                        chunk, 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    )
                    
                    # Convert input_features to match model dtype
                    input_features = inputs.input_features.to(device).to(model_dtype)
                    
                    # Generate with the model directly
                    with torch.no_grad():
                        predicted_ids = model.generate(
                            input_features,
                            max_length=448,
                            num_beams=1,
                            condition_on_prev_tokens=False,
                        )
                    
                    # Decode
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    text = transcription.strip()
                    
                    # Create enriched segment
                    seg_with_text = {
                        "idx": seg.idx,
                        "start": seg.start,
                        "end": seg.end,
                        "speaker_id": seg.speaker_id if seg.speaker_id is not None else 0,
                        "original_text": text,
                        "translated_text": "",  # Will be filled in Phase 2
                        "status": "transcribed"
                    }
                    segments_with_text.append(seg_with_text)
                    
                    # Progress update every few segments
                    if i % 5 == 0 or i == total - 1:
                        progress_pct = 30 + int((i / total) * 5)  # 30-35% range
                        await self._report_progress("transcription", progress_pct, 
                            f"Transcribed {i+1}/{total} segments...")
                    
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"Failed to transcribe segment {i}:\n{tb_str}")
                    segments_with_text.append({
                        "idx": seg.idx,
                        "start": seg.start,
                        "end": seg.end,
                        "speaker_id": seg.speaker_id if seg.speaker_id is not None else 0,
                        "original_text": f"[Transcription error: {str(e)[:50]}]",
                        "translated_text": "",
                        "status": "error",
                        "error": str(e)
                    })
            
            # Clear model from memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Failed to load Whisper for transcription:\n{tb_str}")
            # Return empty transcriptions with error info
            for seg in self.segments:
                segments_with_text.append({
                    "idx": seg.idx,
                    "start": seg.start,
                    "end": seg.end,
                    "speaker_id": seg.speaker_id if seg.speaker_id is not None else 0,
                    "original_text": "[Whisper initialization failed]",
                    "translated_text": "",
                    "status": "error",
                    "error": str(e)
                })
        
        return segments_with_text
    
    async def _extract_speaker_samples(
        self, 
        existing_samples: Dict[int, Any]
    ) -> Dict[int, str]:
        """
        Extract or use existing speaker samples.
        Ensures all samples are accessible paths.
        """
        samples = {}
        samples_dir = Path("temp_chunks") / self.task_id / "speaker_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Load master audio for sample extraction
        audio_np, sr = sf.read(str(self.master_wav))
        
        for speaker_id, sample_data in existing_samples.items():
            # sample_data could be numpy array (from AudioDiarizer) or path string
            if isinstance(sample_data, np.ndarray):
                # Save numpy array to file
                sample_path = samples_dir / f"speaker_{speaker_id}_sample.wav"
                sf.write(str(sample_path), sample_data, sr)
                samples[int(speaker_id)] = str(sample_path.absolute())
            elif isinstance(sample_data, str) and Path(sample_data).exists():
                # Already a path
                samples[int(speaker_id)] = str(Path(sample_data).absolute())
            else:
                # Try to find or generate sample
                # Group segments by speaker
                speaker_segments = [s for s in self.segments if s.speaker_id == int(speaker_id)]
                if speaker_segments:
                    # Find longest segment for this speaker
                    best_seg = max(speaker_segments, key=lambda s: s.end - s.start)
                    
                    # Extract audio
                    start_sample = int(best_seg.start * sr)
                    end_sample = int(best_seg.end * sr)
                    sample_audio = audio_np[start_sample:end_sample]
                    
                    # Save sample
                    sample_path = samples_dir / f"speaker_{speaker_id}_sample.wav"
                    sf.write(str(sample_path), sample_audio, sr)
                    samples[int(speaker_id)] = str(sample_path.absolute())
        
        return samples
    
    def _build_speaker_config(self, assignments: Dict[int, int]) -> Dict[str, Any]:
        """Build speaker configuration dict for validation UI."""
        config = {}
        unique_speakers = set(assignments.values()) if assignments else set()
        
        for spk_id in unique_speakers:
            sample_path = self.speaker_samples.get(spk_id, "")
            # Convert to URL-style path for frontend
            if sample_path:
                path_obj = Path(sample_path)
                if path_obj.exists():
                    # Create URL path: /temp_chunks/...
                    relative = path_obj.relative_to(Path.cwd())
                    sample_url = f"/{relative.as_posix()}"
                else:
                    sample_url = ""
            else:
                sample_url = ""
            
            config[str(spk_id)] = {
                "name": f"Speaker {int(spk_id) + 1}",
                "action": "dub",
                "sample_path": sample_url
            }
        
        return config
    
    def _save_state(
        self, 
        assignments: Dict[int, int],
        transcribed_segments: List[Dict[str, Any]],
        speaker_config: Dict[str, Any]
    ):
        """Save Phase 1 results to database and project manager."""
        # Save to ProjectManager (JSON file)
        ProjectManager.save_state(self.task_id, {
            "segments": [
                {
                    "idx": s.idx,
                    "start": s.start,
                    "end": s.end,
                    "speaker_id": s.speaker_id
                }
                for s in self.segments
            ],
            "assignments": assignments,
            "master_audio": str(self.master_wav),
            "transcribed_segments": transcribed_segments,
            "speaker_config": speaker_config
        })
        
        # Save to SQLite database
        db.update_task(
            self.task_id,
            segments=[
                {
                    "idx": s.idx,
                    "start": s.start,
                    "end": s.end,
                    "speaker_id": s.speaker_id
                }
                for s in self.segments
            ],
            assignments=assignments,
            master_audio=str(self.master_wav),
            transcribed_segments=transcribed_segments,
            speaker_config=speaker_config,
            phase='awaiting_validation',
            status='awaiting_validation',
            progress=35,
            message='Speaker identification complete - review speakers below'
        )
    
    async def _report_progress(self, phase: str, percent: int, message: str):
        """Report progress via callback and broadcast."""
        if self.progress_callback:
            try:
                await self.progress_callback(phase, percent, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Also broadcast via WebSocket
        try:
            await broadcast_to_task(self.task_id, {
                'type': 'progress',
                'data': {
                    'phase': phase,
                    'percent': percent,
                    'message': message
                }
            })
        except Exception as e:
            logger.warning(f"WebSocket broadcast failed: {e}")
        
        # Update database
        try:
            db.update_task(
                self.task_id,
                progress=percent,
                message=message
            )
        except Exception as e:
            logger.warning(f"Database update failed: {e}")
    
    async def _log(self, message: str, style: str = "info"):
        """Log to disk and WebSocket."""
        # Disk
        append_log(self.task_id, message, style)
        
        # WebSocket
        try:
            await broadcast_to_task(self.task_id, {
                'type': 'log',
                'data': {
                    'message': message,
                    'style': style
                }
            })
        except Exception as e:
            logger.warning(f"WebSocket log failed: {e}")


async def run_phase1_diarization(
    task_id: str,
    video_path: str,
    progress_callback: Optional[Callable[[str, int, str], None]] = None,
    transcription_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    max_speakers: int = 5,
    similarity_threshold: float = 0.85
) -> DiarizationResult:
    """
    Convenience function to run Phase 1 diarization.
    
    Args:
        task_id: The task ID
        video_path: Path to input video file
        progress_callback: Optional callback for progress updates
        transcription_callback: Optional callback for transcription updates
        max_speakers: Maximum number of speakers to detect (default 5)
        similarity_threshold: Cosine similarity threshold for same speaker (default 0.85, higher = fewer speakers)
    
    Returns:
        DiarizationResult with all Phase 1 outputs
    """
    diarizer = Phase1Diarizer(
        task_id=task_id,
        video_path=video_path,
        progress_callback=progress_callback,
        transcription_callback=transcription_callback,
        max_speakers=max_speakers,
        similarity_threshold=similarity_threshold
    )
    
    return await diarizer.run()
