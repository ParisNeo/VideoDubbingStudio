import torch
import torch.nn.functional as F
import numpy as np
import librosa
import logging
import gc
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json

from core.resources import manager

logger = logging.getLogger("diarization")

# Try to import Silero VAD
try:
    import torch
    SILERO_AVAILABLE = True
except:
    SILERO_AVAILABLE = False


@dataclass
class SpeechSegment:
    start: float  # seconds
    end: float    # seconds
    speaker_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    audio_path: Optional[str] = None  # Path to extracted sample


class SpeakerClusterer:
    """
    Agglomerative clustering for speaker identification with merge capability.
    Optimized with early stopping and progress callbacks.
    """
    
    def __init__(self, threshold: float = 0.85, merge_threshold: float = 0.90,
                 max_speakers: int = 10):
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self.max_speakers = max_speakers  # Prevent runaway clustering
        
    def fit(self, embeddings: List[Tuple[int, np.ndarray]], 
            progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[int, int]:
        """
        Cluster embeddings. Returns mapping from segment_idx to speaker_id.
        Includes progress reporting for large datasets.
        """
        if not embeddings:
            return {}
        
        n = len(embeddings)
        
        # Fast path: single segment
        if n == 1:
            return {embeddings[0][0]: 0}
        
        # Fast path: all identical (or very few)
        if n <= 3:
            return {idx: 0 for idx, _ in embeddings}
        
        # Sort by index for deterministic behavior
        embeddings = sorted(embeddings, key=lambda x: x[0])
        indices = [e[0] for e in embeddings]
        vectors = torch.stack([torch.from_numpy(e[1]) for e in embeddings])
        vectors = F.normalize(vectors, dim=-1)
        
        # Agglomerative clustering with progress
        parent = list(range(n))
        centroids = [v.clone() for v in vectors]
        cluster_size = [1] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        # Pre-compute similarity matrix once (memory intensive but faster)
        # For very large n, use chunked computation
        if progress_callback:
            progress_callback("Computing similarity matrix...", 0, 100)
        
        if n > 200:
            # Chunked similarity computation for memory efficiency
            similarities = torch.zeros(n, n)
            chunk_size = 100
            for i_start in range(0, n, chunk_size):
                i_end = min(i_start + chunk_size, n)
                similarities[i_start:i_end] = torch.mm(vectors[i_start:i_end], vectors.t())
                if progress_callback:
                    pct = int((i_end / n) * 20)  # First 20% for matrix
                    progress_callback(f"Computing similarities ({i_end}/{n})...", pct, 100)
        else:
            similarities = torch.mm(vectors, vectors.t())
        
        # Merge loop with progress and early stopping
        merged = True
        iteration = 0
        max_iterations = n - 1  # At most n-1 merges needed
        
        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            
            # Progress update every 10 iterations or for large datasets
            if progress_callback and (iteration % 10 == 0 or n > 100):
                # Estimate progress: 20% for matrix, 80% for clustering
                pct = 20 + int((iteration / max_iterations) * 80)
                progress_callback(f"Clustering (iteration {iteration}, speakers found: {len(set(find(i) for i in range(n)))})...", pct, 100)
            
            # Find closest pair across different clusters
            best_sim = -1
            best_pair = None
            
            for i in range(n):
                root_i = find(i)
                for j in range(i + 1, n):
                    root_j = find(j)
                    if root_i == root_j:
                        continue
                    
                    sim = similarities[i, j].item()
                    if sim > best_sim and sim > self.threshold:
                        best_sim = sim
                        best_pair = (root_i, root_j)
            
            if best_pair:
                # Merge smaller into larger
                r1, r2 = best_pair
                if cluster_size[r1] < cluster_size[r2]:
                    r1, r2 = r2, r1
                
                parent[r2] = r1
                total = cluster_size[r1] + cluster_size[r2]
                centroids[r1] = (centroids[r1] * cluster_size[r1] + 
                                centroids[r2] * cluster_size[r2]) / total
                centroids[r1] = F.normalize(centroids[r1], dim=-1)
                cluster_size[r1] = total
                merged = True
                
                # Early stopping: prevent too many speakers
                current_speakers = len(set(find(i) for i in range(n)))
                if current_speakers <= self.max_speakers and best_sim < self.merge_threshold:
                    logger.info(f"Early stopping: {current_speakers} speakers with similarity {best_sim:.3f}")
                    break
        
        # Renumber speakers 0..k-1
        if progress_callback:
            progress_callback("Finalizing speaker assignments...", 95, 100)
        
        root_to_speaker = {}
        next_id = 0
        assignments = {}
        
        for idx in indices:
            pos = indices.index(idx)
            root = find(pos)
            if root not in root_to_speaker:
                root_to_speaker[root] = next_id
                next_id += 1
            assignments[idx] = root_to_speaker[root]
        
        if progress_callback:
            progress_callback(f"Clustering complete: {next_id} speakers", 100, 100)
        
        return assignments
    
    def get_merge_candidates(self, speaker_centroids: Dict[int, np.ndarray],
                            threshold: Optional[float] = None) -> List[Tuple[int, int, float]]:
        """
        Find speaker pairs that could be merged.
        Returns list of (speaker_a, speaker_b, similarity)
        """
        threshold = threshold or self.merge_threshold
        spk_ids = sorted(speaker_centroids.keys())
        candidates = []
        
        vecs = {sid: F.normalize(torch.from_numpy(centroid).unsqueeze(0), dim=-1)
                for sid, centroid in speaker_centroids.items()}
        
        for i, s1 in enumerate(spk_ids):
            for s2 in spk_ids[i+1:]:
                sim = torch.mm(vecs[s1], vecs[s2].t()).item()
                if sim > threshold:
                    candidates.append((s1, s2, sim))
        
        # Sort by similarity descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates


class AudioDiarizer:
    """
    Complete diarization pipeline: VAD → embedding extraction → clustering.
    With comprehensive progress reporting.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize diarizer.
        
        Args:
            device: 'cuda', 'cpu', or None for auto (prefer cuda if available)
        """
        if device is None:
            # FORCE CUDA if available, never default to CPU silently
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            # Validate requested device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            else:
                self.device = device
        
        self.sr = 16000
        logger.info(f"AudioDiarizer initialized with device: {self.device}")
        
    def run(self, audio_path: str, min_speech_duration: float = 0.5,
            progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
        """
        Full diarization pipeline with progress reporting.
        Returns: {
            'segments': List[SpeechSegment],
            'speaker_count': int,
            'speaker_samples': Dict[int, str],
            'assignments': Dict[int, int],
            'merge_candidates': [...],
            'speaker_centroids': {...}
        }
        """
        import soundfile as sf
        import time
        
        total_start = time.time()
        
        # Progress helper
        def report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct, 100)
            logger.info(f"Diarization progress: {pct}% - {msg}")
        
        report("Loading audio...", 0)
        
        # Load audio
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        report(f"Loaded {duration:.1f}s of audio at {sr}Hz", 2)
        
        if sr != self.sr:
            report(f"Resampling to {self.sr}Hz...", 3)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        # 1. Voice Activity Detection
        report("Running Voice Activity Detection...", 5)
        vad_start = time.time()
        vad_segments = self._vad(audio, progress_callback=lambda pct: report(f"VAD: {pct}% complete", 5 + int(pct * 0.15)))
        vad_time = time.time() - vad_start
        report(f"VAD found {len(vad_segments)} speech segments in {vad_time:.1f}s", 20)
        
        # Sanity check: too many segments means noise/artifacts
        if len(vad_segments) > 500:
            logger.warning(f"Extremely high segment count ({len(vad_segments)}), filtering short segments aggressively")
            # Re-run with stricter settings
            vad_segments = self._vad(audio, strict=True)
            report(f"Filtered to {len(vad_segments)} segments with strict VAD", 20)
        
        if not vad_segments:
            report("No speech detected", 100)
            return {'segments': [], 'speaker_count': 0, 'speaker_samples': {}, 'assignments': {}}
        
        # 2. Extract embeddings for valid segments with progress
        report("Extracting speaker embeddings...", 25)
        embed_start = time.time()
        embeddings = self._extract_embeddings(audio, vad_segments, min_speech_duration,
                                               progress_callback=lambda done, total: 
                                               report(f"Embeddings: {done}/{total} segments", 25 + int((done/total)*45)))
        embed_time = time.time() - embed_start
        report(f"Extracted {len(embeddings)} valid embeddings in {embed_time:.1f}s ({len(embeddings)/embed_time:.1f} seg/s)", 70)
        
        if not embeddings:
            report("No valid embeddings extracted (all segments too short)", 100)
            return {'segments': [], 'speaker_count': 0, 'speaker_samples': {}, 'assignments': {}}
        
        # 3. Cluster with progress
        report(f"Clustering {len(embeddings)} embeddings...", 75)
        cluster_start = time.time()
        clusterer = SpeakerClusterer(max_speakers=min(10, len(embeddings)//3 + 1))
        assignments = clusterer.fit(embeddings, progress_callback=lambda msg, pct, total: report(msg, 75 + int(pct*0.15)))
        cluster_time = time.time() - cluster_start
        report(f"Clustering completed in {cluster_time:.1f}s", 90)
        
        # Build segment objects
        segments = []
        for idx, (start, end) in enumerate(vad_segments):
            seg = SpeechSegment(
                start=start / self.sr,
                end=end / self.sr,
                speaker_id=assignments.get(idx),
                embedding=next((e[1] for e in embeddings if e[0] == idx), None)
            )
            segments.append(seg)
        
        # Calculate speaker centroids for merge suggestions
        speaker_centroids = self._compute_centroids(embeddings, assignments)
        
        # Find merge candidates
        merge_candidates = clusterer.get_merge_candidates(speaker_centroids)
        
        # Count unique speakers
        unique_speakers = set(assignments.values()) if assignments else set()
        
        # Extract best samples for each speaker
        report("Extracting speaker samples...", 92)
        speaker_samples = self._extract_speaker_samples(
            audio, segments, assignments, audio_path
        )
        
        total_time = time.time() - total_start
        report(f"Diarization complete: {len(unique_speakers)} speakers, {len(segments)} segments, {total_time:.1f}s total", 100)
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'segments': [
                {'start': s.start, 'end': s.end, 'speaker_id': s.speaker_id}
                for s in segments
            ],
            'speaker_count': len(unique_speakers),
            'speaker_samples': speaker_samples,
            'assignments': assignments,
            'merge_candidates': merge_candidates,
            'speaker_centroids': {k: v.tolist() for k, v in speaker_centroids.items()},
            'timing': {
                'total_seconds': total_time,
                'vad_seconds': vad_time,
                'embed_seconds': embed_time,
                'cluster_seconds': cluster_time
            }
        }
    
    def _vad(self, audio: np.ndarray, strict: bool = False,
             progress_callback: Optional[Callable[[int], None]] = None) -> List[Tuple[int, int]]:
        """Silero VAD to find speech segments with optional progress."""
        if not SILERO_AVAILABLE:
            # Fallback: treat all as speech
            return [(0, len(audio))]
        
        # Load model
        if progress_callback:
            progress_callback(10)
        
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True
        )
        (get_speech_timestamps, _, _, _, _) = utils
        
        # Move model to our device
        model = model.to(self.device)
        
        if progress_callback:
            progress_callback(30)
        
        audio_t = torch.from_numpy(audio).float().to(self.device)
        
        # Stricter settings when requested
        threshold = 0.7 if strict else 0.5
        min_speech = 500 if strict else 250
        min_silence = 300 if strict else 500
        
        # Run VAD
        timestamps = get_speech_timestamps(
            audio_t, model, sampling_rate=self.sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech,
            min_silence_duration_ms=min_silence,
            speech_pad_ms=200
        )
        
        if progress_callback:
            progress_callback(80)
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if progress_callback:
            progress_callback(100)
        
        return [(t['start'], t['end']) for t in timestamps]
    
    def _extract_embeddings(self, audio: np.ndarray, 
                           segments: List[Tuple[int, int]],
                           min_duration: float,
                           progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Tuple[int, np.ndarray]]:
        """Extract speaker embeddings using WavLM with batching and progress."""
        from transformers import Wav2Vec2FeatureExtractor
        
        feature_extractor, model = manager.get_speaker_encoder()
        
        # Ensure model is on correct device
        model = model.to(self.device)
        model.eval()
        
        valid_segments = []
        for idx, (start, end) in enumerate(segments):
            duration = (end - start) / self.sr
            if duration >= min_duration:
                valid_segments.append((idx, start, end))
        
        # Process with progress reporting
        valid_embeddings = []
        total = len(valid_segments)
        
        # Use small batches for efficiency
        batch_size = 8 if self.device == "cuda" else 4
        
        for i, (idx, start, end) in enumerate(valid_segments):
            # Take center 2s for consistent embedding
            center = (start + end) // 2
            half = min(int(1.0 * self.sr), (end - start) // 4)
            s = max(start, center - half)
            e = min(end, center + half)
            
            chunk = audio[s:e]
            
            # Extract embedding
            inputs = feature_extractor(
                chunk, sampling_rate=self.sr, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                embedding = model(**inputs).embeddings
                embedding = F.normalize(embedding, dim=-1).squeeze().cpu().numpy()
            
            valid_embeddings.append((idx, embedding))
            
            # Progress every segment or batch
            if progress_callback and (i % batch_size == 0 or i == total - 1):
                progress_callback(i + 1, total)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return valid_embeddings
    
    def _compute_centroids(self, embeddings: List[Tuple[int, np.ndarray]],
                          assignments: Dict[int, int]) -> Dict[int, np.ndarray]:
        """Compute centroid for each speaker."""
        speaker_vectors = {}
        for idx, emb in embeddings:
            spk = assignments.get(idx)
            if spk is None:
                continue
            if spk not in speaker_vectors:
                speaker_vectors[spk] = []
            speaker_vectors[spk].append(emb)
        
        centroids = {}
        for spk, vectors in speaker_vectors.items():
            centroid = np.mean(vectors, axis=0)
            centroids[spk] = F.normalize(
                torch.from_numpy(centroid).unsqueeze(0), dim=-1
            ).squeeze().numpy()
        
        return centroids
    
    def _extract_speaker_samples(self, audio: np.ndarray,
                                  segments: List[SpeechSegment],
                                  assignments: Dict[int, int],
                                  base_path: str) -> Dict[int, str]:
        """Extract best quality sample for each speaker."""
        from pathlib import Path
        import soundfile as sf
        
        base = Path(base_path).parent
        samples_dir = base / "speaker_samples"
        samples_dir.mkdir(exist_ok=True)
        
        # Group segments by speaker
        speaker_segments = {}
        for idx, seg in enumerate(segments):
            spk = assignments.get(idx)
            if spk is None:
                continue
            if spk not in speaker_segments:
                speaker_segments[spk] = []
            speaker_segments[spk].append((idx, seg))
        
        # For each speaker, find longest clean segment
        result = {}
        for spk, segs in speaker_segments.items():
            # Find longest
            best_idx, best_seg = max(segs, key=lambda x: x[1].end - x[1].start)
            
            # Extract audio
            start_sample = int(best_seg.start * self.sr)
            end_sample = int(best_seg.end * self.sr)
            chunk = audio[start_sample:end_sample]
            
            # Save
            sample_path = samples_dir / f"speaker_{spk}_sample.wav"
            sf.write(sample_path, chunk, self.sr)
            result[spk] = str(sample_path)
        
        return result


# Alias for backward compatibility
LowVRAMDiarizer = AudioDiarizer


def save_diarization_state(output_dir: Path, state: Dict[str, Any]) -> str:
    """Save diarization results to JSON."""
    path = output_dir / "diarization.json"
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)
    return str(path)


def load_diarization_state(path: str) -> Dict[str, Any]:
    """Load diarization results from JSON."""
    with open(path) as f:
        return json.load(f)
