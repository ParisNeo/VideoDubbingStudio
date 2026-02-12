"""
Phase 1: Audio Processing & Diarization Tasks
"""

import os
import json
import logging
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List

from core.resources import manager
from modules.translate_video.workflow.task_definitions import (
    TaskDefinition, TaskContext, TaskResult, TaskType, TaskRegistry
)
from modules.translate_video.audio_processing import extract_audio

logger = logging.getLogger("workflow.phase1")

@TaskRegistry.register
class ExtractAudioTask(TaskDefinition):
    name = "extract_audio"
    description = "Extract audio from video file"
    phase = "identifying"
    task_type = TaskType.CPU_BOUND
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        video_path = context.inputs.get('video_path') or context.project_state.get('video_path')
        if not video_path:
            return TaskResult.failure_result(ValueError("No video_path provided"))
            
        output_path = context.work_dir / f"{context.task_id}_master.wav"
        
        await context.log(f"Extracting audio from {Path(video_path).name}...", "step")
        
        try:
            # Use shared helper
            extract_audio(str(video_path), str(output_path))
            return TaskResult.success_result(master_audio_path=str(output_path))
        except Exception as e:
            return TaskResult.failure_result(e)

@TaskRegistry.register
class RunVADTask(TaskDefinition):
    name = "run_vad"
    description = "Voice Activity Detection"
    phase = "identifying"
    depends_on = ["extract_audio"]
    task_type = TaskType.CPU_BOUND # Silero is fast enough on CPU
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        audio_path = context.get_previous_output("extract_audio", "master_audio_path")
        if not audio_path or not Path(audio_path).exists():
            return TaskResult.failure_result(FileNotFoundError("Master audio not found"))
            
        await context.log("Detecting speech segments...", "step")
        
        try:
            # Load Silero VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
            
            wav = read_audio(audio_path)
            speech_timestamps = get_speech_timestamps(
                wav, model, sampling_rate=16000, 
                threshold=0.5, 
                min_speech_duration_ms=250
            )
            
            segments = [
                {'start': ts['start'] / 16000, 'end': ts['end'] / 16000}
                for ts in speech_timestamps
            ]
            
            await context.log(f"Found {len(segments)} speech segments", "success")
            
            return TaskResult.success_result(speech_segments=segments)
            
        except Exception as e:
            return TaskResult.failure_result(e)

@TaskRegistry.register
class ExtractEmbeddingsTask(TaskDefinition):
    name = "extract_embeddings"
    description = "Extract speaker embeddings"
    phase = "identifying"
    depends_on = ["extract_audio", "run_vad"]
    task_type = TaskType.GPU_BOUND
    required_vram_gb = 1.0
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        audio_path = context.get_previous_output("extract_audio", "master_audio_path")
        segments = context.get_previous_output("run_vad", "speech_segments")
        
        if not segments:
            return TaskResult.success_result(embeddings=[], segments=[])
            
        await context.log(f"Extracting embeddings for {len(segments)} segments...", "step")
        
        # Load audio once
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1: audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
        # Get model
        feature_extractor, model = manager.get_speaker_encoder()
        device = manager.device
        
        embeddings = []
        valid_segments = []
        
        for i, seg in enumerate(segments):
            start = int(seg['start'] * 16000)
            end = int(seg['end'] * 16000)
            
            if end - start < 1600: # Skip extremely short segments (<0.1s)
                continue
                
            chunk = audio[start:end]
            
            try:
                inputs = feature_extractor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                
                with torch.no_grad():
                    outputs = model(input_values)
                    emb = outputs.embeddings
                    emb = F.normalize(emb, p=2, dim=1)
                    
                embeddings.append(emb.cpu().numpy()[0].tolist())
                valid_segments.append({**seg, 'idx': i})
                
                if i % 10 == 0:
                    await context.report_progress(int(i / len(segments) * 100), f"Processed {i}/{len(segments)}")
                    
            except Exception as e:
                logger.warning(f"Embedding failed for segment {i}: {e}")
        
        return TaskResult.success_result(embeddings=embeddings, segments=valid_segments)

@TaskRegistry.register
class ClusterSpeakersTask(TaskDefinition):
    name = "cluster_speakers"
    description = "Cluster embeddings to identify speakers"
    phase = "identifying"
    depends_on = ["extract_embeddings"]
    task_type = TaskType.CPU_BOUND
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        embeddings = context.get_previous_output("extract_embeddings", "embeddings")
        segments = context.get_previous_output("extract_embeddings", "segments")
        
        if not embeddings:
            return TaskResult.success_result(assignments={})
            
        await context.log("Clustering speakers...", "step")
        
        X = np.array(embeddings)
        n_samples = len(X)
        
        # Simple Clustering Logic
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Edge case: only 1 sample
        if n_samples == 1:
            labels = [0]
            best_n = 1
        else:
            # Determine best number of clusters
            # CRITICAL FIX: max_k must be at most n_samples - 1 for silhouette_score
            # (silhouette needs 2 to n_samples - 1 inclusive)
            max_k = min(n_samples - 1, 10) if n_samples > 2 else 2
            min_k = 1 if n_samples == 1 else 2
            
            best_n = 1 if n_samples == 1 else 2
            best_score = -1
            labels = [0] * n_samples
            
            if n_samples >= 2:
                for k in range(min_k, max_k + 1):
                    try:
                        # For k=1, silhouette doesn't work, use a default score
                        if k == 1:
                            clustering = AgglomerativeClustering(n_clusters=1, metric='cosine', linkage='average')
                            lbls = clustering.fit_predict(X)
                            score = -0.5  # Default low score for single cluster
                        else:
                            clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
                            lbls = clustering.fit_predict(X)
                            
                            # Only calculate silhouette if we have valid range
                            n_labels = len(set(lbls))
                            if 1 < n_labels < n_samples:  # silhouette_score requires 1 < n_labels < n_samples
                                score = silhouette_score(X, lbls, metric='cosine')
                            elif n_labels == 1:
                                score = -0.5  # Single cluster penalty
                            else:
                                score = -1  # Invalid configuration
                        
                        if score > best_score:
                            best_score = score
                            best_n = k
                            labels = lbls
                            
                    except Exception as e:
                        logger.debug(f"Clustering with {k} speakers failed: {e}")
        
        # Assign speakers to segments
        assignments = {}
        for i, label in enumerate(labels):
            seg_idx = segments[i]['idx']
            assignments[str(seg_idx)] = int(label)
            segments[i]['speaker_id'] = int(label)
            
        await context.log(f"Identified {best_n} unique speakers", "success")
        
        return TaskResult.success_result(
            assignments=assignments,
            speaker_count=best_n,
            segments=segments # Updated segments with speaker_id
        )

@TaskRegistry.register
class ExtractSpeakerSamplesTask(TaskDefinition):
    name = "extract_samples"
    description = "Generate reference audio for each speaker"
    phase = "identifying"
    depends_on = ["cluster_speakers", "extract_audio"]
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        segments = context.get_previous_output("cluster_speakers", "segments")
        audio_path = context.get_previous_output("extract_audio", "master_audio_path")
        
        await context.log("Extracting speaker samples...", "step")
        
        # Group by speaker
        speakers = {}
        for seg in segments:
            sid = seg['speaker_id']
            if sid not in speakers: speakers[sid] = []
            speakers[sid].append(seg)
            
        audio, sr = sf.read(audio_path)
        samples_dir = context.work_dir / "speaker_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        speaker_config = {}
        
        for sid, segs in speakers.items():
            # Find longest segment
            best_seg = max(segs, key=lambda s: s['end'] - s['start'])
            start = int(best_seg['start'] * sr)
            end = int(best_seg['end'] * sr)
            
            sample_path = samples_dir / f"speaker_{sid}_sample.wav"
            sf.write(str(sample_path), audio[start:end], sr)
            
            # Create URL relative to root
            rel_path = f"/temp_chunks/{context.task_id}/speaker_samples/{sample_path.name}"
            
            speaker_config[str(sid)] = {
                "name": f"Speaker {sid + 1}",
                "action": "dub",
                "sample_path": rel_path
            }
            
        return TaskResult.success_result(speaker_config=speaker_config)

@TaskRegistry.register
class TranscribeSegmentsTask(TaskDefinition):
    name = "transcribe_segments"
    description = "Transcribe speech for preview"
    phase = "identifying"
    depends_on = ["cluster_speakers", "extract_audio"]
    task_type = TaskType.GPU_BOUND
    required_vram_gb = 3.0
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        segments = context.get_previous_output("cluster_speakers", "segments")
        audio_path = context.get_previous_output("extract_audio", "master_audio_path")
        
        await context.log(f"Transcribing {len(segments)} segments...", "step")
        
        transcribed_segments = []
        failed_count = 0
        
        # Try the robust direct Whisper approach first (no torchcodec dependency)
        try:
            transcribed_segments = await cls._transcribe_with_direct_whisper(
                segments, audio_path, context
            )
        except Exception as e:
            logger.warning(f"Direct Whisper transcription failed: {e}, trying pipeline fallback...")
            # Fallback to pipeline method if available
            try:
                transcribed_segments = await cls._transcribe_with_pipeline(
                    segments, audio_path, context
                )
            except Exception as e2:
                logger.error(f"All transcription methods failed: {e2}")
                # Return failure - don't silently succeed with empty segments
                return TaskResult.failure_result(
                    RuntimeError(f"Transcription failed completely. "
                               f"Direct method: {e}. Pipeline method: {e2}. "
                               f"Please check FFmpeg installation.")
                )
        
        # Check if we got any successful transcriptions
        successful = [s for s in transcribed_segments if s.get('original_text') and 
                     not s.get('original_text', '').startswith('[')]
        
        if len(successful) == 0 and len(segments) > 0:
            return TaskResult.failure_result(
                RuntimeError("All segments failed to transcribe. "
                           "This is likely due to torchcodec/FFmpeg issues on Windows. "
                           "Please ensure FFmpeg is installed with shared libraries.")
            )
        
        if failed_count > 0:
            await context.log(f"Warning: {failed_count}/{len(segments)} segments failed to transcribe", "warning")
        
        # Update final task state with everything needed for validation
        from core.database import db
        speaker_config = context.get_previous_output("extract_samples", "speaker_config")
        
        db.update_task(
            context.task_id,
            segments=segments,
            transcribed_segments=transcribed_segments,
            speaker_config=speaker_config,
            phase="awaiting_validation",
            status="awaiting_validation",
            message="Identification complete. Please validate speakers."
        )
        
        return TaskResult.success_result(transcribed_segments=transcribed_segments)
    
    @classmethod
    async def _transcribe_with_direct_whisper(cls, segments, audio_path, context):
        """
        Transcribe using direct Whisper model inference.
        This avoids torchcodec dependency which fails on Windows without proper FFmpeg DLLs.
        """
        from transformers import AutoProcessor
        import torch
        
        # Load Whisper components directly
        whisper_pipe = manager.get_whisper()
        model = whisper_pipe.model
        processor = whisper_pipe.tokenizer
        
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
        
        transcribed_segments = []
        total = len(segments)
        
        for i, seg in enumerate(segments):
            try:
                # Extract audio segment
                start_sample = int(seg['start'] * sr)
                end_sample = int(seg['end'] * sr)
                chunk = audio_np[start_sample:end_sample]
                
                # Ensure mono and correct shape
                if len(chunk.shape) > 1:
                    chunk = chunk.mean(axis=1)
                
                # Resample to 16kHz if needed
                if sr != 16000:
                    import librosa
                    chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
                
                # Convert to float32
                chunk = chunk.astype(np.float32)
                
                # Process with feature extractor
                inputs = feature_extractor(
                    chunk, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                )
                
                # Match model dtype
                input_features = inputs.input_features.to(device).to(model_dtype)
                
                # Generate
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
                
                transcribed_segments.append({
                    **seg,
                    "original_text": text,
                    "translated_text": "",
                    "status": "transcribed"
                })
                
                # Progress update
                if i % 5 == 0 or i == total - 1:
                    progress_pct = int((i + 1) / total * 100)
                    await context.report_progress(progress_pct, f"Transcribed {i+1}/{total}")
                    await context.log(f"Segment {seg.get('idx', i)}: '{text[:50]}...'")
                
            except Exception as e:
                logger.error(f"Direct Whisper failed for segment {i}: {e}")
                # Still add the segment but with error marker
                transcribed_segments.append({
                    **seg,
                    "original_text": f"[Transcription error: {str(e)[:50]}]",
                    "translated_text": "",
                    "status": "error",
                    "error": str(e)
                })
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return transcribed_segments
    
    @classmethod
    async def _transcribe_with_pipeline(cls, segments, audio_path, context):
        """
        Fallback: Use the transformers pipeline (may use torchcodec internally).
        """
        stt = manager.get_whisper()
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1: audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
        transcribed_segments = []
        
        for i, seg in enumerate(segments):
            try:
                start = int(seg['start'] * 16000)
                end = int(seg['end'] * 16000)
                chunk = audio[start:end]
                
                # Try with dict format to avoid torchcodec path issues
                res = stt({"array": chunk, "sampling_rate": 16000})
                text = res.get("text", "").strip()
                
                transcribed_segments.append({
                    **seg,
                    "original_text": text,
                    "translated_text": ""
                })
                
                if i % 5 == 0:
                    await context.report_progress(int(i/len(segments)*100), f"Transcribing {i+1}/{len(segments)}")
                    
            except Exception as e:
                logger.error(f"Pipeline transcription error: {e}")
                # Re-raise to trigger fallback or failure
                raise
