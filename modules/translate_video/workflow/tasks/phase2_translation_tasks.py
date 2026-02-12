"""
Phase 2: Translation & Synthesis Tasks
"""

import asyncio
from modules.translate_video.workflow.task_definitions import (
    TaskDefinition, TaskContext, TaskResult, TaskType, TaskRegistry
)
from core.resources import manager
from core.database import db
from modules.translate_video.pipeline.phase2_subphase2_translation import TranslationSubphase
from modules.translate_video.pipeline.phase2_subphase3_tts import TTSSubphase
from modules.translate_video.pipeline.phase2_models import TranslationSegment

@TaskRegistry.register
class TranslateSegmentTask(TaskDefinition):
    name = "translate_text"
    description = "Translate text using LLM"
    phase = "translating"
    task_type = TaskType.IO_BOUND # LLM might be remote or local
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        # Reconstruct segments from DB
        task = db.get_task(context.task_id)
        raw_segments = task.get('transcribed_segments', [])
        
        if not raw_segments:
            # Check if we have segments in the task at all
            task_segments = task.get('segments', [])
            if task_segments:
                # We have segments but no transcription - this is an error state
                return TaskResult.failure_result(
                    RuntimeError(
                        f"Task has {len(task_segments)} segments but no transcriptions. "
                        "Phase 1 transcription likely failed silently. "
                        "Please restart from 'identifying' phase."
                    )
                )
            return TaskResult.skipped_result("No segments to translate")
        
        # Filter out segments with transcription errors
        valid_segments = [
            s for s in raw_segments 
            if s.get('original_text') and 
            not s.get('original_text', '').startswith('[') and
            not s.get('status') == 'error'
        ]
        
        error_segments = [s for s in raw_segments if s not in valid_segments]
        
        if len(valid_segments) == 0:
            return TaskResult.failure_result(
                RuntimeError(
                    f"All {len(raw_segments)} segments have transcription errors. "
                    "Cannot proceed with translation. "
                    "Please check the logs and restart from Phase 1."
                )
            )
        
        if error_segments:
            await context.log(
                f"Warning: {len(error_segments)}/{len(raw_segments)} segments have transcription errors and will be skipped", 
                "warning"
            )
            
        await context.log(f"Translating {len(valid_segments)} valid segments to {task.get('tgt_lang')}...", "step")
        
        # Convert to TranslationSegment objects
        segments = []
        for s in valid_segments:
            segments.append(TranslationSegment(
                idx=s.get('idx', 0),
                start=s.get('start', 0),
                end=s.get('end', 0),
                speaker_id=s.get('speaker_id', 0),
                speaker_name=s.get('speaker_name', ''),
                original_text=s.get('original_text', '')
            ))
        
        subphase = TranslationSubphase(
            task_id=context.task_id,
            target_language=task.get('tgt_lang', 'en'),
            source_language=task.get('src_lang', 'auto'),
            speaker_config=task.get('speaker_config', {})
        )
        
        # We can implement a bridge callback to report progress via context
        async def progress_bridge(phase, pct, msg):
            await context.report_progress(pct, msg)
            
        subphase.progress_callback = progress_bridge
        
        result_segments = await subphase.run(segments)
        
        # Serialize for next task
        serialized = [s.to_dict() for s in result_segments]
        db.update_task(context.task_id, translation_segments=serialized)
        
        return TaskResult.success_result(translation_segments=serialized)

@TaskRegistry.register
class SynthesizeSegmentTask(TaskDefinition):
    name = "synthesize_speech"
    description = "Generate speech using TTS"
    phase = "translating"
    depends_on = ["translate_text"]
    task_type = TaskType.GPU_BOUND
    required_vram_gb = 2.0
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        # Load from DB or previous output
        task = db.get_task(context.task_id)
        # Check if we have serialized segments in DB or from previous task
        segments_data = context.get_previous_output("translate_text", "translation_segments") or \
                       task.get('translation_segments')
        
        if not segments_data:
            # Check if translate_text actually ran but produced no output
            previous_result = context.previous_results.get("translate_text")
            if previous_result and not previous_result.success:
                # Translation failed - propagate the error
                return TaskResult.failure_result(
                    RuntimeError(f"Previous translation step failed: {previous_result.error_message}")
                )
            
            # Check if there are segments that were skipped or all failed
            raw_segments = task.get('transcribed_segments', [])
            if raw_segments:
                # We had segments but translation produced nothing
                return TaskResult.failure_result(
                    RuntimeError(
                        "No translation segments available for synthesis. "
                        f"Had {len(raw_segments)} transcribed segments but translation produced no output. "
                        "This may indicate all segments had transcription errors."
                    )
                )
            
            return TaskResult.failure_result(ValueError("No segments found for synthesis - no transcribed segments exist"))
        
        # Deserialize
        segments = []
        for s in segments_data:
            ts = TranslationSegment(
                idx=s['idx'], start=s['start'], end=s['end'], 
                speaker_id=s['speaker_id'], original_text=s['original_text'],
                translated_text=s['translated_text']
            )
            # Restore status if already completed
            if s.get('status') == 'completed':
                ts.status = 'completed'
                ts.audio_path = s.get('audio_path')
            segments.append(ts)
            
        await context.log(f"Synthesizing {len(segments)} segments...", "step")
        
        subphase = TTSSubphase(
            task_id=context.task_id,
            tts_engine=task.get('tts_engine', 'f5'),
            speaker_config=task.get('speaker_config', {})
        )
        
        async def progress_bridge(phase, pct, msg):
            await context.report_progress(pct, msg)
            
        subphase.progress_callback = progress_bridge
        
        result_segments = await subphase.run(segments)
        
        return TaskResult.success_result(completed=True)
