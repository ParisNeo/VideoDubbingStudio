"""
Phase 3: Assembly Tasks
"""

from pathlib import Path
from modules.translate_video.workflow.task_definitions import (
    TaskDefinition, TaskContext, TaskResult, TaskType, TaskRegistry
)
from core.database import db
from modules.translate_video.pipeline.phase3_recomposition import Phase3Recomposer

@TaskRegistry.register
class SeparateBackgroundAudioTask(TaskDefinition):
    name = "separate_background"
    description = "Separate background noise using Demucs"
    phase = "recomposing"
    task_type = TaskType.GPU_BOUND
    required_vram_gb = 2.0
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        task = db.get_task(context.task_id)
        if not task.get('separate_audio', False):
            return TaskResult.skipped_result("Background separation disabled")
            
        await context.log("Separating background audio...", "step")
        
        from modules.translate_video.audio_processing import separate_background_foreground
        
        master_audio = task.get('master_audio')
        if not master_audio or not Path(master_audio).exists():
             return TaskResult.skipped_result("Master audio not found for separation")

        output_dir = context.work_dir / "demucs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            paths = separate_background_foreground(master_audio, output_dir)
            return TaskResult.success_result(background_path=paths['background'])
        except Exception as e:
            return TaskResult.failure_result(e)

@TaskRegistry.register
class BuildSpeechTrackTask(TaskDefinition):
    name = "build_speech_track"
    description = "Assemble continuous speech track"
    phase = "recomposing"
    task_type = TaskType.CPU_BOUND
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        task = db.get_task(context.task_id)
        video_path = task.get('video_path')
        
        if not video_path:
            return TaskResult.failure_result(ValueError("Video path not found in task data"))

        recomposer = Phase3Recomposer(
            task_id=context.task_id,
            original_video_path=video_path,
            use_demucs=False # Not needed for this step
        )
        
        await context.log("Assembling speech track...", "step")
        
        try:
            speech_path = await recomposer._build_speech_track()
            return TaskResult.success_result(speech_track_path=speech_path)
        except Exception as e:
            return TaskResult.failure_result(e)

@TaskRegistry.register
class MixAudioTracksTask(TaskDefinition):
    name = "mix_audio_tracks"
    description = "Mix speech with background audio"
    phase = "recomposing"
    task_type = TaskType.CPU_BOUND
    depends_on = ["build_speech_track"]
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        task = db.get_task(context.task_id)
        video_path = task.get('video_path')
        
        # Get inputs
        speech_path = context.get_previous_output("build_speech_track", "speech_track_path")
        bg_path = context.get_previous_output("separate_background", "background_path")
        
        if not speech_path:
            return TaskResult.failure_result(ValueError("Speech track path missing"))

        recomposer = Phase3Recomposer(
            task_id=context.task_id,
            original_video_path=video_path,
            use_demucs=False
        )
        
        # Manually set background path if available
        if bg_path:
            recomposer.background_audio_path = bg_path
            
        await context.log("Mixing audio tracks...", "step")
        
        try:
            final_audio = await recomposer._mix_audio(speech_path)
            return TaskResult.success_result(mixed_audio_path=final_audio)
        except Exception as e:
            return TaskResult.failure_result(e)

@TaskRegistry.register
class MergeVideoTask(TaskDefinition):
    name = "merge_video"
    description = "Merge final audio with video"
    phase = "recomposing"
    task_type = TaskType.CPU_BOUND
    depends_on = ["mix_audio_tracks"]
    
    @classmethod
    async def execute(cls, context: TaskContext) -> TaskResult:
        task = db.get_task(context.task_id)
        video_path = task.get('video_path')
        
        audio_path = context.get_previous_output("mix_audio_tracks", "mixed_audio_path")
        
        if not audio_path:
             return TaskResult.failure_result(ValueError("Mixed audio path missing"))

        recomposer = Phase3Recomposer(
            task_id=context.task_id,
            original_video_path=video_path,
            use_demucs=False
        )
        
        # Progress bridge
        async def progress_bridge(phase, pct, msg):
            await context.report_progress(pct, msg)
        recomposer.progress_callback = progress_bridge
        
        await context.log("Merging audio with video...", "step")
        
        try:
            final_path = await recomposer._merge_with_video(video_path, audio_path, str(recomposer.final_video_path))
            
            # Update main task result
            db.update_task(context.task_id, output_path=final_path)
            
            return TaskResult.success_result(output_path=final_path)
        except Exception as e:
            return TaskResult.failure_result(e)
