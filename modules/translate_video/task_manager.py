import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from core.database import db
from .project_manager import append_log
from .state import broadcast_to_task

# Import Pipeline Phases Directly
from .pipeline.phase1_diarization import run_diarization_phase, run_transcription_phase
from .pipeline.phase2_translation import run_phase2
from .pipeline.phase3_recomposition import run_phase3
from .state import broadcast_to_task


class TaskManager:
    """Manages video translation pipeline execution."""
    
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def start_task(self, task_id: str, is_resume: bool = False):
        """Start or resume a task pipeline."""
        async with self._lock:
            # Check if task is already running
            if task_id in self.active_tasks:
                existing_task = self.active_tasks[task_id]
                if not existing_task.done():
                    print(f"Task {task_id} is already running and active, skipping start request")
                    return
                else:
                    # Clean up completed but un-removed task reference
                    print(f"Cleaning up stale task reference for {task_id}")
                    self.active_tasks.pop(task_id, None)
            
            task = db.get_task(task_id)
            if not task:
                print(f"Task {task_id} not found")
                return
            
            phase = task.get('phase', 'init')
            status = task.get('status', 'pending')
            
            # CRITICAL: Don't start if we're waiting for user input
            interaction_phases = [
                'awaiting_speaker_validation', 
                'awaiting_transcription_review', 
                'awaiting_translation_review', 
                'awaiting_audio_validation'
            ]
            if phase in interaction_phases or status == 'awaiting_validation':
                print(f"Task {task_id} is in '{phase}' state, waiting for user input - not starting pipeline")
                return
            
            # Don't restart completed tasks unless explicitly resuming
            if status == 'completed' and not is_resume:
                print(f"Task {task_id} already completed, not restarting")
                return
            
            print(f"Starting task {task_id} with phase={phase}, status={status}, is_resume={is_resume}")
            
            # Create pipeline task
            pipeline_task = asyncio.create_task(
                self._run_pipeline(task_id, task, is_resume)
            )
            self.active_tasks[task_id] = pipeline_task
            
            pipeline_task.add_done_callback(
                lambda t: asyncio.create_task(self._cleanup_task(task_id))
            )
            
            db.update_task(task_id, was_running_at_shutdown=1)

    async def _run_pipeline(self, task_id: str, task: Dict[str, Any], is_resume: bool):
        """Run the appropriate pipeline phase based on current state."""
        # Import broadcast_to_task at function level to ensure it's available
        from .state import broadcast_to_task
        
        try:
            phase = task.get('phase', 'init')
            video_path = task.get('video_path')
            
            if not video_path:
                raise ValueError("No video_path in task")
            
            # Progress reporter - defined after imports to capture in closure
            async def report(phase_name: str, percent: int, message: str):
                db.update_task(task_id, phase=phase_name, progress=percent, message=message)
                await broadcast_to_task(task_id, {
                    'type': 'progress',
                    'data': {'phase': phase_name, 'percent': percent, 'message': message}
                })
            
            # Waiting states - do not re-run, just return
            interaction_phases = [
                'awaiting_speaker_validation', 
                'awaiting_transcription_review', 
                'awaiting_translation_review', 
                'awaiting_audio_validation'
            ]
            if phase in interaction_phases:
                await report(phase, task.get('progress', 35), "Waiting for user review...")
                return  # CRITICAL: Early return to prevent re-running
            
            # Import new granular phases
            from .pipeline.phase2_translation import TranslationEngine, TTSEngine, TranslationSegment
            import functools

            # ----------------------------------------------------------------
            # PHASE 1: DIARIZATION (Identification)
            # ----------------------------------------------------------------
            if phase in ['init', 'identifying']:
                await report("identifying", 5, "Starting speaker identification...")
                
                # This stops at 'awaiting_speaker_validation'
                await run_diarization_phase(
                    task_id=task_id,
                    video_path=video_path,
                    progress_callback=report
                )
                return

            # ----------------------------------------------------------------
            # PHASE 2: TRANSCRIPTION (After Speaker Validation)
            # ----------------------------------------------------------------
            if phase == 'transcribing':
                await report("transcribing", 25, "Starting transcription...")
                
                # This stops at 'awaiting_transcription_review'
                await run_transcription_phase(
                    task_id=task_id,
                    source_language=task.get('src_lang', 'auto'),
                    progress_callback=report
                )
                return

            # ----------------------------------------------------------------
            # PHASE 3: TRANSLATION (After Transcription Review)
            # ----------------------------------------------------------------
            if phase == 'translating':
                # Check if we already have translated segments (manual validation just happened)
                # If they exist, skip the LLM translation and go straight to synthesis.
                current_segments = task.get('segments', [])
                if current_segments and all(s.get('translated_text') for s in current_segments):
                    print(f"Task {task_id} already translated, moving to synthesis")
                    db.update_task(task_id, phase='synthesizing')
                    phase = 'synthesizing' # Force local update for immediate execution
                else:
                    await report("translating", 45, "Starting translation...")
                    
                    # Get segments
                    seg_data = task.get('transcribed_segments', []) or task.get('segments', [])
                if not seg_data: raise ValueError("No segments for translation")
                segments = [TranslationSegment.from_dict(s) for s in seg_data]
                
                # Run translation
                translator = TranslationEngine(
                    target_language=task.get('tgt_lang', 'en'),
                    source_language=task.get('src_lang', 'auto')
                )
                
                loop = asyncio.get_running_loop()
                def trans_progress(current, total):
                    pct = 45 + int((current / total) * 15)
                    asyncio.run_coroutine_threadsafe(
                        report("translating", pct, f"Translated {current}/{total} segments"), loop
                    )
                
                segments = await loop.run_in_executor(
                    None, functools.partial(translator.translate_batch, segments, progress_callback=trans_progress)
                )
                
                # Stop for Translation Review
                db.update_task(
                    task_id,
                    segments=[s.to_dict() for s in segments],
                    phase='awaiting_translation_review',
                    status='awaiting_validation',
                    progress=60,
                    message="Translation complete - review and confirm"
                )
                
                await broadcast_to_task(task_id, {
                    'type': 'translation_ready',
                    'data': {
                        'segments': [s.to_dict() for s in segments],
                        'target_language': task.get('tgt_lang', 'en'),
                        'source_language': task.get('src_lang', 'auto'),
                        'can_edit': True
                    }
                })
                return

            # ----------------------------------------------------------------
            # PHASE 4: TTS SYNTHESIS (After Translation Review)
            # ----------------------------------------------------------------
            if phase == 'synthesizing':
                speaker_config = task.get('speaker_config')
                if not speaker_config: raise ValueError("No speaker_config")
                
                seg_data = task.get('segments', [])
                if not seg_data: raise ValueError("No segments for TTS")
                
                await report("synthesizing", 65, "Starting voice synthesis...")
                
                output_dir = Path("temp_chunks") / task_id / "synthesized"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                segments = [TranslationSegment.from_dict(s) for s in seg_data]
                tts = TTSEngine(
                    task.get('tts_engine', 'f5'), 
                    speaker_config, 
                    target_language=task.get('tgt_lang', 'en')
                )
                
                batch_size = 5
                total = len(segments)
                loop = asyncio.get_running_loop()
                
                def threadsafe_report(phase_str, pct, msg):
                    asyncio.run_coroutine_threadsafe(report(phase_str, pct, msg), loop)
                
                for i in range(0, total, batch_size):
                    batch = segments[i:i + batch_size]
                    
                    def synth_progress(current, total_in_batch):
                        overall = i + current
                        pct = 65 + int((overall / total) * 20)
                        threadsafe_report("synthesizing", pct, f"Synthesized {overall}/{total} segments")
                    
                    await loop.run_in_executor(
                        None, functools.partial(tts.synthesize_batch, batch, output_dir, progress_callback=synth_progress)
                    )
                    # Save incremental progress
                    db.update_task(task_id, segments=[s.to_dict() for s in segments])
                
                # Stop for Audio Review (NEW STEP)
                db.update_task(
                    task_id,
                    segments=[s.to_dict() for s in segments],
                    translation_segments=[s.to_dict() for s in segments],
                    phase="awaiting_audio_validation",
                    status="awaiting_validation",
                    progress=85,
                    message="TTS complete - Listen to generated audio before final mix"
                )
                
                await broadcast_to_task(task_id, {
                    'type': 'audio_ready',
                    'data': {
                        'segments': [s.to_dict() for s in segments]
                    }
                })
                return

            # ----------------------------------------------------------------
            # PHASE 5: RECOMPOSITION (Final Assembly)
            # ----------------------------------------------------------------
            if phase == 'recomposing':
                await report("recomposing", 90, "Starting final assembly...")
                
                result = await run_phase3(
                    task_id=task_id,
                    original_video_path=video_path,
                    use_demucs=task.get('separate_audio', False),
                    progress_callback=report
                )
                return
            
            # Phase 3: Final Assembly
            if phase in ['recomposing']:
                await report("recomposing", 82, "Starting final assembly...")
                
                result = await run_phase3(
                    task_id=task_id,
                    original_video_path=video_path,
                    use_demucs=task.get('separate_audio', False),
                    progress_callback=report
                )
                
                # Complete!
                return
            
            # Validation states - waiting for user
            if phase in ['awaiting_validation', 'awaiting_translation_review']:
                # Nothing to do, waiting for user input
                return
                
        except asyncio.CancelledError:
            db.update_task(task_id, status='cancelled', was_running_at_shutdown=0)
            raise
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Pipeline failed: {e}\n{tb_str}")
            db.update_task(task_id, status='failed', error_message=str(e), error_traceback=tb_str)
            await broadcast_to_task(task_id, {'type': 'error', 'message': str(e)})

    async def _cleanup_task(self, task_id: str):
        async with self._lock:
            self.active_tasks.pop(task_id, None)

    async def cancel_task(self, task_id: str, force: bool = False):
        """Cancel a running task."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            try:
                # Give it a moment to cancel gracefully
                await asyncio.wait_for(self.active_tasks[task_id], timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # Expected
        db.update_task(task_id, status='cancelled', was_running_at_shutdown=0)

    async def pause_task(self, task_id: str):
        await self.cancel_task(task_id)
        db.update_task(task_id, status='paused')

    async def resume_task(self, task_id: str):
        task = db.get_task(task_id)
        if not task: return
        
        phase = task.get('phase', 'init')
        interaction_phases = [
            'init', 
            'awaiting_speaker_validation', 
            'awaiting_transcription_review', 
            'awaiting_translation_review', 
            'awaiting_audio_validation', 
            'complete'
        ]
        
        if phase in interaction_phases:
            # It's waiting for user input, simply update status, don't restart pipeline
            new_status = 'completed' if phase == 'complete' else 'awaiting_validation'
            db.update_task(task_id, status=new_status, was_running_at_shutdown=0)
            
            # Broadcast state sync to wake up UI
            from .state import broadcast_to_task
            await broadcast_to_task(task_id, {'type': 'state_sync', 'data': db.get_task(task_id)})
            return
            
        db.update_task(task_id, status='queued', was_running_at_shutdown=1)
        await self.start_task(task_id, is_resume=True)

    async def restart_task(self, task_id: str, from_phase: Optional[str] = None):
        """Stop current task and jump to a specific phase."""
        # 1. Kill any active worker first
        await self.cancel_task(task_id)
        # Short delay to allow OS/Thread handles to release
        await asyncio.sleep(0.3)
        
        new_phase = from_phase or 'init'
        
        # 2. Determine if target is an interaction step or computation step
        interaction_phases = [
            'init', 
            'awaiting_speaker_validation', 
            'awaiting_transcription_review', 
            'awaiting_translation_review', 
            'awaiting_audio_validation', 
            'complete'
        ]
        
        new_status = 'awaiting_validation' if new_phase in interaction_phases else 'queued'
        
        # 3. Reset progress and update state
        db.update_task(
            task_id, 
            phase=new_phase, 
            status=new_status, 
            progress=0, 
            was_running_at_shutdown=0,
            message=f"Jumped to {new_phase}"
        )
        
        # 4. If target is computation, start the worker
        if new_status == 'queued':
            await self.start_task(task_id)
        else:
            # Broadcast state sync so UI updates immediately
            from .state import broadcast_to_task
            task_data = db.get_task(task_id)
            await broadcast_to_task(task_id, {'type': 'state_sync', 'data': task_data})

    async def recover_interrupted_tasks(self):
        tasks = db.get_interrupted_tasks()
        for t in tasks:
            print(f"Recovering task {t['task_id']}")
            await self.start_task(t['task_id'], is_resume=True)

    async def shutdown(self):
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)

task_manager = TaskManager()
