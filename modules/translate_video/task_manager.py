import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from core.database import db
from .project_manager import append_log
from .state import broadcast_to_task

# Import Pipeline Phases Directly
from .pipeline.phase1_diarization import run_phase1
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
            if task_id in self.active_tasks:
                print(f"Task {task_id} already running, skipping")
                return
            
            task = db.get_task(task_id)
            if not task:
                print(f"Task {task_id} not found")
                return
            
            phase = task.get('phase', 'init')
            status = task.get('status', 'pending')
            
            # CRITICAL: Don't start if we're waiting for user input
            if phase in ['awaiting_validation', 'awaiting_translation_review']:
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
            if phase in ['awaiting_validation', 'awaiting_translation_review']:
                # Task is waiting for user input via WebSocket
                await report(phase, task.get('progress', 35), 
                    "Waiting for user review..." if phase == 'awaiting_validation' else "Waiting for translation review...")
                return  # CRITICAL: Early return to prevent re-running
            
            # Phase 1: Speaker Identification + Transcription
            if phase in ['init', 'identifying']:
                await report("identifying", 5, "Starting speaker identification...")
                
                result = await run_phase1(
                    task_id=task_id,
                    video_path=video_path,
                    source_language=task.get('src_lang', 'auto'),
                    progress_callback=report
                )
                
                # Phase 1 ends at 'awaiting_validation' - user reviews transcriptions
                return
            
            # Phase 1.5: Translation Review (after transcription validation)
            # This phase runs the actual translation and sends results to UI
            if phase == 'awaiting_translation_review':
                # This shouldn't auto-run - we need user to trigger translation
                # But if we have edited_transcriptions, we should process them
                if task.get('edited_transcriptions'):
                    await report("translating", 40, "Processing edited transcriptions...")
                    # Continue to translation phase below
                else:
                    # Just waiting for user to trigger translation
                    await report("awaiting_translation_review", 40, "Waiting for user to start translation...")
                    return
            
            # Phase 1.5: Actually run translation (triggered by user)
            if phase == 'running_translation':
                await report("translating", 40, "Starting translation...")
                
                # Import here to avoid circular dependencies
                from .pipeline.phase2_translation import TranslationEngine
                
                # Get segments
                seg_data = task.get('transcribed_segments', []) or task.get('segments', [])
                if not seg_data:
                    raise ValueError("No transcribed segments found")
                
                from .pipeline.phase2_translation import TranslationSegment
                segments = [TranslationSegment.from_dict(s) for s in seg_data]
                
                # Run translation
                translator = TranslationEngine(
                    target_language=task.get('tgt_lang', 'en'),
                    source_language=task.get('src_lang', 'auto')
                )
                
                def trans_progress(current, total):
                    pct = 40 + int((current / total) * 20)
                    asyncio.create_task(report("translating", pct, 
                        f"Translated {current}/{total} segments"))
                
                segments = translator.translate_batch(segments, trans_progress)
                
                # Save translated segments
                db.update_task(
                    task_id,
                    segments=[s.to_dict() for s in segments],
                    phase='awaiting_translation_review',  # Go back to review state
                    status='awaiting_validation',  # Need user validation
                    progress=60,
                    message="Translation complete - review and confirm"
                )
                
                # Send to frontend
                from .state import broadcast_to_task
                await broadcast_to_task(task_id, {
                    'type': 'translation_ready',
                    'data': {
                        'segments': [s.to_dict() for s in segments],
                        'target_language': task.get('tgt_lang', 'en'),
                        'source_language': task.get('src_lang', 'auto'),
                        'can_edit': True
                    }
                })
                
                await report("awaiting_translation_review", 60, "Translation complete - awaiting review")
                return
            
            # Phase 2: TTS Synthesis (after translation validation)
            if phase in ['translating', 'tts_synthesis']:
                speaker_config = task.get('speaker_config')
                if not speaker_config:
                    raise ValueError("No speaker_config - validation required")
                
                # Get potentially edited segments
                seg_data = task.get('segments', [])
                if not seg_data:
                    raise ValueError("No segments - run translation review first")
                
                await report("tts_synthesis", 65, "Starting voice synthesis...")
                
                # Run TTS only (translation already done)
                from .pipeline.phase2_translation import TTSEngine, TranslationSegment
                import numpy as np
                
                output_dir = Path("temp_chunks") / task_id / "synthesized"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                segments = [TranslationSegment.from_dict(s) for s in seg_data]
                tts_engine = task.get('tts_engine', 'f5')
                
                tts = TTSEngine(tts_engine, speaker_config)
                
                # Process in batches
                batch_size = 10
                total = len(segments)
                
                for i in range(0, total, batch_size):
                    batch = segments[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (total + batch_size - 1) // batch_size
                    
                    await report("tts_synthesis", 
                        65 + int((i / total) * 15),
                        f"Synthesizing batch {batch_num}/{total_batches}")
                    
                    def synth_progress(current, total_in_batch):
                        overall = i + current
                        pct = 65 + int((overall / total) * 15)
                        asyncio.create_task(report("tts_synthesis", pct,
                            f"Synthesized {overall}/{total} segments"))
                    
                    tts.synthesize_batch(batch, output_dir, synth_progress)
                    
                    # Save progress
                    db.update_task(task_id, segments=[s.to_dict() for s in segments])
                
                await report("tts_synthesis", 80, "Voice synthesis complete")
                
                # Move to Phase 3
                db.update_task(
                    task_id,
                    segments=[s.to_dict() for s in segments],
                    translation_segments=[s.to_dict() for s in segments],
                    phase="recomposing",
                    status="queued",
                    progress=80,
                    message="TTS complete - starting final assembly..."
                )
                
                # Continue to Phase 3
                await self._run_pipeline(task_id, db.get_task(task_id), is_resume)
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
        db.update_task(task_id, status='queued', was_running_at_shutdown=1)
        await self.start_task(task_id, is_resume=True)

    async def restart_task(self, task_id: str, from_phase: Optional[str] = None):
        await self.cancel_task(task_id)
        await asyncio.sleep(0.5)
        
        new_phase = from_phase or 'init'
        db.update_task(task_id, phase=new_phase, status='queued', progress=0)
        await self.start_task(task_id)

    async def recover_interrupted_tasks(self):
        tasks = db.get_interrupted_tasks()
        for t in tasks:
            print(f"Recovering task {t['task_id']}")
            await self.start_task(t['task_id'], is_resume=True)

    async def shutdown(self):
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)

task_manager = TaskManager()
