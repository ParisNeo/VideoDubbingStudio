import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from core.database import db
from .logic import start_identification_task, start_dubbing_task
from .project_manager import ProjectManager, append_log
from .state import broadcast_to_task

# Import Phase 3
from .pipeline.phase3_recomposition import run_phase3_recomposition

class TaskManager:
    """Manages the video translation pipeline lifecycle with full resumption support."""
    
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def start_task(self, task_id: str, is_resume: bool = False):
        """Start or resume a task pipeline."""
        async with self._lock:
            if task_id in self.active_tasks:
                # Already running
                return
            
            task = db.get_task(task_id)
            if not task:
                print(f"Task {task_id} not found in database")
                return
            
            # Mark as running in database (for crash recovery)
            db.update_task(
                task_id,
                was_running_at_shutdown=1,
                status='processing' if not is_resume else 'resuming'
            )
            
            # Create and track the task
            pipeline_task = asyncio.create_task(self._run_pipeline(task_id, task, is_resume))
            self.active_tasks[task_id] = pipeline_task
            
            # Clean up when done
            pipeline_task.add_done_callback(
                lambda t: asyncio.create_task(self._cleanup_task(task_id))
            )
    
    async def _cleanup_task(self, task_id: str):
        """Remove task from active tracking and clear running flag."""
        async with self._lock:
            self.active_tasks.pop(task_id, None)
        
        # Clear the running flag since we're done
        db.update_task(task_id, was_running_at_shutdown=0)
    
    async def recover_interrupted_tasks(self):
        """
        Called on server startup to resume tasks that were running when
        the server was last shut down (crash recovery).
        """
        interrupted = db.get_interrupted_tasks()
        if not interrupted:
            print("No interrupted tasks to recover")
            return
        
        print(f"Recovering {len(interrupted)} interrupted task(s)...")
        
        for task in interrupted:
            task_id = task['task_id']
            current_attempts = task.get('resume_attempts', 0)
            
            # Limit resume attempts to prevent infinite loops on broken tasks
            if current_attempts >= 3:
                print(f"Task {task_id}: Max resume attempts reached, marking as failed")
                db.update_task(
                    task_id,
                    status='failed',
                    was_running_at_shutdown=0,
                    error_message=f"Max resume attempts ({current_attempts}) exceeded"
                )
                continue
            
            # Increment resume attempts
            db.update_task(task_id, resume_attempts=current_attempts + 1)
            
            # Determine if we can resume from current phase or need to restart phase
            phase = task.get('phase', 'init')
            print(f"Task {task_id}: Attempting to resume from phase '{phase}' (attempt {current_attempts + 1})")
            
            # Start the task with resume flag
            try:
                await self.start_task(task_id, is_resume=True)
            except Exception as e:
                print(f"Failed to resume task {task_id}: {e}")
                db.update_task(
                    task_id,
                    status='failed',
                    was_running_at_shutdown=0,
                    error_message=f"Resume failed: {str(e)}"
                )
    
    async def shutdown(self):
        """
        Graceful shutdown - mark all running tasks but keep them resumable.
        This is called when the server is shutting down cleanly.
        """
        print("TaskManager: Graceful shutdown initiated...")
        
        async with self._lock:
            # Cancel all running tasks gracefully
            cancel_tasks = []
            for task_id, task in list(self.active_tasks.items()):
                if not task.done():
                    task.cancel()
                    cancel_tasks.append(task_id)
            
            # Wait for cancellations with timeout
            if cancel_tasks:
                print(f"Cancelling {len(cancel_tasks)} active task(s)...")
                # Note: tasks are already cancelled, we just need to let them finish cleanup
                # The was_running_at_shutdown flag remains True so they resume on restart
        
        print("TaskManager: Shutdown complete")
    
    async def cancel_task(self, task_id: str):
        """Cancel a running task permanently."""
        async with self._lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].cancel()
                db.update_task(
                    task_id,
                    status='cancelled',
                    was_running_at_shutdown=0
                )
                await broadcast_to_task(task_id, {
                    'type': 'status_update',
                    'data': {'status': 'cancelled', 'was_running_at_shutdown': False}
                })
    
    async def pause_task(self, task_id: str):
        """Pause a running task (cooperative)."""
        db.update_task(task_id, status='paused', was_running_at_shutdown=0)
        await broadcast_to_task(task_id, {
            'type': 'status_update',
            'data': {'status': 'paused', 'was_running_at_shutdown': False}
        })
    
    async def resume_task(self, task_id: str):
        """Resume a paused or interrupted task."""
        task = db.get_task(task_id)
        if not task:
            return
        
        current_status = task.get('status')
        if current_status not in ['paused', 'failed', 'error', 'resuming']:
            # Can also resume from processing if we think it's stuck
            if current_status != 'processing':
                print(f"Task {task_id}: Cannot resume from status '{current_status}'")
                return
        
        # Reset status and start
        db.update_task(task_id, status='queued', was_running_at_shutdown=1, error_message=None)
        await self.start_task(task_id, is_resume=True)
    
    async def restart_task(self, task_id: str, from_phase: Optional[str] = None):
        """
        Restart a task from a specific phase or from the beginning.
        This is useful for retrying failed tasks or changing parameters.
        """
        task = db.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Cancel if currently running
        async with self._lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].cancel()
                # Wait a moment for cleanup
                await asyncio.sleep(0.5)
        
        # Determine restart phase
        restart_phase = from_phase or 'init'
        
        # Reset task state but keep uploaded file info
        db.update_task(
            task_id,
            status='queued',
            phase=restart_phase,
            progress=0 if restart_phase == 'init' else task.get('progress', 0),
            was_running_at_shutdown=1,
            error_message=None,
            resume_attempts=0,  # Reset resume attempts on manual restart
            output_path=None,
            # Keep: video_path, original_filename, input_filename, source, tgt_lang, speaker_config
        )
        
        # Clear translation segments if restarting from init
        if restart_phase == 'init':
            # Note: We keep the database entries but they'll be overwritten
            pass
        
        # Reload task and start fresh
        fresh_task = db.get_task(task_id)
        await self.start_task(task_id, is_resume=False)
        
        return fresh_task
    
    async def _run_pipeline(self, task_id: str, task: Dict[str, Any], is_resume: bool = False):
        """Main pipeline execution loop with resumption support."""
        try:
            phase = task.get('phase', 'init')
            status = task.get('status', 'queued')
            
            # Log resume status
            if is_resume:
                await self._broadcast_log(task_id, f"Resuming task from phase: {phase}", "step")
            
            # Determine which phase to run based on current state
            if phase == 'init' or phase == 'identifying':
                await self._phase1_diarization(task_id, task, is_resume)
            elif phase == 'awaiting_validation':
                # Waiting for user input, nothing to do
                await self._broadcast_log(task_id, "Waiting for speaker validation...", "info")
            elif phase == 'translating':
                await self._phase2_translation(task_id, task, is_resume)
            elif phase == 'recomposing':
                await self._phase3_recomposition(task_id, task, is_resume)
            elif phase == 'complete':
                # Already done
                await broadcast_to_task(task_id, {
                    'type': 'status_update',
                    'data': {
                        'status': 'completed',
                        'phase': 'complete',
                        'progress': 100,
                        'was_running_at_shutdown': False
                    }
                })
                
        except asyncio.CancelledError:
            # Task was cancelled - don't mark as failed
            db.update_task(task_id, status='cancelled', was_running_at_shutdown=0)
            await broadcast_to_task(task_id, {
                'type': 'status_update',
                'data': {
                    'status': 'cancelled',
                    'was_running_at_shutdown': False
                }
            })
            raise
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Pipeline failed: {e}"
            append_log(task_id, error_msg, "error")
            db.update_task(
                task_id,
                status='failed',
                was_running_at_shutdown=0,
                error_message=str(e)
            )
            await broadcast_to_task(task_id, {
                'type': 'status_update',
                'data': {
                    'status': 'failed',
                    'error_message': str(e),
                    'was_running_at_shutdown': False
                }
            })
    
    async def _phase1_diarization(self, task_id: str, task: Dict[str, Any], is_resume: bool = False):
        """Phase 1: Speaker identification and diarization."""
        # Check if we can skip this phase (already have results)
        if is_resume and task.get('segments') and task.get('assignments'):
            # We already have diarization results, skip to validation
            await self._broadcast_log(task_id, "Diarization already completed, moving to validation", "success")
            db.update_task(
                task_id,
                phase='awaiting_validation',
                status='awaiting_validation',
                progress=30,
                was_running_at_shutdown=0  # Safe to pause here
            )
            await broadcast_to_task(task_id, {
                'type': 'status_update',
                'data': {
                    'status': 'awaiting_validation',
                    'phase': 'awaiting_validation',
                    'progress': 30,
                    'speaker_config': task.get('speaker_config', {}),
                    'was_running_at_shutdown': False
                }
            })
            return
        
        # Look for video path in multiple possible keys for compatibility
        video_path = task.get('video_path') or task.get('file_path') or task.get('video_file_path')
        
        if not video_path:
            # Check if there's an input_file or original_filename we can construct from
            input_file = task.get('input_filename') or task.get('filename') or task.get('original_filename')
            if input_file:
                # Try to construct path from common locations
                potential_paths = [
                    Path("uploads") / input_file,
                    Path("uploads") / f"{task_id}_{input_file}",
                ]
                for p in potential_paths:
                    if p.exists():
                        video_path = str(p)
                        break
        
        if not video_path:
            raise ValueError(f"No video path found in task. Available keys: {list(task.keys())}")
        
        # Ensure video_path is a string and exists
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise ValueError(f"Video file not found: {video_path}")
        
        # Update task with the resolved path for future reference
        db.update_task(
            task_id,
            video_path=video_path,
            status='processing',
            phase='identifying',
            progress=5,
            message='Starting speaker identification...',
            was_running_at_shutdown=1  # Mark as running for crash recovery
        )
        
        await broadcast_to_task(task_id, {
            'type': 'status_update',
            'data': {
                'status': 'processing',
                'phase': 'identifying',
                'progress': 5,
                'message': 'Starting speaker identification...',
                'was_running_at_shutdown': True
            }
        })
        
        # Call the logic function from logic.py
        await start_identification_task(task_id, video_path)
    
    async def _phase2_translation(self, task_id: str, task: Dict[str, Any], is_resume: bool = False):
        """Phase 2: Translation and TTS generation."""
        # Check if we have checkpoint data to resume from
        checkpoint = task.get('checkpoint_data', {})
        last_completed_idx = checkpoint.get('last_completed_idx', -1) if checkpoint else -1
        
        if is_resume and last_completed_idx >= 0:
            await self._broadcast_log(
                task_id, 
                f"Resuming translation from segment {last_completed_idx + 1}", 
                "step"
            )
        
        db.update_task(
            task_id,
            status='processing',
            phase='translating',
            progress=35,
            message='Starting translation and dubbing...',
            was_running_at_shutdown=1
        )
        
        await broadcast_to_task(task_id, {
            'type': 'status_update',
            'data': {
                'status': 'processing',
                'phase': 'translating',
                'progress': 35,
                'was_running_at_shutdown': True
            }
        })
        
        # Get speaker configuration
        speaker_config = task.get('speaker_config', {})
        
        # Call the dubbing logic with resume info
        await start_dubbing_task(task_id, speaker_config, is_resume, last_completed_idx)
    
    async def _phase3_recomposition(self, task_id: str, task: Dict[str, Any], is_resume: bool = False):
        """Phase 3: Final video recomposition using the new modular pipeline."""
        await self._broadcast_log(task_id, "Phase 3: Final video assembly", "step")
        
        # Get video path
        video_path = task.get('video_path')
        if not video_path or not Path(video_path).exists():
            # Try alternative paths
            potential_paths = [
                Path("uploads") / f"{task_id}_{task.get('original_filename', '')}",
                Path("uploads") / task.get('original_filename', ''),
            ]
            for p in potential_paths:
                if p.exists():
                    video_path = str(p)
                    break
        
        if not video_path or not Path(video_path).exists():
            raise ValueError(f"Video file not found for recomposition: {video_path}")
        
        # Update status
        db.update_task(
            task_id,
            status='processing',
            phase='recomposing',
            progress=80,
            message='Assembling final video...',
            was_running_at_shutdown=1
        )
        
        await broadcast_to_task(task_id, {
            'type': 'status_update',
            'data': {
                'status': 'processing',
                'phase': 'recomposing',
                'progress': 80,
                'was_running_at_shutdown': True
            }
        })
        
        # Create progress callback
        async def progress_callback(phase: str, percent: int, message: str):
            await self._broadcast_log(task_id, message, "info")
            db.update_task(task_id, progress=percent, message=message)
            await broadcast_to_task(task_id, {
                'type': 'progress',
                'data': {'phase': phase, 'percent': percent, 'message': message}
            })
        
        # Run Phase 3
        try:
            final_video_path = await run_phase3_recomposition(
                task_id=task_id,
                original_video_path=video_path,
                use_demucs=False,  # Can be made configurable
                progress_callback=progress_callback
            )
            
            if final_video_path:
                await self._broadcast_log(task_id, f"Video complete: {final_video_path}", "success")
                # Status already updated by Phase3Recomposer
            else:
                raise RuntimeError("Phase 3 returned no output path")
                
        except Exception as e:
            await self._broadcast_log(task_id, f"Final assembly failed: {e}", "error")
            raise
    
    async def _broadcast_log(self, task_id: str, message: str, style: str = "info"):
        """Helper to log and broadcast."""
        append_log(task_id, message, style)
        from .logic import log_ws  # Avoid circular import
        await log_ws(task_id, message, style)

# Singleton instance
task_manager = TaskManager()
