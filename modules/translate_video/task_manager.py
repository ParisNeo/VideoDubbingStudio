import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from core.database import db
from .project_manager import append_log
from .state import broadcast_to_task

# Import Workflow System
from .workflow import WorkflowBuilder, TaskExecutor

class TaskManager:
    """Manages the video translation pipeline using granular tasks."""
    
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.executors: Dict[str, TaskExecutor] = {}
        self._lock = asyncio.Lock()
    
    async def start_task(self, task_id: str, is_resume: bool = False):
        """Start or resume a task pipeline."""
        async with self._lock:
            if task_id in self.active_tasks:
                return
            
            task = db.get_task(task_id)
            if not task:
                print(f"Task {task_id} not found")
                return
            
            # Create Executor
            executor = TaskExecutor(task_id)
            self.executors[task_id] = executor
            
            # Start workflow execution
            pipeline_task = asyncio.create_task(self._run_workflow(task_id, executor, task, is_resume))
            self.active_tasks[task_id] = pipeline_task
            
            pipeline_task.add_done_callback(
                lambda t: asyncio.create_task(self._cleanup_task(task_id))
            )
            
            db.update_task(task_id, was_running_at_shutdown=1)

    async def _run_workflow(self, task_id: str, executor: TaskExecutor, task: Dict[str, Any], is_resume: bool):
        try:
            phase = task.get('phase', 'init')
            
            # Define Workflow based on phase
            builder = WorkflowBuilder.create_default_workflow(task)
            
            # Filter tasks if we are in a specific phase and NOT resuming the whole pipeline
            # If resuming from 'translating', we only run phase 2 tasks, etc.
            # But the executor handles "resume_from" logic. 
            
            # However, since we have the "Validation Gap" (stop after Phase 1),
            # we need to conditionally run parts of the workflow.
            
            active_definitions = []
            
            if phase in ['init', 'identifying']:
                # Run Phase 1 tasks
                active_definitions = [t for t in builder if t['phase'] in ['identifying', 'init']]
                
            elif phase == 'translating':
                # Run Phase 2 tasks
                active_definitions = [t for t in builder if t['phase'] == 'translating']
                
            elif phase == 'recomposing':
                # Run Phase 3 tasks
                active_definitions = [t for t in builder if t['phase'] == 'recomposing']
                
            elif phase == 'awaiting_validation':
                # Waiting state
                return
            
            if not active_definitions:
                print(f"No tasks for phase {phase}")
                return

            await executor.execute_workflow(active_definitions)
            
            # If Phase 1 finished successfully, we are now awaiting validation
            if phase == 'identifying':
                # Check status via DB to ensure it wasn't cancelled/failed
                updated = db.get_task(task_id)
                if updated['status'] == 'awaiting_validation':
                    pass # Already handled by TranscribeSegmentsTask
                    
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Workflow failed: {e}\n{tb_str}")
            db.update_task(task_id, status='failed', error_message=str(e))
            await broadcast_to_task(task_id, {'type': 'error', 'message': str(e)})

    async def _cleanup_task(self, task_id: str):
        async with self._lock:
            self.active_tasks.pop(task_id, None)
            self.executors.pop(task_id, None)
        db.update_task(task_id, was_running_at_shutdown=0)

    async def cancel_task(self, task_id: str):
        if task_id in self.executors:
            self.executors[task_id].cancel()
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
        db.update_task(task_id, status='cancelled', was_running_at_shutdown=0)

    async def pause_task(self, task_id: str):
        # Cancellation without failure
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
        
        # Clear old workflow tasks if restarting from scratch
        if new_phase == 'init':
            # Could clear workflow_tasks table for this task_id
            pass
            
        await self.start_task(task_id)

    async def recover_interrupted_tasks(self):
        # Simplistic recovery: just restart them
        tasks = db.get_interrupted_tasks()
        for t in tasks:
            print(f"Recovering task {t['task_id']}")
            await self.start_task(t['task_id'], is_resume=True)

    async def shutdown(self):
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)

task_manager = TaskManager()
