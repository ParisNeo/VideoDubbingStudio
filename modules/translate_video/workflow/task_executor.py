"""
Task execution engine with granular tracking and resumption support.
"""

import asyncio
import traceback
from typing import Dict, Any, Optional, List, Type
from datetime import datetime
from pathlib import Path

from core.database import db, TaskStatus
from .task_definitions import (
    TaskDefinition, TaskContext, TaskResult, 
    TaskType
)
import torch

class TaskExecutionError(Exception):
    """Exception raised when a task fails after all retry attempts."""
    def __init__(self, task_name: str, attempts: int, last_error: Exception):
        self.task_name = task_name
        self.attempts = attempts
        self.last_error = last_error
        self.traceback = traceback.format_exc()
        super().__init__(f"Task '{task_name}' failed after {attempts} attempts: {last_error}")


class TaskExecutor:
    """
    Executes workflow tasks with full lifecycle management.
    
    Features:
    - Granular task tracking in database
    - Automatic retry with exponential backoff
    - Checkpoint-based resumption for long tasks
    - GPU memory management between tasks
    - Cancellation support
    """
    
    def __init__(self, task_id: str):
        self.project_task_id = task_id
        self.current_context: Optional[TaskContext] = None
        self.cancellation_event = asyncio.Event()
        self._results_cache: Dict[str, TaskResult] = {}
        self._execution_order: List[str] = []
    
    async def execute_workflow(self, 
                              task_definitions: List[Dict[str, Any]],
                              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a full workflow of tasks.
        
        Args:
            task_definitions: List of task definitions from WorkflowBuilder
            resume_from: Task name to resume from (for crash recovery)
        
        Returns:
            Final execution summary
        """
        # Create/update workflow tasks in database
        db.create_workflow_tasks(self.project_task_id, task_definitions)
        
        # Build execution plan
        execution_plan = self._build_execution_plan(task_definitions, resume_from)
        
        # Execute tasks in order
        for task_def in execution_plan:
            task_name = task_def['name']
            
            # Check for cancellation
            if self.cancellation_event.is_set():
                await self._mark_remaining_cancelled(task_name, execution_plan)
                raise asyncio.CancelledError("Workflow cancelled")
            
            # Check conditional
            if task_def.get('condition'):
                context = await self._build_context(task_def)
                if not task_def['condition'](context):
                    await self._mark_skipped(task_name, "Condition evaluated to false")
                    continue
            
            # Execute the task
            result = await self._execute_single_task(task_def)
            
            if not result.success:
                # Task failed after all retries
                raise TaskExecutionError(task_name, task_def['max_attempts'], 
                                        Exception(result.error_message or "Unknown error"))
            
            # Cache result for downstream tasks
            self._results_cache[task_name] = result
            self._execution_order.append(task_name)
        
        # Workflow complete
        return {
            'success': True,
            'completed_tasks': self._execution_order,
            'final_outputs': self._results_cache.get(self._execution_order[-1], TaskResult(success=True)).output_data if self._execution_order else {}
        }
    
    async def execute_single_task(self, task_name: str,
                                   task_definitions: List[Dict[str, Any]]) -> TaskResult:
        """
        Execute a single task by name (for targeted re-run).
        """
        task_def = next((t for t in task_definitions if t['name'] == task_name), None)
        if not task_def:
            raise ValueError(f"Task '{task_name}' not found in workflow")
        
        return await self._execute_single_task(task_def)
    
    async def _execute_single_task(self, task_def: Dict[str, Any]) -> TaskResult:
        """Execute a single task with retry logic."""
        task_name = task_def['name']
        max_attempts = task_def.get('max_attempts', 3)
        task_type = task_def.get('task_type', 'cpu_bound')
        
        # Get task class
        from .task_definitions import TaskRegistry
        task_class = TaskRegistry.get(task_name)
        if not task_class:
            raise ValueError(f"Task class for '{task_name}' not found in registry")
        
        # Update database: mark as queued then running
        db.update_workflow_task(self.project_task_id, task_name, 'queued')
        db.update_current_task(self.project_task_id, task_name, 'queued')
        
        # Update main task status
        db.update_task(
            self.project_task_id,
            status='processing',
            phase=task_def.get('phase', 'processing'),
            current_task_name=task_name,
            current_task_status='queued'
        )
        
        # Build execution context
        context = await self._build_context(task_def)
        self.current_context = context
        
        # Execute with retries
        last_result = None
        for attempt in range(1, max_attempts + 1):
            try:
                # Mark as running
                db.update_workflow_task(self.project_task_id, task_name, 'running')
                db.update_current_task(self.project_task_id, task_name, 'running')
                
                # Report start
                await context.log(f"Starting task '{task_name}' (attempt {attempt}/{max_attempts})", "step")
                
                # Pre-execution: GPU setup for GPU-bound tasks
                if task_type == 'gpu_bound':
                    await self._prepare_gpu(task_def, context)
                
                # Execute
                result = await asyncio.wait_for(
                    task_class.execute(context),
                    timeout=task_def.get('timeout_seconds')
                )
                
                # Post-execution: GPU cleanup for GPU-bound tasks
                if task_type == 'gpu_bound':
                    await self._cleanup_gpu(task_def, context)
                
                # Success
                db.update_workflow_task(
                    self.project_task_id, task_name, 'completed',
                    output_data=result.output_data if result.success else None,
                    checkpoint_data=result.checkpoint_data if result.success else None
                )
                db.update_current_task(self.project_task_id, task_name, 'completed')
                
                await context.log(f"Task '{task_name}' completed successfully", "success")
                
                # Update progress based on workflow position
                await self._update_overall_progress(task_def)
                
                return result
                
            except asyncio.TimeoutError as e:
                last_result = TaskResult.failure_result(e, None)
                await context.log(f"Task '{task_name}' timed out (attempt {attempt})", "error")
                
                if attempt < max_attempts:
                    wait_time = 2 ** attempt  # Exponential backoff
                    await context.log(f"Retrying in {wait_time}s...", "info")
                    await asyncio.sleep(wait_time)
                
            except asyncio.CancelledError:
                # Propagate cancellation
                db.update_workflow_task(self.project_task_id, task_name, 'cancelled')
                raise
                
            except Exception as e:
                last_result = TaskResult.failure_result(e, None)
                tb_str = traceback.format_exc()
                await context.log(f"Task '{task_name}' failed: {str(e)}\n{tb_str}", "error")
                
                # Save checkpoint if task supports it
                checkpoint = None
                if hasattr(task_class, 'supports_checkpoints') and task_class.supports_checkpoints:
                    # Try to get checkpoint from exception or context
                    checkpoint = getattr(e, 'checkpoint_data', None)
                
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    await context.log(f"Retrying in {wait_time}s...", "info")
                    
                    # Save failed attempt info
                    db.update_workflow_task(
                        self.project_task_id, task_name, 'failed',
                        error_message=str(e),
                        error_traceback=tb_str,
                        checkpoint_data=checkpoint
                    )
                    
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure
                    db.update_workflow_task(
                        self.project_task_id, task_name, 'failed',
                        error_message=str(e),
                        error_traceback=tb_str
                    )
                    db.update_current_task(self.project_task_id, task_name, 'failed')
                    db.update_task(
                        self.project_task_id,
                        status='failed',
                        error_message=f"Task '{task_name}' failed: {str(e)}",
                        error_traceback=tb_str
                    )
        
        # All attempts exhausted
        return last_result or TaskResult.failure_result(Exception("All attempts failed"), None)
    
    async def _build_context(self, task_def: Dict[str, Any]) -> TaskContext:
        """Build execution context for a task."""
        # Get main task info for paths
        main_task = db.get_task(self.project_task_id)
        
        # Merge inputs: task-specific + previous results
        inputs = dict(task_def.get('inputs', {}))
        
        # Add outputs from dependency tasks
        for dep_name in task_def.get('depends_on', []):
            if dep_name in self._results_cache:
                dep_outputs = self._results_cache[dep_name].output_data
                # Prefix with dependency name to avoid conflicts
                for key, value in dep_outputs.items():
                    inputs[f"{dep_name}.{key}"] = value
        
        work_dir = Path("temp_chunks") / self.project_task_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = Path("outputs") / self.project_task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return TaskContext(
            task_id=self.project_task_id,
            task_name=task_def['name'],
            inputs=inputs,
            previous_results=dict(self._results_cache),
            project_state=main_task or {},
            work_dir=work_dir,
            output_dir=output_dir,
            uploads_dir=Path("uploads"),
            cancellation_event=self.cancellation_event,
            progress_callback=self._on_task_progress,
            log_callback=self._on_task_log
        )
    
    async def _prepare_gpu(self, task_def: Dict[str, Any], context: TaskContext):
        """Prepare GPU for a GPU-bound task."""
        from core.resources import manager
        
        gpu_exclusive = task_def.get('gpu_exclusive', False)
        required_vram = task_def.get('required_vram_gb')
        
        if gpu_exclusive:
            # Clear all models to ensure clean state
            await context.log("Clearing GPU cache for exclusive task...", "info")
            manager.clear_cache()
        
        # Check VRAM availability
        if required_vram and torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_vram < required_vram:
                await context.log(f"WARNING: Available VRAM ({total_vram:.1f}GB) "
                                 f"below recommended ({required_vram}GB)", "warning")
    
    async def _cleanup_gpu(self, task_def: Dict[str, Any], context: TaskContext):
        """Cleanup GPU after a GPU-bound task."""
        from core.resources import manager
        
        # Clear models that aren't needed by upcoming tasks
        # This is handled by the ResourceManager's LRU cache
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    async def _on_task_progress(self, task_name: str, percent: int, message: str):
        """Handle progress update from a task."""
        # Broadcast via WebSocket
        from modules.translate_video.state import broadcast_to_task
        
        await broadcast_to_task(self.project_task_id, {
            'type': 'task_progress',
            'data': {
                'task_name': task_name,
                'percent': percent,
                'message': message
            }
        })
        
        # Update database
        workflow_progress = db.get_workflow_progress(self.project_task_id)
        overall_percent = workflow_progress.get('percent', 0)
        
        db.update_task(
            self.project_task_id,
            progress=overall_percent,
            message=f"[{task_name}] {message}"
        )
    
    async def _on_task_log(self, message: str, style: str = "info"):
        """Handle log message from a task."""
        from modules.translate_video.state import broadcast_to_task
        
        await broadcast_to_task(self.project_task_id, {
            'type': 'log',
            'data': {'message': message, 'style': style}
        })
    
    async def _update_overall_progress(self, completed_task_def: Dict[str, Any]):
        """Update overall project progress after task completion."""
        progress = db.get_workflow_progress(self.project_task_id)
        
        db.update_task(
            self.project_task_id,
            progress=progress.get('percent', 0),
            message=f"Completed: {completed_task_def['name']}"
        )
    
    def _build_execution_plan(self, 
                             task_definitions: List[Dict[str, Any]],
                             resume_from: Optional[str]) -> List[Dict[str, Any]]:
        """Build ordered execution plan, optionally resuming from a point."""
        if not resume_from:
            return task_definitions
        
        # Find resume point
        resume_index = None
        for i, task_def in enumerate(task_definitions):
            if task_def['name'] == resume_from:
                resume_index = i
                break
        
        if resume_index is None:
            raise ValueError(f"Resume point '{resume_from}' not found in workflow")
        
        # Return tasks from resume point onwards
        # But first, check which previous tasks have completed results we can use
        all_tasks = db.get_all_workflow_tasks(self.project_task_id)
        
        # Load cached results for completed tasks
        for task in all_tasks:
            if task['status'] == 'completed' and task['output_data']:
                self._results_cache[task['task_name']] = TaskResult(
                    success=True,
                    output_data=task['output_data'],
                    checkpoint_data=task.get('checkpoint_data')
                )
                self._execution_order.append(task['task_name'])
        
        # Return remaining tasks
        return task_definitions[resume_index:]
    
    async def _mark_remaining_cancelled(self, from_task_name: str, 
                                         all_tasks: List[Dict[str, Any]]):
        """Mark all remaining tasks as cancelled."""
        found = False
        for task_def in all_tasks:
            if task_def['name'] == from_task_name:
                found = True
            if found:
                db.update_workflow_task(self.project_task_id, task_def['name'], 'cancelled')
    
    async def _mark_skipped(self, task_name: str, reason: str):
        """Mark a task as skipped."""
        db.update_workflow_task(
            self.project_task_id, task_name, 'skipped',
            output_data={'skipped': True, 'reason': reason}
        )
        
        from modules.translate_video.state import broadcast_to_task
        await broadcast_to_task(self.project_task_id, {
            'type': 'log',
            'data': {'message': f"Skipped task '{task_name}': {reason}", 'style': 'info'}
        })
    
    def cancel(self):
        """Signal cancellation of the workflow."""
        self.cancellation_event.set()
