import sqlite3
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum, auto

class TaskStatus(str, Enum):
    """Granular task statuses for fine-grained workflow control."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

class WorkflowPhase(str, Enum):
    """Macro phases for high-level organization."""
    INIT = "init"
    AUDIO_EXTRACTION = "audio_extraction"
    SPEAKER_IDENTIFICATION = "speaker_identification"
    TRANSCRIPTION = "transcription"
    VALIDATION = "validation"
    TRANSLATION = "translation"
    TTS_SYNTHESIS = "tts_synthesis"
    AUDIO_RECOMPOSITION = "audio_recomposition"
    VIDEO_MERGE = "video_merge"
    COMPLETE = "complete"
    FAILED = "failed"

class Database:
    """SQLite-based task persistence with granular task tracking."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Database, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_path = Path("tasks.db")
        self._local = threading.local()
        self._init_db()
        self._initialized = True
    
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize database schema with granular task tracking."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if tables exist and need migration
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        if cursor.fetchone() is not None:
            self._migrate_if_needed(cursor, conn)
            return
        
        # Main tasks table
        cursor.execute('''
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                filename TEXT,
                file_path TEXT,
                video_path TEXT,
                
                -- Legacy status fields (for backward compatibility)
                status TEXT DEFAULT 'pending',
                phase TEXT DEFAULT 'init',
                progress REAL DEFAULT 0,
                
                -- Configuration
                source TEXT,
                tgt_lang TEXT DEFAULT 'en',
                src_lang TEXT DEFAULT 'auto',
                separate_audio INTEGER DEFAULT 0,
                tts_engine TEXT DEFAULT 'f5',
                whisper_model TEXT DEFAULT 'large-v2',
                vad_threshold REAL DEFAULT 0.25,
                
                -- Result tracking
                output_path TEXT,
                error_message TEXT,
                error_traceback TEXT,
                
                -- Timing
                created_at TEXT,
                updated_at TEXT,
                
                -- Lifecycle
                was_running_at_shutdown INTEGER DEFAULT 0,
                resume_attempts INTEGER DEFAULT 0,
                last_checkpoint_phase TEXT,
                
                -- Legacy data blob
                data TEXT,
                
                -- NEW: Current active task for granular tracking
                current_task_id TEXT,
                current_task_name TEXT,
                current_task_status TEXT,
                
                -- Aggregated metadata
                speaker_config TEXT,
                segments TEXT,
                assignments TEXT,
                master_audio TEXT,
                transcribed_segments TEXT,
                translation_segments TEXT,
                background_audio TEXT
            )
        ''')
        
        # NEW: Granular workflow_tasks table for individual task tracking
        cursor.execute('''
            CREATE TABLE workflow_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                task_name TEXT NOT NULL,
                task_order INTEGER NOT NULL,
                
                -- Task classification
                phase TEXT NOT NULL,  -- Macro phase this belongs to
                task_group TEXT,      -- Sub-group within phase
                
                -- Status and timing
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                
                -- Execution tracking
                attempt_count INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 3,
                
                -- Input/output data
                input_data TEXT,      -- JSON of inputs
                output_data TEXT,     -- JSON of results
                checkpoint_data TEXT, -- For resume within long tasks
                
                -- Error tracking
                error_message TEXT,
                error_traceback TEXT,
                
                -- Foreign key
                FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            )
        ''')
        
        # Index for fast task queries
        cursor.execute('''
            CREATE INDEX idx_workflow_tasks_lookup 
            ON workflow_tasks(task_id, task_order, status)
        ''')
        
        # Index for phase-based queries
        cursor.execute('''
            CREATE INDEX idx_workflow_tasks_phase 
            ON workflow_tasks(task_id, phase, status)
        ''')
        
        # Legacy translation_segments table (kept for compatibility)
        cursor.execute('''
            CREATE TABLE translation_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                segment_idx INTEGER,
                original_text TEXT,
                translated_text TEXT,
                audio_path TEXT,
                status TEXT DEFAULT 'pending',
                start_time REAL DEFAULT 0,
                end_time REAL DEFAULT 0,
                speaker_id INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_translation_segments_task 
            ON translation_segments(task_id, segment_idx)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
        ''')
        
        conn.commit()
    
    def _migrate_if_needed(self, cursor, conn):
        """Migrate existing database to new schema."""
        cursor.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Add new granular tracking columns if missing
        new_columns = {
            'current_task_id': 'TEXT',
            'current_task_name': 'TEXT',
            'current_task_status': 'TEXT',
            'translation_segments': 'TEXT',
            'background_audio': 'TEXT',
            'error_traceback': 'TEXT',
            'whisper_model': 'TEXT'
        }
        
        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                try:
                    cursor.execute(f'ALTER TABLE tasks ADD COLUMN {col_name} {col_type}')
                    print(f"Migrated database: added column {col_name}")
                except sqlite3.OperationalError as e:
                    print(f"Migration warning for {col_name}: {e}")
        
        # Check if workflow_tasks table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_tasks'")
        if not cursor.fetchone():
            print("Creating workflow_tasks table for granular task tracking...")
            cursor.execute('''
                CREATE TABLE workflow_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    task_order INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    task_group TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    attempt_count INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    input_data TEXT,
                    output_data TEXT,
                    checkpoint_data TEXT,
                    error_message TEXT,
                    error_traceback TEXT,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                )
            ''')
            cursor.execute('CREATE INDEX idx_workflow_tasks_lookup ON workflow_tasks(task_id, task_order, status)')
            cursor.execute('CREATE INDEX idx_workflow_tasks_phase ON workflow_tasks(task_id, phase, status)')
        
        conn.commit()
    
    def create_task(self, task_id: str, filename: str, file_path: str):
        """Create a new task entry."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO tasks (task_id, filename, file_path, video_path, created_at, updated_at, data, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (task_id, filename, file_path, file_path, now, now, '{}', 'pending'))
        
        conn.commit()
    
    def create_workflow_tasks(self, task_id: str, task_definitions: List[Dict[str, Any]]):
        """Create granular workflow tasks for a project."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute("DELETE FROM workflow_tasks WHERE task_id = ?", (task_id,))
        
        for i, task_def in enumerate(task_definitions):
            cursor.execute('''
                INSERT INTO workflow_tasks (
                    task_id, task_name, task_order, phase, task_group,
                    status, created_at, max_attempts, input_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_id,
                task_def['name'],
                i,
                task_def.get('phase', 'init'),
                task_def.get('group'),
                'pending',
                now,
                task_def.get('max_attempts', 3),
                json.dumps(task_def.get('inputs', {})) if task_def.get('inputs') else None
            ))
        
        conn.commit()
    
    def get_next_pending_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the next pending workflow task for a project."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM workflow_tasks 
            WHERE task_id = ? AND status IN ('pending', 'failed', 'queued')
            ORDER BY task_order 
            LIMIT 1
        ''', (task_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._parse_workflow_task_row(row)
    
    def get_workflow_task(self, task_id: str, task_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific workflow task by name."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM workflow_tasks 
            WHERE task_id = ? AND task_name = ?
        ''', (task_id, task_name))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._parse_workflow_task_row(row)
    
    def update_workflow_task(
        self,
        task_id: str,
        task_name: str,
        status: str,
        output_data: Optional[Dict] = None,
        checkpoint_data: Optional[Dict] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None
    ):
        """Update a workflow task's status and data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        # Build update dynamically
        updates = []
        params = []
        
        if status:
            updates.append("status = ?")
            params.append(status)
            
            if status == 'running':
                updates.append("started_at = ?")
                params.append(now)
                updates.append("attempt_count = attempt_count + 1")
            elif status in ('completed', 'skipped'):
                updates.append("completed_at = ?")
                params.append(now)
        
        if output_data is not None:
            updates.append("output_data = ?")
            params.append(json.dumps(output_data))
        
        if checkpoint_data is not None:
            updates.append("checkpoint_data = ?")
            params.append(json.dumps(checkpoint_data))
        
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        
        if error_traceback is not None:
            updates.append("error_traceback = ?")
            params.append(error_traceback)
        
        if not updates:
            return
        
        params.extend([task_id, task_name])
        
        sql = f"UPDATE workflow_tasks SET {', '.join(updates)} WHERE task_id = ? AND task_name = ?"
        cursor.execute(sql, params)
        conn.commit()
    
    def _parse_workflow_task_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Parse a workflow task database row into a dict."""
        result = dict(row)
        
        # Parse JSON fields
        for field in ['input_data', 'output_data', 'checkpoint_data']:
            if result.get(field) and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = None
        
        return result
    
    def get_all_workflow_tasks(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all workflow tasks for a project, ordered by execution order."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM workflow_tasks 
            WHERE task_id = ? 
            ORDER BY task_order
        ''', (task_id,))
        
        return [self._parse_workflow_task_row(row) for row in cursor.fetchall()]
    
    def get_workflow_progress(self, task_id: str) -> Dict[str, Any]:
        """Get aggregated progress statistics for a project's workflow."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) as skipped,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                SUM(CASE WHEN status IN ('pending', 'queued') THEN 1 ELSE 0 END) as pending
            FROM workflow_tasks 
            WHERE task_id = ?
        ''', (task_id,))
        
        row = cursor.fetchone()
        
        total = row['total'] or 0
        completed = row['completed'] or 0
        failed = row['failed'] or 0
        skipped = row['skipped'] or 0
        running = row['running'] or 0
        pending = row['pending'] or 0
        
        # Calculate percentage (exclude skipped from denominator for progress)
        denominator = total - skipped if total > 0 else 1
        completed_denom = completed if denominator > 0 else 0
        percent = (completed_denom / denominator * 100) if denominator > 0 else 0
        
        # Get current task
        cursor.execute('''
            SELECT task_name, phase, status FROM workflow_tasks 
            WHERE task_id = ? AND status IN ('running', 'queued')
            ORDER BY task_order LIMIT 1
        ''', (task_id,))
        
        current = cursor.fetchone()
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'running': running,
            'pending': pending,
            'percent': round(percent, 1),
            'current_task': dict(current) if current else None
        }
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve main task by ID with enriched workflow data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        result = dict(row)
        
        # Parse JSON fields
        json_fields = ['data', 'speaker_config', 'segments', 'assignments', 
                      'checkpoint_data', 'transcribed_segments', 'translation_segments']
        for field in json_fields:
            raw = result.get(field)
            if raw and isinstance(raw, str):
                try:
                    result[field] = json.loads(raw)
                except json.JSONDecodeError:
                    result[field] = None
        
        # Handle boolean fields
        def safe_int(value, default=0):
            if value is None:
                return default
            if isinstance(value, bool):
                return 1 if value else 0
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    if value.strip().lower() in ['true', 'yes', '1', 'on']:
                        return 1
                    return int(value) if value.strip().lower() not in ['false', 'no', '0', 'off'] else 0
                except ValueError:
                    return default
            return default
        
        result['was_running_at_shutdown'] = bool(safe_int(result.get('was_running_at_shutdown'), 0))
        result['separate_audio'] = bool(safe_int(result.get('separate_audio'), 0))
        result['resume_attempts'] = safe_int(result.get('resume_attempts'), 0)
        
        # Ensure language fields
        if result.get('target_language') and not result.get('tgt_lang'):
            result['tgt_lang'] = result['target_language']
        elif result.get('tgt_lang') and not result.get('target_language'):
            result['target_language'] = result['tgt_lang']
        elif not result.get('tgt_lang') and not result.get('target_language'):
            result['tgt_lang'] = 'en'
            result['target_language'] = 'en'
        
        if not result.get('src_lang'):
            result['src_lang'] = 'auto'
        if not result.get('tts_engine'):
            result['tts_engine'] = 'f5'
        
        # Add workflow progress
        result['workflow_progress'] = self.get_workflow_progress(task_id)
        result['workflow_tasks'] = self.get_all_workflow_tasks(task_id)
        
        return result
    
    def update_task(self, task_id: str, **updates: Dict[str, Any]):
        """Update main task with new fields."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Field mapping for compatibility
        field_mapping = {'tgt_lang': 'target_language'}
        
        # Normalize updates
        normalized_updates = {}
        for key, value in updates.items():
            if key in field_mapping:
                normalized_updates[field_mapping[key]] = value
            normalized_updates[key] = value
        
        # Direct columns
        direct_columns = {
            'filename', 'file_path', 'video_path', 'status', 'phase', 'progress',
            'target_language', 'source', 'created_at', 'updated_at', 'error_message',
            'error_traceback', 'output_path', 'input_filename', 'speaker_config', 
            'segments', 'assignments', 'master_audio', 'checkpoint_data', 
            'was_running_at_shutdown', 'resume_attempts', 'last_checkpoint_phase',
            'src_lang', 'separate_audio', 'tts_engine', 'transcribed_segments',
            'translation_segments', 'background_audio', 'current_task_id',
            'current_task_name', 'current_task_status', 'whisper_model'
        }
        
        base_updates = {}
        data_updates = {}
        
        for key, value in normalized_updates.items():
            is_direct = key in direct_columns
            
            if is_direct:
                if key in ['speaker_config', 'segments', 'assignments', 'checkpoint_data',
                          'transcribed_segments', 'translation_segments'] and value is not None:
                    base_updates[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
                elif key in ['was_running_at_shutdown', 'separate_audio', 'resume_attempts']:
                    if isinstance(value, bool):
                        base_updates[key] = 1 if value else 0
                    elif isinstance(value, int):
                        base_updates[key] = value
                    else:
                        base_updates[key] = 1 if str(value).lower() in ['true', 'yes', '1', 'on'] else 0
                else:
                    base_updates[key] = value
            else:
                data_updates[key] = value
        
        # Execute base updates
        if base_updates:
            set_clause = ', '.join([f"{k} = ?" for k in base_updates.keys()])
            values = list(base_updates.values()) + [datetime.now().isoformat(), task_id]
            cursor.execute(f'UPDATE tasks SET {set_clause}, updated_at = ? WHERE task_id = ?', values)
        
        # Update data blob
        if data_updates:
            current_data = {}
            try:
                cursor.execute('SELECT data FROM tasks WHERE task_id = ?', (task_id,))
                row = cursor.fetchone()
                if row and row['data']:
                    current_data = json.loads(row['data'])
            except:
                pass
            
            current_data.update(data_updates)
            cursor.execute('UPDATE tasks SET data = ?, updated_at = ? WHERE task_id = ?',
                          (json.dumps(current_data), datetime.now().isoformat(), task_id))
        
        conn.commit()
    
    def update_current_task(self, task_id: str, task_name: str, task_status: str):
        """Update which granular task is currently active."""
        self.update_task(
            task_id,
            current_task_id=f"{task_id}:{task_name}",
            current_task_name=task_name,
            current_task_status=task_status
        )
    
    def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent tasks with workflow summary."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tasks 
            ORDER BY datetime(updated_at) DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        result = []
        for row in rows:
            task = self.get_task(row['task_id'])
            if task:
                # Lightweight summary
                summary = {
                    'task_id': task.get('task_id'),
                    'filename': task.get('filename') or task.get('original_filename', 'Unknown'),
                    'status': task.get('status', 'unknown'),
                    'phase': task.get('phase', 'unknown'),
                    'progress': task.get('progress', 0),
                    'created_at': task.get('created_at', ''),
                    'updated_at': task.get('updated_at', ''),
                    'was_running_at_shutdown': task.get('was_running_at_shutdown', False),
                    'resume_attempts': task.get('resume_attempts', 0),
                    'error_message': task.get('error_message', ''),
                    'src_lang': task.get('src_lang', 'auto'),
                    'tgt_lang': task.get('tgt_lang', 'en'),
                    'separate_audio': task.get('separate_audio', False),
                    'tts_engine': task.get('tts_engine', 'f5'),
                    'current_task': {
                        'name': task.get('current_task_name'),
                        'status': task.get('current_task_status')
                    } if task.get('current_task_name') else None,
                    'workflow_summary': task.get('workflow_progress', {})
                }
                result.append(summary)
        
        return result
    
    def get_interrupted_tasks(self) -> List[Dict[str, Any]]:
        """Find tasks that were running when the app shut down."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE was_running_at_shutdown = 1
               OR (status IN ('processing', 'queued', 'running')
                   AND phase NOT IN ('complete', 'failed', 'cancelled'))
            ORDER BY datetime(updated_at) DESC
        ''')
        
        rows = cursor.fetchall()
        result = []
        for row in rows:
            task = self.get_task(row['task_id'])
            if task:
                result.append(task)
        
        return result
    
    def clear_running_flags(self):
        """Clear all was_running_at_shutdown flags."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE tasks SET was_running_at_shutdown = 0')
        conn.commit()
    
    def delete_task(self, task_id: str):
        """Delete task and all related data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Cascade will delete workflow_tasks
        cursor.execute('DELETE FROM translation_segments WHERE task_id = ?', (task_id,))
        cursor.execute('DELETE FROM tasks WHERE task_id = ?', (task_id,))
        conn.commit()
    
    # Legacy translation segments methods (kept for compatibility)
    def save_translation_segment(self, task_id: str, segment_idx: int,
                                  original_text: str, translated_text: str,
                                  audio_path: Optional[str] = None,
                                  status: str = 'completed',
                                  start_time: float = 0, end_time: float = 0,
                                  speaker_id: int = 0):
        """Legacy method - now stored in task's translation_segments JSON."""
        # Also save to legacy table for compatibility
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute('''
            SELECT id FROM translation_segments 
            WHERE task_id = ? AND segment_idx = ?
        ''', (task_id, segment_idx))
        
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('''
                UPDATE translation_segments 
                SET original_text = ?, translated_text = ?, audio_path = ?, 
                    status = ?, start_time = ?, end_time = ?, speaker_id = ?, updated_at = ?
                WHERE task_id = ? AND segment_idx = ?
            ''', (original_text, translated_text, audio_path, status, 
                  start_time, end_time, speaker_id, now, task_id, segment_idx))
        else:
            cursor.execute('''
                INSERT INTO translation_segments 
                (task_id, segment_idx, original_text, translated_text, audio_path, 
                 status, start_time, end_time, speaker_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (task_id, segment_idx, original_text, translated_text, audio_path,
                  status, start_time, end_time, speaker_id, now, now))
        
        conn.commit()
    
    def get_translation_segments(self, task_id: str) -> List[Dict[str, Any]]:
        """Get legacy translation segments."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM translation_segments 
            WHERE task_id = ? 
            ORDER BY segment_idx
        ''', (task_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def delete_translation_segments(self, task_id: str):
        """Delete legacy translation segments."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM translation_segments WHERE task_id = ?', (task_id,))
        conn.commit()

# Singleton instance
db = Database()
