import sqlite3
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

class Database:
    """SQLite-based task persistence with JSON fields for flexibility."""
    
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
        """Initialize database schema - migrates existing tables if needed."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if tasks table exists and its structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check existing columns
            cursor.execute("PRAGMA table_info(tasks)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Add missing columns if needed
            required_columns = {
                'filename': 'TEXT',
                'file_path': 'TEXT', 
                'video_path': 'TEXT',
                'status': 'TEXT DEFAULT \'queued\'',
                'phase': 'TEXT DEFAULT \'init\'',
                'progress': 'REAL DEFAULT 0',
                'target_language': 'TEXT DEFAULT \'en\'',
                'source': 'TEXT',
                'created_at': 'TEXT',
                'updated_at': 'TEXT',
                'data': 'TEXT',
                'error_message': 'TEXT',
                'output_path': 'TEXT',
                'input_filename': 'TEXT',
                'tgt_lang': 'TEXT',
                'src_lang': 'TEXT DEFAULT \'auto\'',  # NEW: source language
                'separate_audio': 'INTEGER DEFAULT 0',  # NEW: background separation flag
                'tts_engine': 'TEXT DEFAULT \'f5\'',  # NEW: TTS engine selection
                'speaker_config': 'TEXT',
                'segments': 'TEXT',
                'assignments': 'TEXT',
                'master_audio': 'TEXT',
                'checkpoint_data': 'TEXT',
                'was_running_at_shutdown': 'INTEGER DEFAULT 0',
                'resume_attempts': 'INTEGER DEFAULT 0',
                'last_checkpoint_phase': 'TEXT',
                'transcribed_segments': 'TEXT'  # NEW: pre-transcribed segments from Phase 1
            }
            
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f'ALTER TABLE tasks ADD COLUMN {col_name} {col_type}')
                        print(f"Migrated database: added column {col_name}")
                    except sqlite3.OperationalError as e:
                        print(f"Migration warning for {col_name}: {e}")
            
            # Check translation_segments table columns
            cursor.execute("PRAGMA table_info(translation_segments)")
            seg_columns = {row[1] for row in cursor.fetchall()}
            
            seg_required = {
                'start_time': 'REAL DEFAULT 0',
                'end_time': 'REAL DEFAULT 0',
                'speaker_id': 'INTEGER DEFAULT 0'
            }
            
            for col_name, col_type in seg_required.items():
                if col_name not in seg_columns:
                    try:
                        cursor.execute(f'ALTER TABLE translation_segments ADD COLUMN {col_name} {col_type}')
                        print(f"Migrated translation_segments: added column {col_name}")
                    except sqlite3.OperationalError as e:
                        print(f"Migration warning for {col_name}: {e}")
            
            conn.commit()
            return
        
        # Tasks table - fresh creation with all columns including new ones
        cursor.execute('''
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                filename TEXT,
                file_path TEXT,
                video_path TEXT,
                status TEXT DEFAULT 'queued',
                phase TEXT DEFAULT 'init',
                progress REAL DEFAULT 0,
                target_language TEXT DEFAULT 'en',
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                data TEXT,
                error_message TEXT,
                output_path TEXT,
                input_filename TEXT,
                tgt_lang TEXT,
                src_lang TEXT DEFAULT 'auto',  -- NEW: source language for transcription
                separate_audio INTEGER DEFAULT 0,  -- NEW: background separation
                tts_engine TEXT DEFAULT 'f5',  -- NEW: TTS engine (f5 or fishspeech)
                speaker_config TEXT,
                segments TEXT,
                assignments TEXT,
                master_audio TEXT,
                transcribed_segments TEXT,  -- NEW: pre-transcribed segments from Phase 1
                checkpoint_data TEXT,
                was_running_at_shutdown INTEGER DEFAULT 0,
                resume_attempts INTEGER DEFAULT 0,
                last_checkpoint_phase TEXT
            )
        ''')
        
        # Translation segments table with timing info
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translation_segments (
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
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_translation_segments_task 
            ON translation_segments(task_id, segment_idx)
        ''')
        
        # Create index for faster status queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
        ''')
        
        conn.commit()
    
    def create_task(self, task_id: str, filename: str, file_path: str):
        """Create a new task entry."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO tasks (task_id, filename, file_path, video_path, created_at, updated_at, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (task_id, filename, file_path, file_path, now, now, '{}'))
        
        conn.commit()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task by ID, merging stored data with base fields."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Build result dict from row
        result = dict(row)
        
        # Parse JSON fields
        json_fields = ['data', 'speaker_config', 'segments', 'assignments', 'checkpoint_data', 'transcribed_segments']
        for field in json_fields:
            raw = result.get(field)
            if raw and isinstance(raw, str):
                try:
                    result[field] = json.loads(raw)
                except json.JSONDecodeError:
                    result[field] = None
        
        # Handle boolean fields
        if 'was_running_at_shutdown' in result:
            result['was_running_at_shutdown'] = bool(result['was_running_at_shutdown'])
        if 'separate_audio' in result:
            result['separate_audio'] = bool(result['separate_audio'])
        
        # Add computed fields for compatibility
        if result.get('file_path') and not result.get('video_path'):
            result['video_path'] = result['file_path']
        
        # Add alias for API compatibility: target_language -> tgt_lang
        if result.get('target_language') and 'tgt_lang' not in result:
            result['tgt_lang'] = result['target_language']
        
        return result
    
    def update_task(self, task_id: str, **updates: Dict[str, Any]):
        """Update task with new fields. Handles field name mapping for compatibility."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Field name mapping: API names -> database column names
        field_mapping = {
            'tgt_lang': 'target_language',
        }
        
        # Fields that go into the data blob vs direct columns
        direct_columns = {
            'filename', 'file_path', 'video_path', 'status', 'phase', 'progress',
            'target_language', 'source', 'created_at', 'updated_at', 'error_message',
            'output_path', 'input_filename', 'speaker_config', 'segments', 
            'assignments', 'master_audio', 'checkpoint_data', 'was_running_at_shutdown',
            'resume_attempts', 'last_checkpoint_phase',
            # NEW fields
            'src_lang', 'separate_audio', 'tts_engine', 'transcribed_segments'
        }
        
        # Apply mappings: convert API field names to DB column names
        normalized_updates = {}
        for key, value in updates.items():
            if key in field_mapping:
                normalized_updates[field_mapping[key]] = value
            else:
                normalized_updates[key] = key
        
        # Separate base fields from extra data
        base_updates = {}
        data_updates = {}
        
        for key, value in normalized_updates.items():
            if key in direct_columns:
                # JSON-serialize complex types
                if key in ['speaker_config', 'segments', 'assignments', 'checkpoint_data', 'transcribed_segments'] and value is not None:
                    if isinstance(value, (dict, list)):
                        base_updates[key] = json.dumps(value)
                    else:
                        base_updates[key] = value
                elif key in ['was_running_at_shutdown', 'separate_audio']:
                    # Boolean to integer
                    base_updates[key] = 1 if value else 0
                else:
                    base_updates[key] = value
            else:
                data_updates[key] = value
        
        # Update base fields directly
        if base_updates:
            set_clause = ', '.join([f"{k} = ?" for k in base_updates.keys()])
            values = list(base_updates.values()) + [datetime.now().isoformat(), task_id]
            cursor.execute(f'UPDATE tasks SET {set_clause}, updated_at = ? WHERE task_id = ?', values)
        
        # Merge and update data blob
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
    
    def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent tasks."""
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
                # Return lightweight summary
                result.append({
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
                    # NEW: include config options in summary
                    'src_lang': task.get('src_lang', 'auto'),
                    'tgt_lang': task.get('tgt_lang', 'en'),
                    'separate_audio': task.get('separate_audio', False),
                    'tts_engine': task.get('tts_engine', 'f5')
                })
        
        return result
    
    def get_interrupted_tasks(self) -> List[Dict[str, Any]]:
        """Find tasks that were running when the app shut down (for crash recovery)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Find tasks marked as running OR tasks with "running-like" status but not completed/failed
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE was_running_at_shutdown = 1
               OR (status IN ('processing', 'queued', 'identifying', 'translating', 'recomposing')
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
        """Clear all was_running_at_shutdown flags (called on clean shutdown)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE tasks SET was_running_at_shutdown = 0')
        conn.commit()
    
    def delete_task(self, task_id: str):
        """Delete task and its segments."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM translation_segments WHERE task_id = ?', (task_id,))
        cursor.execute('DELETE FROM tasks WHERE task_id = ?', (task_id,))
        conn.commit()
    
    # -------------------------------------------------------------------------
    # Translation Segments Methods
    # -------------------------------------------------------------------------
    
    def save_translation_segment(
        self,
        task_id: str,
        segment_idx: int,
        original_text: str,
        translated_text: str,
        audio_path: Optional[str] = None,
        status: str = 'completed',
        start_time: float = 0,
        end_time: float = 0,
        speaker_id: int = 0
    ):
        """Save or update a translation segment with full timing info."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        # Check if segment exists
        cursor.execute('''
            SELECT id FROM translation_segments 
            WHERE task_id = ? AND segment_idx = ?
        ''', (task_id, segment_idx))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            cursor.execute('''
                UPDATE translation_segments 
                SET original_text = ?, translated_text = ?, audio_path = ?, 
                    status = ?, start_time = ?, end_time = ?, speaker_id = ?, updated_at = ?
                WHERE task_id = ? AND segment_idx = ?
            ''', (original_text, translated_text, audio_path, status, 
                  start_time, end_time, speaker_id, now, task_id, segment_idx))
        else:
            # Insert new
            cursor.execute('''
                INSERT INTO translation_segments 
                (task_id, segment_idx, original_text, translated_text, audio_path, 
                 status, start_time, end_time, speaker_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (task_id, segment_idx, original_text, translated_text, audio_path,
                  status, start_time, end_time, speaker_id, now, now))
        
        conn.commit()
    
    def get_translation_segments(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all translation segments for a task."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM translation_segments 
            WHERE task_id = ? 
            ORDER BY segment_idx
        ''', (task_id,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def delete_translation_segments(self, task_id: str):
        """Delete all translation segments for a task (for restart)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM translation_segments WHERE task_id = ?', (task_id,))
        conn.commit()

# Singleton instance
db = Database()
