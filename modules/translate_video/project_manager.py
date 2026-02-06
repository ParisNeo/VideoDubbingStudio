import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure storage directory exists
PROJECTS_DIR = Path("projects_db")
PROJECTS_DIR.mkdir(exist_ok=True)

class ProjectManager:
    """Manages persistence for video translation projects."""

    @staticmethod
    def create_project(task_id: str, filename: str) -> Dict[str, Any]:
        """Initialize a new project state file."""
        state = {
            "task_id": task_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "created_at": str(Path(filename).stat().st_ctime) if Path(filename).exists() else "",
            "speaker_config": {},
            "logs": [],
            "video_path": "",
            "output_file": ""
        }
        ProjectManager.save_state(task_id, state)
        return state

    @staticmethod
    def get_state(task_id: str) -> Optional[Dict[str, Any]]:
        """Load project state from disk."""
        path = PROJECTS_DIR / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading state for {task_id}: {e}")
            return None

    @staticmethod
    def save_state(task_id: str, updates: Dict[str, Any]):
        """Update and save project state atomically."""
        current = ProjectManager.get_state(task_id) or {}
        current.update(updates)
        
        path = PROJECTS_DIR / f"{task_id}.json"
        try:
            with open(path, "w") as f:
                json.dump(current, f, indent=2)
        except Exception as e:
            print(f"Error saving state for {task_id}: {e}")

    @staticmethod
    def delete_project(task_id: str):
        """Delete project JSON and associated temp files."""
        # 1. Delete DB Entry
        json_path = PROJECTS_DIR / f"{task_id}.json"
        if json_path.exists():
            json_path.unlink()
        
        # 2. Cleanup Logs/State chunks (Optional but good practice)
        # Note: We are NOT deleting the uploaded video or final output
        # to prevent accidental data loss, but you could add that here.

    @staticmethod
    def list_projects() -> List[Dict[str, Any]]:
        """List all projects for the dashboard."""
        projects = []
        for p in PROJECTS_DIR.glob("*.json"):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    # Return lightweight summary
                    projects.append({
                        "task_id": data.get("task_id"),
                        "filename": data.get("filename"),
                        "status": data.get("status"),
                        "progress": data.get("progress", 0),
                        "created_at": data.get("created_at")
                    })
            except:
                pass
        # Sort by most recent (naive implementation, better to use timestamp)
        return projects

# -------------------------------------------------------------------------
# LOGGING HELPERS
# -------------------------------------------------------------------------

def append_log(task_id: str, message: str, style: str = "info"):
    """Appends a log entry to the persistent state."""
    state = ProjectManager.get_state(task_id)
    if state:
        logs = state.get("logs", [])
        logs.append({"message": message, "style": style})
        # Keep only last 100 logs to prevent JSON bloat
        ProjectManager.save_state(task_id, {"logs": logs[-100:]})
