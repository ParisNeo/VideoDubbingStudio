import asyncio
from typing import Dict, Set, Any
from fastapi import WebSocket

# Active WebSocket connections by task
active_connections: Dict[str, Set[WebSocket]] = {}

# Task control events
task_events: Dict[str, Dict[str, asyncio.Event]] = {}

async def connect_task_websocket(task_id: str, websocket: WebSocket):
    """Register a WebSocket connection for a task."""
    await websocket.accept()
    if task_id not in active_connections:
        active_connections[task_id] = set()
    active_connections[task_id].add(websocket)
    
    # Send current state immediately
    from core.database import db
    task = db.get_task(task_id)
    if task:
        await websocket.send_json({
            'type': 'state_sync',
            'data': task
        })

async def disconnect_task_websocket(task_id: str, websocket: WebSocket):
    """Remove a WebSocket connection."""
    if task_id in active_connections:
        active_connections[task_id].discard(websocket)
        if not active_connections[task_id]:
            del active_connections[task_id]

async def broadcast_to_task(task_id: str, message: Dict[str, Any]):
    """Send message to all connected clients for a task."""
    if task_id not in active_connections:
        return
    
    dead_sockets = set()
    for ws in active_connections[task_id]:
        try:
            await ws.send_json(message)
        except Exception:
            dead_sockets.add(ws)
    
    # Clean up dead connections
    for ws in dead_sockets:
        active_connections[task_id].discard(ws)
