"""
WebSocket manager for INSIGHT Web Application.

Provides real-time communication for:
- Research progress updates
- Live notifications
- Collaborative features
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Set

from fastapi import WebSocket


@dataclass
class WebSocketConnection:
    """WebSocket connection with metadata."""
    websocket: WebSocket
    session_ids: Set[str]
    connected_at: datetime
    user_id: int = None


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.

    Features:
    - Multiple sessions per connection
    - Broadcast to session subscribers
    - Connection health monitoring
    """

    def __init__(self):
        """Initialize WebSocket manager."""
        # Map session_id -> list of connections
        self._session_connections: Dict[str, List[WebSocket]] = defaultdict(list)
        # Map websocket -> connection info
        self._connections: Dict[WebSocket, WebSocketConnection] = {}
        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: int = None
    ) -> None:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket instance
            session_id: Session to subscribe to
            user_id: Optional user ID
        """
        await websocket.accept()

        async with self._lock:
            conn = WebSocketConnection(
                websocket=websocket,
                session_ids={session_id},
                connected_at=datetime.utcnow(),
                user_id=user_id
            )
            self._connections[websocket] = conn
            self._session_connections[session_id].append(websocket)

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

    def disconnect(self, websocket: WebSocket, session_id: str = None) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            websocket: WebSocket instance
            session_id: Session that was subscribed
        """
        if websocket in self._connections:
            conn = self._connections[websocket]

            # Remove from all session subscriptions
            for sid in conn.session_ids:
                if websocket in self._session_connections[sid]:
                    self._session_connections[sid].remove(websocket)

            del self._connections[websocket]

    def add_subscription(self, websocket: WebSocket, session_id: str) -> None:
        """
        Add session subscription to existing connection.

        Args:
            websocket: WebSocket instance
            session_id: Session to subscribe to
        """
        if websocket in self._connections:
            self._connections[websocket].session_ids.add(session_id)
            self._session_connections[session_id].append(websocket)

    async def broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any]
    ) -> None:
        """
        Broadcast message to all connections subscribed to a session.

        Args:
            session_id: Target session
            message: Message to send
        """
        message["timestamp"] = datetime.utcnow().isoformat()

        disconnected = []

        for websocket in self._session_connections.get(session_id, []):
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws, session_id)

    async def broadcast_progress(
        self,
        session_id: str,
        progress: float,
        current_task: str = None,
        results_count: int = 0
    ) -> None:
        """
        Broadcast progress update.

        Args:
            session_id: Session ID
            progress: Progress percentage
            current_task: Current task description
            results_count: Number of results
        """
        await self.broadcast_to_session(session_id, {
            "type": "progress",
            "session_id": session_id,
            "progress": progress,
            "current_task": current_task,
            "results_count": results_count
        })

    async def broadcast_completion(
        self,
        session_id: str,
        status: str,
        summary: str = None
    ) -> None:
        """
        Broadcast completion notification.

        Args:
            session_id: Session ID
            status: Final status (completed/failed)
            summary: Optional summary
        """
        await self.broadcast_to_session(session_id, {
            "type": "completion",
            "session_id": session_id,
            "status": status,
            "summary": summary
        })

    async def broadcast_result(
        self,
        session_id: str,
        task_id: str,
        content: str
    ) -> None:
        """
        Broadcast new result.

        Args:
            session_id: Session ID
            task_id: Task ID
            content: Result content
        """
        await self.broadcast_to_session(session_id, {
            "type": "result",
            "session_id": session_id,
            "task_id": task_id,
            "content": content[:500]  # Truncate for WebSocket
        })

    async def send_to_user(
        self,
        user_id: int,
        message: Dict[str, Any]
    ) -> None:
        """
        Send message to all connections for a user.

        Args:
            user_id: Target user ID
            message: Message to send
        """
        message["timestamp"] = datetime.utcnow().isoformat()

        for websocket, conn in self._connections.items():
            if conn.user_id == user_id:
                try:
                    await websocket.send_json(message)
                except Exception:
                    pass

    def get_connection_count(self) -> int:
        """Get total connection count."""
        return len(self._connections)

    def get_session_connection_count(self, session_id: str) -> int:
        """Get connection count for a session."""
        return len(self._session_connections.get(session_id, []))

    async def ping_all(self) -> None:
        """Send ping to all connections for health check."""
        disconnected = []

        for websocket in self._connections:
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                disconnected.append(websocket)

        for ws in disconnected:
            self.disconnect(ws)
