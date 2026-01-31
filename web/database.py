"""
Database module for INSIGHT Web Application.

Provides SQLite-based persistence for:
- Research sessions
- Users and authentication
- Templates
- Webhooks
- Favorites and search history
"""

import asyncio
import aiosqlite
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class User:
    """User model."""
    id: int
    username: str
    email: str
    password_hash: str
    created_at: datetime
    is_admin: bool = False


@dataclass
class ResearchSession:
    """Research session model."""
    id: str
    objective: str
    tools: List[str]
    max_iterations: int
    status: str
    progress: float
    current_task: Optional[str]
    completed_tasks: List[str]
    results_count: int
    key_findings: Optional[str]
    citations: List[Dict]
    user_id: Optional[int]
    created_at: datetime
    updated_at: datetime


@dataclass
class Template:
    """Research template model."""
    id: str
    name: str
    description: str
    objective_template: str
    default_tools: List[str]
    default_iterations: int
    created_by: int
    created_at: datetime


@dataclass
class Webhook:
    """Webhook configuration model."""
    id: str
    user_id: int
    url: str
    events: List[str]
    secret: Optional[str]
    is_active: bool
    created_at: datetime


# =============================================================================
# Database Class
# =============================================================================

class Database:
    """
    Async SQLite database manager with connection management.

    Provides CRUD operations for all application data.
    Features:
    - Lazy connection initialization
    - Connection health checks
    - Proper cleanup
    """

    def __init__(self, db_path: str = "insight.db"):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish database connection with locking."""
        async with self._lock:
            if self._connection is not None:
                # Check if connection is still valid
                try:
                    await self._connection.execute("SELECT 1")
                    return  # Connection is valid
                except Exception:
                    # Connection is stale, close it
                    try:
                        await self._connection.close()
                    except Exception:
                        pass
                    self._connection = None

            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row

    async def disconnect(self) -> None:
        """Close database connection safely."""
        async with self._lock:
            if self._connection:
                try:
                    await self._connection.close()
                except Exception:
                    pass
                finally:
                    self._connection = None

    async def execute(self, query: str, params: tuple = ()) -> aiosqlite.Cursor:
        """Execute a query with connection management."""
        if not self._connection:
            await self.connect()
        try:
            return await self._connection.execute(query, params)
        except aiosqlite.OperationalError as e:
            # Connection may have been lost, try to reconnect once
            if "no such table" not in str(e).lower():
                await self.connect()
                return await self._connection.execute(query, params)
            raise

    async def commit(self) -> None:
        """Commit transaction."""
        if self._connection:
            await self._connection.commit()

    async def fetchone(self, query: str, params: tuple = ()) -> Optional[aiosqlite.Row]:
        """Fetch single row."""
        cursor = await self.execute(query, params)
        return await cursor.fetchone()

    async def fetchall(self, query: str, params: tuple = ()) -> List[aiosqlite.Row]:
        """Fetch all rows."""
        cursor = await self.execute(query, params)
        return await cursor.fetchall()

    # =========================================================================
    # Schema Creation
    # =========================================================================

    async def create_tables(self) -> None:
        """Create all database tables."""
        queries = [
            # Users table
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,

            # Sessions table
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                tools TEXT NOT NULL,
                max_iterations INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                progress REAL DEFAULT 0,
                current_task TEXT,
                completed_tasks TEXT DEFAULT '[]',
                results_count INTEGER DEFAULT 0,
                key_findings TEXT,
                citations TEXT DEFAULT '[]',
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """,

            # Results table
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """,

            # Templates table
            """
            CREATE TABLE IF NOT EXISTS templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                objective_template TEXT NOT NULL,
                default_tools TEXT NOT NULL,
                default_iterations INTEGER DEFAULT 5,
                created_by INTEGER,
                is_public INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
            """,

            # Webhooks table
            """
            CREATE TABLE IF NOT EXISTS webhooks (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                events TEXT NOT NULL,
                secret TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,

            # Favorites table
            """
            CREATE TABLE IF NOT EXISTS favorites (
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, session_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """,

            # Search history table
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                query TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,

            # API keys table
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_hash TEXT NOT NULL,
                name TEXT,
                permissions TEXT DEFAULT '[]',
                is_active INTEGER DEFAULT 1,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,

            # Create indexes
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
            "CREATE INDEX IF NOT EXISTS idx_results_session ON results(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id)",
        ]

        for query in queries:
            await self.execute(query)
        await self.commit()

    # =========================================================================
    # User Operations
    # =========================================================================

    async def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        is_admin: bool = False
    ) -> int:
        """Create a new user."""
        cursor = await self.execute(
            """
            INSERT INTO users (username, email, password_hash, is_admin)
            VALUES (?, ?, ?, ?)
            """,
            (username, email, password_hash, int(is_admin))
        )
        await self.commit()
        return cursor.lastrowid

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        row = await self.fetchone(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )
        if row:
            return User(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                password_hash=row["password_hash"],
                is_admin=bool(row["is_admin"]),
                created_at=datetime.fromisoformat(row["created_at"])
            )
        return None

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        row = await self.fetchone(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        )
        if row:
            return User(
                id=row["id"],
                username=row["username"],
                email=row["email"],
                password_hash=row["password_hash"],
                is_admin=bool(row["is_admin"]),
                created_at=datetime.fromisoformat(row["created_at"])
            )
        return None

    # =========================================================================
    # Session Operations
    # =========================================================================

    async def create_session(
        self,
        session_id: str,
        objective: str,
        tools: List[str],
        max_iterations: int,
        user_id: Optional[int] = None
    ) -> str:
        """Create a new research session."""
        await self.execute(
            """
            INSERT INTO sessions (id, objective, tools, max_iterations, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, objective, json.dumps(tools), max_iterations, user_id)
        )
        await self.commit()
        return session_id

    async def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get session by ID."""
        row = await self.fetchone(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        if row:
            return ResearchSession(
                id=row["id"],
                objective=row["objective"],
                tools=json.loads(row["tools"]),
                max_iterations=row["max_iterations"],
                status=row["status"],
                progress=row["progress"],
                current_task=row["current_task"],
                completed_tasks=json.loads(row["completed_tasks"]),
                results_count=row["results_count"],
                key_findings=row["key_findings"],
                citations=json.loads(row["citations"]),
                user_id=row["user_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            )
        return None

    async def update_session(
        self,
        session_id: str,
        **kwargs
    ) -> None:
        """Update session fields."""
        updates = []
        values = []

        for key, value in kwargs.items():
            if key in ["tools", "completed_tasks", "citations"]:
                value = json.dumps(value)
            updates.append(f"{key} = ?")
            values.append(value)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(session_id)

        await self.execute(
            f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?",
            tuple(values)
        )
        await self.commit()

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        await self.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self.commit()

    async def get_sessions(
        self,
        user_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        """Get sessions with optional filtering."""
        query = "SELECT * FROM sessions WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = await self.fetchall(query, tuple(params))
        return [dict(row) for row in rows]

    async def get_all_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get all sessions (admin)."""
        return await self.get_sessions(status=status, limit=limit)

    async def search_sessions(self, query: str) -> List[Dict]:
        """Search sessions by objective."""
        rows = await self.fetchall(
            """
            SELECT * FROM sessions
            WHERE objective LIKE ? OR key_findings LIKE ?
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (f"%{query}%", f"%{query}%")
        )
        return [dict(row) for row in rows]

    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Delete sessions older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        cursor = await self.execute(
            "DELETE FROM sessions WHERE created_at < ? AND status = 'completed'",
            (cutoff.isoformat(),)
        )
        await self.commit()
        return cursor.rowcount

    # =========================================================================
    # Results Operations
    # =========================================================================

    async def add_result(
        self,
        session_id: str,
        task_id: str,
        content: str,
        metadata: Dict = None,
        embedding: List[float] = None
    ) -> int:
        """Add a result to a session."""
        cursor = await self.execute(
            """
            INSERT INTO results (session_id, task_id, content, metadata, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                task_id,
                content,
                json.dumps(metadata or {}),
                json.dumps(embedding) if embedding else None
            )
        )
        await self.commit()

        # Update results count
        await self.execute(
            """
            UPDATE sessions SET results_count = (
                SELECT COUNT(*) FROM results WHERE session_id = ?
            ) WHERE id = ?
            """,
            (session_id, session_id)
        )
        await self.commit()

        return cursor.lastrowid

    async def get_session_results(self, session_id: str) -> List[Dict]:
        """Get all results for a session."""
        rows = await self.fetchall(
            "SELECT * FROM results WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        return [
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"]
            }
            for row in rows
        ]

    # =========================================================================
    # Template Operations
    # =========================================================================

    async def create_template(
        self,
        name: str,
        description: str,
        objective_template: str,
        default_tools: List[str],
        default_iterations: int,
        created_by: int
    ) -> str:
        """Create a new template."""
        import uuid
        template_id = str(uuid.uuid4())

        await self.execute(
            """
            INSERT INTO templates
            (id, name, description, objective_template, default_tools, default_iterations, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                template_id,
                name,
                description,
                objective_template,
                json.dumps(default_tools),
                default_iterations,
                created_by
            )
        )
        await self.commit()
        return template_id

    async def get_templates(self) -> List[Dict]:
        """Get all templates."""
        rows = await self.fetchall(
            "SELECT * FROM templates WHERE is_public = 1 ORDER BY name"
        )
        return [dict(row) for row in rows]

    async def get_template(self, template_id: str) -> Optional[Dict]:
        """Get template by ID."""
        row = await self.fetchone(
            "SELECT * FROM templates WHERE id = ?",
            (template_id,)
        )
        return dict(row) if row else None

    # =========================================================================
    # Webhook Operations
    # =========================================================================

    async def create_webhook(
        self,
        user_id: int,
        url: str,
        events: List[str],
        secret: Optional[str] = None
    ) -> str:
        """Create a new webhook."""
        import uuid
        webhook_id = str(uuid.uuid4())

        await self.execute(
            """
            INSERT INTO webhooks (id, user_id, url, events, secret)
            VALUES (?, ?, ?, ?, ?)
            """,
            (webhook_id, user_id, url, json.dumps(events), secret)
        )
        await self.commit()
        return webhook_id

    async def get_webhooks(self, user_id: int) -> List[Dict]:
        """Get webhooks for a user."""
        rows = await self.fetchall(
            "SELECT * FROM webhooks WHERE user_id = ? AND is_active = 1",
            (user_id,)
        )
        return [dict(row) for row in rows]

    async def delete_webhook(self, webhook_id: str, user_id: int) -> None:
        """Delete a webhook."""
        await self.execute(
            "DELETE FROM webhooks WHERE id = ? AND user_id = ?",
            (webhook_id, user_id)
        )
        await self.commit()

    # =========================================================================
    # Favorites Operations
    # =========================================================================

    async def add_favorite(self, user_id: int, session_id: str) -> None:
        """Add session to favorites."""
        await self.execute(
            "INSERT OR IGNORE INTO favorites (user_id, session_id) VALUES (?, ?)",
            (user_id, session_id)
        )
        await self.commit()

    async def remove_favorite(self, user_id: int, session_id: str) -> None:
        """Remove session from favorites."""
        await self.execute(
            "DELETE FROM favorites WHERE user_id = ? AND session_id = ?",
            (user_id, session_id)
        )
        await self.commit()

    async def get_favorites(self, user_id: int) -> List[Dict]:
        """Get user's favorite sessions."""
        rows = await self.fetchall(
            """
            SELECT s.* FROM sessions s
            JOIN favorites f ON s.id = f.session_id
            WHERE f.user_id = ?
            ORDER BY f.created_at DESC
            """,
            (user_id,)
        )
        return [dict(row) for row in rows]

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_sessions = await self.fetchone("SELECT COUNT(*) as count FROM sessions")
        active_sessions = await self.fetchone(
            "SELECT COUNT(*) as count FROM sessions WHERE status IN ('pending', 'running')"
        )
        total_users = await self.fetchone("SELECT COUNT(*) as count FROM users")
        total_results = await self.fetchone("SELECT COUNT(*) as count FROM results")

        return {
            "total_sessions": total_sessions["count"] if total_sessions else 0,
            "active_sessions": active_sessions["count"] if active_sessions else 0,
            "total_users": total_users["count"] if total_users else 0,
            "total_results": total_results["count"] if total_results else 0
        }


# =============================================================================
# Database Instance Management
# =============================================================================

_db_instance: Optional[Database] = None


async def init_db() -> Database:
    """Initialize database and create tables."""
    global _db_instance
    _db_instance = Database()
    await _db_instance.connect()
    await _db_instance.create_tables()
    return _db_instance


async def get_db() -> Database:
    """Get database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = await init_db()
    return _db_instance
