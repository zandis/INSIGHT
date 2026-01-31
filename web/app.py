"""
INSIGHT Web Application.

A FastAPI-based web interface for the INSIGHT medical research system.
Provides REST API, WebSocket support, and interactive dashboard.
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from web.database import (
    Database,
    ResearchSession,
    User,
    get_db,
    init_db,
)
from web.auth import (
    AuthManager,
    get_current_user,
    create_access_token,
)
from web.tasks import TaskQueue, ResearchTask
from web.websocket_manager import WebSocketManager
from web.export import ExportManager
from web.notifications import NotificationManager

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="INSIGHT API",
    description="Autonomous AI System for Medical Research",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Global managers
ws_manager = WebSocketManager()
task_queue = TaskQueue()
export_manager = ExportManager()
notification_manager = NotificationManager()
auth_manager = AuthManager()


# =============================================================================
# Pydantic Models
# =============================================================================

class ResearchRequest(BaseModel):
    """Request model for starting research."""
    objective: str = Field(..., min_length=10, max_length=5000)
    tools: List[str] = Field(default=["MYGENE", "PUBMED", "MYVARIANT"])
    max_iterations: int = Field(default=5, ge=1, le=50)
    user_data: Optional[str] = None


class ResearchResponse(BaseModel):
    """Response model for research operations."""
    session_id: str
    status: str
    message: str
    created_at: datetime


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    session_id: str
    status: str
    progress: float
    current_task: Optional[str]
    completed_tasks: List[str]
    results_count: int


class ExportRequest(BaseModel):
    """Request model for export operations."""
    session_id: str
    format: str = Field(default="json", pattern="^(json|pdf|csv|bibtex|markdown)$")
    include_citations: bool = True


class UserCreate(BaseModel):
    """Request model for user creation."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    """Request model for login."""
    username: str
    password: str


class WebhookConfig(BaseModel):
    """Configuration for webhooks."""
    url: str
    events: List[str] = ["completed", "failed"]
    secret: Optional[str] = None


class TemplateCreate(BaseModel):
    """Request model for creating research templates."""
    name: str
    description: str
    objective_template: str
    default_tools: List[str]
    default_iterations: int = 5


# =============================================================================
# Startup and Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Initialize database
    await init_db()

    # Start background task processor
    asyncio.create_task(task_queue.process_tasks())

    print("INSIGHT Web Application started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await task_queue.shutdown()
    print("INSIGHT Web Application shutdown")


# =============================================================================
# HTML Pages
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "INSIGHT - Medical Research AI"}
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "title": "Dashboard"}
    )


@app.get("/research/new", response_class=HTMLResponse)
async def new_research(request: Request):
    """Render new research page."""
    return templates.TemplateResponse(
        "new_research.html",
        {"request": request, "title": "New Research"}
    )


@app.get("/research/{session_id}", response_class=HTMLResponse)
async def view_research(request: Request, session_id: str):
    """Render research results page."""
    return templates.TemplateResponse(
        "research_view.html",
        {"request": request, "session_id": session_id, "title": "Research Results"}
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Render admin panel."""
    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "title": "Admin Panel"}
    )


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.post("/api/auth/register", response_model=Dict[str, Any])
async def register(user: UserCreate, db: Database = Depends(get_db)):
    """Register a new user."""
    existing = await db.get_user_by_username(user.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user_id = await db.create_user(
        username=user.username,
        email=user.email,
        password_hash=auth_manager.hash_password(user.password)
    )

    return {"user_id": user_id, "message": "User created successfully"}


@app.post("/api/auth/login", response_model=Dict[str, Any])
async def login(credentials: LoginRequest, db: Database = Depends(get_db)):
    """Login and get access token."""
    user = await db.get_user_by_username(credentials.username)

    if not user or not auth_manager.verify_password(
        credentials.password, user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600
    }


# =============================================================================
# Research Endpoints
# =============================================================================

@app.post("/api/research/start", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Start a new research session."""
    session_id = str(uuid.uuid4())

    # Create session in database
    await db.create_session(
        session_id=session_id,
        objective=request.objective,
        tools=request.tools,
        max_iterations=request.max_iterations,
        user_id=current_user.id if current_user else None
    )

    # Create research task
    task = ResearchTask(
        session_id=session_id,
        objective=request.objective,
        tools=request.tools,
        max_iterations=request.max_iterations,
        user_data=request.user_data
    )

    # Queue the task
    await task_queue.enqueue(task)

    return ResearchResponse(
        session_id=session_id,
        status="queued",
        message="Research started successfully",
        created_at=datetime.utcnow()
    )


@app.get("/api/research/{session_id}/status", response_model=TaskStatusResponse)
async def get_research_status(
    session_id: str,
    db: Database = Depends(get_db)
):
    """Get status of a research session."""
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return TaskStatusResponse(
        session_id=session_id,
        status=session.status,
        progress=session.progress,
        current_task=session.current_task,
        completed_tasks=session.completed_tasks,
        results_count=session.results_count
    )


@app.get("/api/research/{session_id}/results")
async def get_research_results(
    session_id: str,
    db: Database = Depends(get_db)
):
    """Get results of a research session."""
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    results = await db.get_session_results(session_id)

    return {
        "session_id": session_id,
        "objective": session.objective,
        "status": session.status,
        "results": results,
        "key_findings": session.key_findings,
        "citations": session.citations
    }


@app.delete("/api/research/{session_id}")
async def delete_research(
    session_id: str,
    db: Database = Depends(get_db)
):
    """Delete a research session."""
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete_session(session_id)

    return {"message": "Session deleted successfully"}


@app.get("/api/research/history")
async def get_research_history(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: Database = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get research history."""
    user_id = current_user.id if current_user else None
    sessions = await db.get_sessions(user_id=user_id, limit=limit, offset=offset)

    return {
        "sessions": sessions,
        "total": len(sessions),
        "limit": limit,
        "offset": offset
    }


# =============================================================================
# Export Endpoints
# =============================================================================

@app.post("/api/export")
async def export_results(
    request: ExportRequest,
    db: Database = Depends(get_db)
):
    """Export research results in various formats."""
    session = await db.get_session(request.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    results = await db.get_session_results(request.session_id)

    # Generate export
    export_data = await export_manager.export(
        session=session,
        results=results,
        format=request.format,
        include_citations=request.include_citations
    )

    # Return appropriate response based on format
    if request.format == "json":
        return JSONResponse(content=export_data)

    elif request.format in ["pdf", "csv", "bibtex", "markdown"]:
        filename = f"insight_export_{request.session_id[:8]}.{request.format}"

        return StreamingResponse(
            iter([export_data]),
            media_type=export_manager.get_media_type(request.format),
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )


@app.get("/api/export/bibtex/{session_id}")
async def export_bibtex(
    session_id: str,
    db: Database = Depends(get_db)
):
    """Export citations in BibTeX format."""
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    bibtex = await export_manager.generate_bibtex(session.citations)

    return StreamingResponse(
        iter([bibtex]),
        media_type="application/x-bibtex",
        headers={"Content-Disposition": f"attachment; filename=citations_{session_id[:8]}.bib"}
    )


# =============================================================================
# Templates Endpoints
# =============================================================================

@app.get("/api/templates")
async def list_templates(db: Database = Depends(get_db)):
    """List available research templates."""
    templates = await db.get_templates()
    return {"templates": templates}


@app.post("/api/templates")
async def create_template(
    template: TemplateCreate,
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new research template."""
    template_id = await db.create_template(
        name=template.name,
        description=template.description,
        objective_template=template.objective_template,
        default_tools=template.default_tools,
        default_iterations=template.default_iterations,
        created_by=current_user.id
    )

    return {"template_id": template_id, "message": "Template created"}


@app.get("/api/templates/{template_id}")
async def get_template(template_id: str, db: Database = Depends(get_db)):
    """Get a specific template."""
    template = await db.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return template


# =============================================================================
# Webhook Endpoints
# =============================================================================

@app.post("/api/webhooks")
async def register_webhook(
    config: WebhookConfig,
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Register a webhook for notifications."""
    webhook_id = await db.create_webhook(
        user_id=current_user.id,
        url=config.url,
        events=config.events,
        secret=config.secret
    )

    return {"webhook_id": webhook_id, "message": "Webhook registered"}


@app.get("/api/webhooks")
async def list_webhooks(
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List registered webhooks."""
    webhooks = await db.get_webhooks(user_id=current_user.id)
    return {"webhooks": webhooks}


@app.delete("/api/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a webhook."""
    await db.delete_webhook(webhook_id, user_id=current_user.id)
    return {"message": "Webhook deleted"}


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket, session_id)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif message.get("type") == "subscribe":
                # Subscribe to additional sessions
                extra_session = message.get("session_id")
                if extra_session:
                    ws_manager.add_subscription(websocket, extra_session)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id)


# =============================================================================
# File Upload Endpoints
# =============================================================================

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Upload a file for research context."""
    # Validate file type
    allowed_types = [".txt", ".pdf", ".csv", ".json", ".md"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {allowed_types}"
        )

    # Save file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_id = str(uuid.uuid4())
    file_path = upload_dir / f"{file_id}{file_ext}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "path": str(file_path)
    }


# =============================================================================
# Metrics and Admin Endpoints
# =============================================================================

@app.get("/api/metrics")
async def get_metrics(db: Database = Depends(get_db)):
    """Get system metrics."""
    stats = await db.get_stats()

    # Try to get metrics from metrics module
    try:
        from metrics import get_metrics as get_insight_metrics
        metrics = get_insight_metrics()
        api_metrics = metrics.get_summary()
    except ImportError:
        api_metrics = {}

    return {
        "database": stats,
        "api": api_metrics,
        "task_queue": {
            "pending": task_queue.pending_count,
            "processing": task_queue.processing_count
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        from health_check import get_health_checker
        checker = get_health_checker()
        overall, results = checker.get_overall_status()

        return {
            "status": overall.value,
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "latency_ms": result.latency_ms
                }
                for name, result in results.items()
            }
        }
    except ImportError:
        return {"status": "healthy", "message": "Basic health check passed"}


@app.get("/api/admin/sessions")
async def admin_list_sessions(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: Database = Depends(get_db)
):
    """Admin endpoint to list all sessions."""
    sessions = await db.get_all_sessions(status=status, limit=limit)
    return {"sessions": sessions, "total": len(sessions)}


@app.post("/api/admin/cleanup")
async def admin_cleanup(
    days_old: int = Query(default=30, ge=1),
    db: Database = Depends(get_db)
):
    """Admin endpoint to cleanup old sessions."""
    deleted = await db.cleanup_old_sessions(days=days_old)
    return {"deleted_count": deleted, "message": f"Cleaned up sessions older than {days_old} days"}


# =============================================================================
# Server-Sent Events for Streaming
# =============================================================================

@app.get("/api/research/{session_id}/stream")
async def stream_research_progress(session_id: str):
    """Stream research progress using Server-Sent Events."""

    async def event_generator():
        last_progress = 0

        while True:
            # Get current status
            try:
                db = await get_db()
                session = await db.get_session(session_id)

                if not session:
                    yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                    break

                # Send update if progress changed
                if session.progress != last_progress or session.status in ["completed", "failed"]:
                    yield f"data: {json.dumps({
                        'status': session.status,
                        'progress': session.progress,
                        'current_task': session.current_task,
                        'results_count': session.results_count
                    })}\n\n"

                    last_progress = session.progress

                # Stop if completed or failed
                if session.status in ["completed", "failed"]:
                    break

                await asyncio.sleep(1)

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# =============================================================================
# Comparison Endpoints
# =============================================================================

@app.post("/api/compare")
async def compare_sessions(
    session_ids: List[str],
    db: Database = Depends(get_db)
):
    """Compare multiple research sessions."""
    if len(session_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 sessions required")

    if len(session_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 sessions for comparison")

    sessions = []
    for sid in session_ids:
        session = await db.get_session(sid)
        if session:
            results = await db.get_session_results(sid)
            sessions.append({
                "session_id": sid,
                "objective": session.objective,
                "status": session.status,
                "results_count": len(results),
                "key_findings": session.key_findings,
                "tools_used": session.tools
            })

    return {
        "sessions": sessions,
        "comparison": {
            "total_sessions": len(sessions),
            "common_themes": [],  # Would be computed by NLP
            "unique_findings": []  # Would be computed by NLP
        }
    }


# =============================================================================
# Search and Favorites
# =============================================================================

@app.get("/api/search")
async def search_sessions(
    q: str = Query(..., min_length=3),
    db: Database = Depends(get_db)
):
    """Search across research sessions."""
    results = await db.search_sessions(query=q)
    return {"query": q, "results": results, "count": len(results)}


@app.post("/api/favorites/{session_id}")
async def add_favorite(
    session_id: str,
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a session to favorites."""
    await db.add_favorite(user_id=current_user.id, session_id=session_id)
    return {"message": "Added to favorites"}


@app.delete("/api/favorites/{session_id}")
async def remove_favorite(
    session_id: str,
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove a session from favorites."""
    await db.remove_favorite(user_id=current_user.id, session_id=session_id)
    return {"message": "Removed from favorites"}


@app.get("/api/favorites")
async def list_favorites(
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List favorite sessions."""
    favorites = await db.get_favorites(user_id=current_user.id)
    return {"favorites": favorites}


# =============================================================================
# Run Application
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
