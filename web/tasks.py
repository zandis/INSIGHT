"""
Task queue module for INSIGHT Web Application.

Provides async task processing for long-running research operations.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchTask:
    """Research task definition."""
    session_id: str
    objective: str
    tools: List[str]
    max_iterations: int
    user_data: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    current_task: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskQueue:
    """
    Async task queue for research operations.

    Features:
    - FIFO task processing
    - Progress tracking
    - Cancellation support
    - Callback hooks
    """

    def __init__(self, max_concurrent: int = 2):
        """
        Initialize task queue.

        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self._queue: asyncio.Queue[ResearchTask] = asyncio.Queue()
        self._processing: Dict[str, ResearchTask] = {}
        self._completed: Dict[str, ResearchTask] = {}
        self._max_concurrent = max_concurrent
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {
            "started": [],
            "progress": [],
            "completed": [],
            "failed": []
        }

    @property
    def pending_count(self) -> int:
        """Get number of pending tasks."""
        return self._queue.qsize()

    @property
    def processing_count(self) -> int:
        """Get number of processing tasks."""
        return len(self._processing)

    def add_callback(self, event: str, callback: Callable) -> None:
        """
        Add callback for task events.

        Args:
            event: Event name (started, progress, completed, failed)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _trigger_callbacks(self, event: str, task: ResearchTask) -> None:
        """Trigger callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                print(f"Callback error: {e}")

    async def enqueue(self, task: ResearchTask) -> None:
        """
        Add a task to the queue.

        Args:
            task: Research task to enqueue
        """
        await self._queue.put(task)
        print(f"Task {task.session_id} queued")

    async def process_tasks(self) -> None:
        """Main task processing loop."""
        self._running = True

        while self._running:
            # Check if we can process more tasks
            if len(self._processing) >= self._max_concurrent:
                await asyncio.sleep(0.5)
                continue

            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                # Start processing
                asyncio.create_task(self._process_task(task))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Task queue error: {e}")

    async def _process_task(self, task: ResearchTask) -> None:
        """
        Process a single research task.

        Args:
            task: Task to process
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        self._processing[task.session_id] = task

        await self._trigger_callbacks("started", task)

        try:
            # Import INSIGHT modules
            from web.database import get_db

            db = await get_db()

            # Update database status
            await db.update_session(
                task.session_id,
                status="running",
                progress=0
            )

            # Run the actual research
            await self._run_research(task, db)

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.progress = 100.0

            await db.update_session(
                task.session_id,
                status="completed",
                progress=100.0
            )

            await self._trigger_callbacks("completed", task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()

            try:
                db = await get_db()
                await db.update_session(
                    task.session_id,
                    status="failed"
                )
            except Exception:
                pass

            await self._trigger_callbacks("failed", task)
            print(f"Task {task.session_id} failed: {e}")

        finally:
            self._completed[task.session_id] = task
            if task.session_id in self._processing:
                del self._processing[task.session_id]

    async def _run_research(self, task: ResearchTask, db: Any) -> None:
        """
        Run the INSIGHT research pipeline.

        Args:
            task: Research task
            db: Database instance
        """
        try:
            # Import INSIGHT modules
            from utils import create_index, insert_doc_llama_index, query_knowledge_base, read_file
            from agents import boss_agent, worker_agent
            from utils import handle_python_result, handle_results, get_key_results
            from config import OPENAI_API_KEY
            import os
            from collections import defaultdict, deque

            api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")

            if not api_key:
                raise ValueError("OpenAI API key not configured")

            # Initialize
            master_index = create_index(api_key=api_key)
            summaries = []
            completed_tasks = []
            cache = defaultdict(list)
            task_list = deque()
            doc_store = {"tasks": {}}

            # Tool descriptions
            tool_description_mapping = {
                "MYGENE": "1) Query mygene API for gene information.",
                "PUBMED": "2) Query PubMed API for biomedical literature.",
                "MYVARIANT": "3) Query myvariant API for genetic variants."
            }

            tool_description = "\n".join(
                tool_description_mapping[t] for t in task.tools if t in tool_description_mapping
            )

            # Load user data if provided
            if task.user_data:
                temp_index = create_index(api_key=api_key)
                insert_doc_llama_index(temp_index, data=task.user_data, doc_id="user_data")
                exec_summary, _ = query_knowledge_base(temp_index, list_index=False)
                insert_doc_llama_index(index=master_index, doc_id="user_data", data=exec_summary)
                summaries.append(exec_summary)

            # Research loop
            result = []
            current_task_str = ""

            for iteration in range(task.max_iterations):
                # Update progress
                progress = (iteration / task.max_iterations) * 100
                task.progress = progress

                await db.update_session(
                    task.session_id,
                    progress=progress,
                    current_task=current_task_str
                )

                await self._trigger_callbacks("progress", task)

                # Get tasks from boss agent
                index = create_index(api_key)

                task_list = boss_agent(
                    objective=task.objective,
                    tool_description=tool_description,
                    task_list=task_list,
                    summaries=summaries,
                    completed_tasks=completed_tasks,
                    previous_task=current_task_str,
                    previous_result=result
                )

                if not task_list:
                    break

                # Execute first task
                current_task_str = task_list.popleft()
                task.current_task = current_task_str

                await db.update_session(
                    task.session_id,
                    current_task=current_task_str
                )

                # Run worker agent
                result_str, result_is_python = worker_agent(
                    task.objective,
                    current_task_str,
                    master_index,
                    cache,
                    task.tools
                )

                completed_tasks.append(current_task_str)

                # Process results
                doc_store_task_key = f"{iteration}_{current_task_str}"
                doc_store["tasks"][doc_store_task_key] = {"results": []}

                if result_is_python:
                    result = handle_python_result(
                        result_str, cache, current_task_str,
                        doc_store, doc_store_task_key
                    )

                if result:
                    handle_results(
                        result, index, doc_store, doc_store_task_key,
                        iteration, 20000
                    )

                    if index.docstore.docs:
                        exec_summary, citation = query_knowledge_base(index, list_index=False)
                        insert_doc_llama_index(
                            index=master_index,
                            doc_id=str(iteration),
                            data=exec_summary,
                            metadata={"citation_data": citation}
                        )
                        summaries.append(exec_summary)

                        # Store result in database
                        await db.add_result(
                            session_id=task.session_id,
                            task_id=doc_store_task_key,
                            content=exec_summary,
                            metadata={"citation": citation}
                        )

                # Update completed tasks in database
                await db.update_session(
                    task.session_id,
                    completed_tasks=completed_tasks
                )

            # Get key results
            key_results = get_key_results(master_index, task.objective, top_k=20)

            # Format key findings
            key_findings = "\n\n".join(
                f"{q}{r}" for q, r in key_results
            ) if key_results else None

            # Update final status
            await db.update_session(
                task.session_id,
                key_findings=key_findings,
                progress=100.0,
                status="completed"
            )

        except Exception as e:
            raise RuntimeError(f"Research failed: {e}")

    async def cancel_task(self, session_id: str) -> bool:
        """
        Cancel a task.

        Args:
            session_id: Session ID to cancel

        Returns:
            True if cancelled
        """
        if session_id in self._processing:
            task = self._processing[session_id]
            task.status = TaskStatus.CANCELLED
            return True
        return False

    def get_task_status(self, session_id: str) -> Optional[ResearchTask]:
        """
        Get task status.

        Args:
            session_id: Session ID

        Returns:
            Task or None
        """
        if session_id in self._processing:
            return self._processing[session_id]
        if session_id in self._completed:
            return self._completed[session_id]
        return None

    async def shutdown(self) -> None:
        """Shutdown the task queue."""
        self._running = False
        print("Task queue shutdown")
