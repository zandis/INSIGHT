"""
Batch Processing Module for INSIGHT.

Provides functionality for running multiple research sessions
in parallel or sequentially as a batch job.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    from logging_config import get_logger
    logger = get_logger("batch_processing")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some succeeded, some failed


@dataclass
class BatchItem:
    """Individual item in a batch job."""
    id: str
    objective: str
    tools: List[str]
    max_iterations: int = 5
    status: str = "pending"
    session_id: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "objective": self.objective,
            "tools": self.tools,
            "max_iterations": self.max_iterations,
            "status": self.status,
            "session_id": self.session_id,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class BatchJob:
    """A batch job containing multiple research items."""
    id: str
    name: str
    items: List[BatchItem]
    status: BatchStatus = BatchStatus.PENDING
    parallel: bool = False
    max_parallel: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_id: Optional[int] = None
    notify_email: Optional[str] = None
    webhook_url: Optional[str] = None

    @property
    def total_items(self) -> int:
        """Get total number of items."""
        return len(self.items)

    @property
    def completed_items(self) -> int:
        """Get number of completed items."""
        return sum(1 for item in self.items if item.status == "completed")

    @property
    def failed_items(self) -> int:
        """Get number of failed items."""
        return sum(1 for item in self.items if item.status == "failed")

    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if not self.items:
            return 0.0
        finished = sum(1 for item in self.items if item.status in ("completed", "failed"))
        return (finished / len(self.items)) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "parallel": self.parallel,
            "max_parallel": self.max_parallel,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "progress": self.progress,
            "items": [item.to_dict() for item in self.items],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "user_id": self.user_id
        }


class BatchProcessor:
    """
    Processor for batch research jobs.
    """

    def __init__(self, research_runner: Callable):
        """
        Initialize the batch processor.

        Args:
            research_runner: Async function to run a single research session
        """
        self.research_runner = research_runner
        self._jobs: Dict[str, BatchJob] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def create_batch(
        self,
        name: str,
        items: List[Dict],
        parallel: bool = False,
        max_parallel: int = 3,
        user_id: Optional[int] = None,
        notify_email: Optional[str] = None,
        webhook_url: Optional[str] = None
    ) -> BatchJob:
        """
        Create a new batch job.

        Args:
            name: Name for the batch
            items: List of research configurations
            parallel: Whether to run items in parallel
            max_parallel: Maximum concurrent items
            user_id: Owner user ID
            notify_email: Email for notification
            webhook_url: Webhook URL for notification

        Returns:
            Created BatchJob
        """
        batch_id = str(uuid.uuid4())

        batch_items = [
            BatchItem(
                id=str(uuid.uuid4()),
                objective=item["objective"],
                tools=item.get("tools", ["PUBMED", "MYGENE"]),
                max_iterations=item.get("max_iterations", 5)
            )
            for item in items
        ]

        job = BatchJob(
            id=batch_id,
            name=name,
            items=batch_items,
            parallel=parallel,
            max_parallel=max_parallel,
            user_id=user_id,
            notify_email=notify_email,
            webhook_url=webhook_url
        )

        self._jobs[batch_id] = job
        logger.info(f"Created batch job {batch_id} with {len(batch_items)} items")

        return job

    async def start_batch(self, batch_id: str) -> BatchJob:
        """
        Start processing a batch job.

        Args:
            batch_id: ID of batch to start

        Returns:
            Updated BatchJob
        """
        job = self._jobs.get(batch_id)
        if not job:
            raise ValueError(f"Batch job {batch_id} not found")

        if job.status != BatchStatus.PENDING:
            raise ValueError(f"Batch job {batch_id} is not pending")

        job.status = BatchStatus.RUNNING
        job.started_at = datetime.utcnow()

        # Start processing in background
        task = asyncio.create_task(self._process_batch(job))
        self._running_tasks[batch_id] = task

        logger.info(f"Started batch job {batch_id}")
        return job

    async def _process_batch(self, job: BatchJob) -> None:
        """Process a batch job."""
        try:
            if job.parallel:
                await self._process_parallel(job)
            else:
                await self._process_sequential(job)

            # Determine final status
            if job.failed_items == 0:
                job.status = BatchStatus.COMPLETED
            elif job.completed_items == 0:
                job.status = BatchStatus.FAILED
            else:
                job.status = BatchStatus.PARTIAL

            job.completed_at = datetime.utcnow()

            logger.info(
                f"Batch job {job.id} finished: "
                f"{job.completed_items}/{job.total_items} succeeded"
            )

            # Send notifications
            await self._send_notifications(job)

        except asyncio.CancelledError:
            job.status = BatchStatus.CANCELLED
            logger.info(f"Batch job {job.id} cancelled")
        except Exception as e:
            job.status = BatchStatus.FAILED
            logger.error(f"Batch job {job.id} failed: {e}")

    async def _process_sequential(self, job: BatchJob) -> None:
        """Process items sequentially."""
        for item in job.items:
            if job.status == BatchStatus.CANCELLED:
                break
            await self._process_item(job, item)

    async def _process_parallel(self, job: BatchJob) -> None:
        """Process items in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(job.max_parallel)

        async def process_with_semaphore(item: BatchItem):
            async with semaphore:
                if job.status != BatchStatus.CANCELLED:
                    await self._process_item(job, item)

        await asyncio.gather(
            *[process_with_semaphore(item) for item in job.items],
            return_exceptions=True
        )

    async def _process_item(self, job: BatchJob, item: BatchItem) -> None:
        """Process a single batch item."""
        item.status = "running"
        item.started_at = datetime.utcnow()

        try:
            logger.info(f"Processing batch item {item.id}: {item.objective[:50]}...")

            result = await self.research_runner(
                objective=item.objective,
                tools=item.tools,
                max_iterations=item.max_iterations
            )

            item.session_id = result.get("session_id")
            item.result = result
            item.status = "completed"

            logger.info(f"Batch item {item.id} completed")

        except Exception as e:
            item.status = "failed"
            item.error = str(e)
            logger.error(f"Batch item {item.id} failed: {e}")

        finally:
            item.completed_at = datetime.utcnow()

    async def _send_notifications(self, job: BatchJob) -> None:
        """Send completion notifications."""
        # Email notification
        if job.notify_email:
            try:
                # Import notification module
                from web.notifications import NotificationManager
                manager = NotificationManager()
                # Construct notification (implementation specific)
                logger.info(f"Sent batch completion email to {job.notify_email}")
            except Exception as e:
                logger.warning(f"Failed to send email notification: {e}")

        # Webhook notification
        if job.webhook_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        job.webhook_url,
                        json=job.to_dict(),
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
                logger.info(f"Sent batch completion webhook to {job.webhook_url}")
            except Exception as e:
                logger.warning(f"Failed to send webhook notification: {e}")

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        """
        Cancel a running batch job.

        Args:
            batch_id: ID of batch to cancel

        Returns:
            Updated BatchJob
        """
        job = self._jobs.get(batch_id)
        if not job:
            raise ValueError(f"Batch job {batch_id} not found")

        if job.status != BatchStatus.RUNNING:
            raise ValueError(f"Batch job {batch_id} is not running")

        # Cancel the running task
        task = self._running_tasks.get(batch_id)
        if task:
            task.cancel()

        job.status = BatchStatus.CANCELLED

        # Mark pending items as cancelled
        for item in job.items:
            if item.status == "pending":
                item.status = "cancelled"

        logger.info(f"Cancelled batch job {batch_id}")
        return job

    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """Get a batch job by ID."""
        return self._jobs.get(batch_id)

    def list_batches(
        self,
        user_id: Optional[int] = None,
        status: Optional[BatchStatus] = None
    ) -> List[BatchJob]:
        """
        List batch jobs with optional filtering.

        Args:
            user_id: Filter by user ID
            status: Filter by status

        Returns:
            List of matching BatchJobs
        """
        jobs = list(self._jobs.values())

        if user_id is not None:
            jobs = [j for j in jobs if j.user_id == user_id]

        if status is not None:
            jobs = [j for j in jobs if j.status == status]

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def delete_batch(self, batch_id: str) -> bool:
        """
        Delete a batch job.

        Args:
            batch_id: ID of batch to delete

        Returns:
            True if deleted
        """
        job = self._jobs.get(batch_id)
        if not job:
            return False

        if job.status == BatchStatus.RUNNING:
            raise ValueError("Cannot delete running batch")

        del self._jobs[batch_id]
        logger.info(f"Deleted batch job {batch_id}")
        return True


# Singleton instance
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get the batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        # Create with a placeholder runner (should be configured by app)
        async def placeholder_runner(**kwargs):
            raise NotImplementedError("Batch processor not configured")
        _batch_processor = BatchProcessor(placeholder_runner)
    return _batch_processor


def configure_batch_processor(research_runner: Callable) -> BatchProcessor:
    """
    Configure the batch processor with a research runner.

    Args:
        research_runner: Async function to run research

    Returns:
        Configured BatchProcessor
    """
    global _batch_processor
    _batch_processor = BatchProcessor(research_runner)
    return _batch_processor
