"""
Scheduler Module for INSIGHT.

Provides cron-like scheduling for automated research jobs.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    from logging_config import get_logger
    logger = get_logger("scheduler")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ScheduleFrequency(Enum):
    """Supported schedule frequencies."""
    ONCE = "once"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ScheduledJob:
    """A scheduled research job."""
    id: str
    name: str
    objective: str
    tools: List[str]
    max_iterations: int = 5
    frequency: ScheduleFrequency = ScheduleFrequency.ONCE
    cron_expression: Optional[str] = None  # For custom schedules
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    last_status: Optional[str] = None
    last_session_id: Optional[str] = None
    enabled: bool = True
    run_count: int = 0
    user_id: Optional[int] = None
    notify_email: Optional[str] = None
    webhook_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "objective": self.objective,
            "tools": self.tools,
            "max_iterations": self.max_iterations,
            "frequency": self.frequency.value,
            "cron_expression": self.cron_expression,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_status": self.last_status,
            "last_session_id": self.last_session_id,
            "enabled": self.enabled,
            "run_count": self.run_count,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat()
        }

    def calculate_next_run(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate the next run time based on frequency."""
        base_time = from_time or datetime.utcnow()

        if self.frequency == ScheduleFrequency.ONCE:
            return None  # No next run for one-time jobs

        elif self.frequency == ScheduleFrequency.HOURLY:
            return base_time + timedelta(hours=1)

        elif self.frequency == ScheduleFrequency.DAILY:
            return base_time + timedelta(days=1)

        elif self.frequency == ScheduleFrequency.WEEKLY:
            return base_time + timedelta(weeks=1)

        elif self.frequency == ScheduleFrequency.MONTHLY:
            # Approximate month as 30 days
            return base_time + timedelta(days=30)

        elif self.frequency == ScheduleFrequency.CUSTOM and self.cron_expression:
            return self._parse_cron_next(base_time)

        return None

    def _parse_cron_next(self, from_time: datetime) -> Optional[datetime]:
        """Parse cron expression to get next run time."""
        try:
            import croniter
            cron = croniter.croniter(self.cron_expression, from_time)
            return cron.get_next(datetime)
        except ImportError:
            logger.warning("croniter not installed, custom schedules not supported")
            return None
        except Exception as e:
            logger.error(f"Invalid cron expression: {e}")
            return None


class Scheduler:
    """
    Scheduler for automated research jobs.
    """

    def __init__(self, research_runner: Callable):
        """
        Initialize the scheduler.

        Args:
            research_runner: Async function to run research
        """
        self.research_runner = research_runner
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._check_interval = 60  # Check every minute

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_run_jobs()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            await asyncio.sleep(self._check_interval)

    async def _check_and_run_jobs(self) -> None:
        """Check for jobs that need to run."""
        now = datetime.utcnow()

        for job in self._jobs.values():
            if not job.enabled:
                continue

            if job.next_run and job.next_run <= now:
                logger.info(f"Running scheduled job: {job.name}")
                asyncio.create_task(self._execute_job(job))

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a scheduled job."""
        job.last_run = datetime.utcnow()
        job.run_count += 1

        try:
            result = await self.research_runner(
                objective=job.objective,
                tools=job.tools,
                max_iterations=job.max_iterations
            )

            job.last_status = "completed"
            job.last_session_id = result.get("session_id")
            logger.info(f"Scheduled job {job.name} completed")

            # Send success notification
            await self._send_notification(job, "completed", result)

        except Exception as e:
            job.last_status = "failed"
            logger.error(f"Scheduled job {job.name} failed: {e}")

            # Send failure notification
            await self._send_notification(job, "failed", {"error": str(e)})

        # Calculate next run
        if job.frequency != ScheduleFrequency.ONCE:
            job.next_run = job.calculate_next_run(job.last_run)
        else:
            job.enabled = False  # Disable one-time jobs after execution

    async def _send_notification(
        self,
        job: ScheduledJob,
        status: str,
        data: Dict
    ) -> None:
        """Send job completion notification."""
        # Email notification
        if job.notify_email:
            try:
                from web.notifications import NotificationManager
                manager = NotificationManager()
                # Send email (implementation specific)
            except Exception as e:
                logger.warning(f"Failed to send email: {e}")

        # Webhook notification
        if job.webhook_url:
            try:
                import aiohttp
                payload = {
                    "job_id": job.id,
                    "job_name": job.name,
                    "status": status,
                    "run_time": job.last_run.isoformat() if job.last_run else None,
                    **data
                }
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        job.webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
            except Exception as e:
                logger.warning(f"Failed to send webhook: {e}")

    def create_job(
        self,
        name: str,
        objective: str,
        tools: List[str],
        frequency: ScheduleFrequency = ScheduleFrequency.ONCE,
        max_iterations: int = 5,
        cron_expression: Optional[str] = None,
        start_time: Optional[datetime] = None,
        user_id: Optional[int] = None,
        notify_email: Optional[str] = None,
        webhook_url: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> ScheduledJob:
        """
        Create a new scheduled job.

        Args:
            name: Job name
            objective: Research objective
            tools: Tools to use
            frequency: How often to run
            max_iterations: Research iterations
            cron_expression: Custom cron schedule
            start_time: When to first run
            user_id: Owner user ID
            notify_email: Email for notifications
            webhook_url: Webhook for notifications
            config: Additional configuration

        Returns:
            Created ScheduledJob
        """
        job_id = str(uuid.uuid4())

        # Determine frequency
        freq = frequency
        if cron_expression:
            freq = ScheduleFrequency.CUSTOM

        job = ScheduledJob(
            id=job_id,
            name=name,
            objective=objective,
            tools=tools,
            max_iterations=max_iterations,
            frequency=freq,
            cron_expression=cron_expression,
            user_id=user_id,
            notify_email=notify_email,
            webhook_url=webhook_url,
            config=config or {}
        )

        # Set initial next run time
        if start_time:
            job.next_run = start_time
        else:
            job.next_run = job.calculate_next_run()

        self._jobs[job_id] = job
        logger.info(f"Created scheduled job: {name} ({job_id})")

        return job

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        user_id: Optional[int] = None,
        enabled_only: bool = False
    ) -> List[ScheduledJob]:
        """
        List scheduled jobs.

        Args:
            user_id: Filter by user ID
            enabled_only: Only return enabled jobs

        Returns:
            List of ScheduledJobs
        """
        jobs = list(self._jobs.values())

        if user_id is not None:
            jobs = [j for j in jobs if j.user_id == user_id]

        if enabled_only:
            jobs = [j for j in jobs if j.enabled]

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def update_job(
        self,
        job_id: str,
        **updates
    ) -> Optional[ScheduledJob]:
        """
        Update a scheduled job.

        Args:
            job_id: Job ID to update
            **updates: Fields to update

        Returns:
            Updated job or None if not found
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)

        # Recalculate next run if frequency changed
        if "frequency" in updates or "cron_expression" in updates:
            job.next_run = job.calculate_next_run()

        logger.info(f"Updated scheduled job: {job_id}")
        return job

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a scheduled job.

        Args:
            job_id: Job ID to delete

        Returns:
            True if deleted
        """
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info(f"Deleted scheduled job: {job_id}")
            return True
        return False

    def enable_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Enable a job."""
        job = self._jobs.get(job_id)
        if job:
            job.enabled = True
            if not job.next_run:
                job.next_run = job.calculate_next_run()
            logger.info(f"Enabled scheduled job: {job_id}")
        return job

    def disable_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Disable a job."""
        job = self._jobs.get(job_id)
        if job:
            job.enabled = False
            logger.info(f"Disabled scheduled job: {job_id}")
        return job

    def run_job_now(self, job_id: str) -> Optional[asyncio.Task]:
        """
        Trigger immediate execution of a job.

        Args:
            job_id: Job ID to run

        Returns:
            Running task or None
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        logger.info(f"Manually triggered job: {job_id}")
        return asyncio.create_task(self._execute_job(job))

    def get_upcoming_jobs(self, hours: int = 24) -> List[ScheduledJob]:
        """
        Get jobs scheduled to run within the specified hours.

        Args:
            hours: Number of hours to look ahead

        Returns:
            List of upcoming jobs
        """
        cutoff = datetime.utcnow() + timedelta(hours=hours)
        upcoming = []

        for job in self._jobs.values():
            if job.enabled and job.next_run and job.next_run <= cutoff:
                upcoming.append(job)

        return sorted(upcoming, key=lambda j: j.next_run or datetime.max)


# Singleton instance
_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Get the scheduler instance."""
    global _scheduler
    if _scheduler is None:
        async def placeholder_runner(**kwargs):
            raise NotImplementedError("Scheduler not configured")
        _scheduler = Scheduler(placeholder_runner)
    return _scheduler


def configure_scheduler(research_runner: Callable) -> Scheduler:
    """
    Configure the scheduler with a research runner.

    Args:
        research_runner: Async function to run research

    Returns:
        Configured Scheduler
    """
    global _scheduler
    _scheduler = Scheduler(research_runner)
    return _scheduler
