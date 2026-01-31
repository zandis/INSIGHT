"""
Metrics and monitoring module for INSIGHT.

This module provides comprehensive tracking of:
- API calls and their costs
- Performance metrics
- Resource usage
- Progress tracking
"""

import gc
import json
import os
import psutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from constants import MEMORY_CHECK_INTERVAL_SECONDS, MAX_MEMORY_USAGE_MB
from logging_config import get_logger

logger = get_logger("metrics")

F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# OpenAI Pricing (as of 2024 - update as needed)
# =============================================================================

PRICING = {
    # GPT-4 models
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},

    # GPT-3.5 models
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.001, "output": 0.002},

    # Legacy models
    "text-davinci-003": {"input": 0.02, "output": 0.02},

    # Embedding models
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}

# Per 1000 tokens
def calculate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """
    Calculate the cost of an API call.

    Args:
        model: The model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = PRICING.get(model, {"input": 0.01, "output": 0.01})
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return input_cost + output_cost


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class APICallMetric:
    """Metrics for a single API call."""
    timestamp: float
    api_name: str
    model: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        """Calculate cost after initialization."""
        if self.model and self.cost_usd == 0.0:
            self.cost_usd = calculate_cost(
                self.model, self.input_tokens, self.output_tokens
            )


@dataclass
class TaskMetric:
    """Metrics for a task execution."""
    task_id: int
    task_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    api_calls: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    results_count: int = 0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class SessionMetrics:
    """Aggregate metrics for a session."""
    session_id: str
    start_time: float
    objective: str
    total_api_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    peak_memory_mb: float = 0.0


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Centralized metrics collection for the INSIGHT application.

    Thread-safe singleton that collects and aggregates metrics
    from all components.
    """

    _instance: Optional['MetricsCollector'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'MetricsCollector':
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        if self._initialized:
            return

        self._api_calls: List[APICallMetric] = []
        self._tasks: Dict[int, TaskMetric] = {}
        self._session: Optional[SessionMetrics] = None
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._initialized = True

        logger.info("Metrics collector initialized")

    def start_session(self, objective: str) -> str:
        """
        Start a new metrics session.

        Args:
            objective: The research objective

        Returns:
            Session ID
        """
        session_id = f"session_{int(time.time())}"
        self._session = SessionMetrics(
            session_id=session_id,
            start_time=time.time(),
            objective=objective
        )
        logger.info(f"Started metrics session: {session_id}")
        return session_id

    def record_api_call(
        self,
        api_name: str,
        model: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Record an API call metric.

        Args:
            api_name: Name of the API (openai, pubmed, mygene, etc.)
            model: Model name for OpenAI calls
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Call latency in milliseconds
            success: Whether the call succeeded
            error: Error message if failed
        """
        metric = APICallMetric(
            timestamp=time.time(),
            api_name=api_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error
        )

        with self._lock:
            self._api_calls.append(metric)
            self._counters[f"api_calls_{api_name}"] += 1

            if self._session:
                self._session.total_api_calls += 1
                self._session.total_tokens += input_tokens + output_tokens
                self._session.total_cost_usd += metric.cost_usd

            # Update histograms
            self._histograms[f"latency_{api_name}"].append(latency_ms)

        logger.debug(
            f"API call recorded: {api_name} "
            f"(tokens: {input_tokens}+{output_tokens}, "
            f"latency: {latency_ms:.2f}ms, "
            f"cost: ${metric.cost_usd:.6f})"
        )

    def start_task(self, task_id: int, task_name: str) -> None:
        """
        Record the start of a task.

        Args:
            task_id: Unique task identifier
            task_name: Task description
        """
        metric = TaskMetric(
            task_id=task_id,
            task_name=task_name,
            start_time=time.time()
        )

        with self._lock:
            self._tasks[task_id] = metric

        logger.info(f"Task started: {task_id} - {task_name[:50]}...")

    def complete_task(
        self,
        task_id: int,
        success: bool = True,
        results_count: int = 0
    ) -> None:
        """
        Record task completion.

        Args:
            task_id: Task identifier
            success: Whether the task succeeded
            results_count: Number of results generated
        """
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Unknown task completed: {task_id}")
                return

            task = self._tasks[task_id]
            task.end_time = time.time()
            task.status = "completed" if success else "failed"
            task.results_count = results_count

            if self._session:
                if success:
                    self._session.tasks_completed += 1
                else:
                    self._session.tasks_failed += 1

        logger.info(
            f"Task completed: {task_id} - "
            f"status={task.status}, "
            f"duration={task.duration_seconds:.2f}s, "
            f"results={results_count}"
        )

    def record_cache_access(self, hit: bool) -> None:
        """
        Record a cache access.

        Args:
            hit: True for cache hit, False for miss
        """
        with self._lock:
            if hit:
                self._counters["cache_hits"] += 1
                if self._session:
                    self._session.cache_hits += 1
            else:
                self._counters["cache_misses"] += 1
                if self._session:
                    self._session.cache_misses += 1

    def update_memory_usage(self) -> float:
        """
        Update current memory usage metric.

        Returns:
            Current memory usage in MB
        """
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)

            with self._lock:
                self._gauges["memory_mb"] = memory_mb
                if self._session:
                    self._session.peak_memory_mb = max(
                        self._session.peak_memory_mb, memory_mb
                    )

            return memory_mb
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return 0.0

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a named gauge value."""
        with self._lock:
            self._gauges[name] = value

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.

        Returns:
            Dictionary with metric summary
        """
        with self._lock:
            summary = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "api_calls_count": len(self._api_calls),
                "tasks_count": len(self._tasks),
            }

            # Add session metrics if available
            if self._session:
                summary["session"] = {
                    "id": self._session.session_id,
                    "objective": self._session.objective[:100],
                    "duration_seconds": time.time() - self._session.start_time,
                    "total_api_calls": self._session.total_api_calls,
                    "total_tokens": self._session.total_tokens,
                    "total_cost_usd": round(self._session.total_cost_usd, 4),
                    "tasks_completed": self._session.tasks_completed,
                    "tasks_failed": self._session.tasks_failed,
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                    "peak_memory_mb": round(self._session.peak_memory_mb, 2),
                }

            # Add histogram summaries
            summary["latency_summaries"] = {}
            for name, values in self._histograms.items():
                if values:
                    summary["latency_summaries"][name] = {
                        "count": len(values),
                        "avg_ms": round(sum(values) / len(values), 2),
                        "min_ms": round(min(values), 2),
                        "max_ms": round(max(values), 2),
                    }

            return summary

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self._session:
            total = self._session.cache_hits + self._session.cache_misses
            if total > 0:
                return round(self._session.cache_hits / total, 4)
        return 0.0

    def export_to_json(self, path: str) -> None:
        """
        Export metrics to a JSON file.

        Args:
            path: File path for export
        """
        summary = self.get_summary()

        # Add detailed API calls
        with self._lock:
            summary["api_calls"] = [
                {
                    "timestamp": m.timestamp,
                    "api": m.api_name,
                    "model": m.model,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "latency_ms": m.latency_ms,
                    "success": m.success,
                    "cost_usd": m.cost_usd,
                }
                for m in self._api_calls
            ]

            summary["tasks"] = [
                {
                    "id": t.task_id,
                    "name": t.task_name,
                    "status": t.status,
                    "duration_seconds": t.duration_seconds,
                    "results_count": t.results_count,
                }
                for t in self._tasks.values()
            ]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Metrics exported to {path}")

    def print_summary(self) -> None:
        """Print a formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("INSIGHT SESSION METRICS SUMMARY")
        print("=" * 60)

        if "session" in summary:
            session = summary["session"]
            print(f"\nSession ID: {session['id']}")
            print(f"Duration: {session['duration_seconds']:.2f} seconds")
            print(f"\nAPI Usage:")
            print(f"  Total Calls: {session['total_api_calls']}")
            print(f"  Total Tokens: {session['total_tokens']:,}")
            print(f"  Total Cost: ${session['total_cost_usd']:.4f}")
            print(f"\nTasks:")
            print(f"  Completed: {session['tasks_completed']}")
            print(f"  Failed: {session['tasks_failed']}")
            print(f"\nPerformance:")
            print(f"  Cache Hit Rate: {session['cache_hit_rate']:.2%}")
            print(f"  Peak Memory: {session['peak_memory_mb']:.2f} MB")

        if summary.get("latency_summaries"):
            print(f"\nLatency (ms):")
            for name, stats in summary["latency_summaries"].items():
                print(f"  {name}: avg={stats['avg_ms']}, "
                      f"min={stats['min_ms']}, max={stats['max_ms']}")

        print("=" * 60 + "\n")

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._api_calls.clear()
            self._tasks.clear()
            self._session = None
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

        logger.info("Metrics reset")


# =============================================================================
# Progress Tracker
# =============================================================================

class ProgressTracker:
    """
    Track and display progress of long-running operations.
    """

    def __init__(self, total: int, description: str = "Progress") -> None:
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Description of the operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def update(self, amount: int = 1) -> None:
        """
        Update progress.

        Args:
            amount: Amount to increment
        """
        with self._lock:
            self.current = min(self.current + amount, self.total)

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self._lock:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0

            return {
                "description": self.description,
                "current": self.current,
                "total": self.total,
                "percentage": (self.current / self.total * 100) if self.total > 0 else 0,
                "elapsed_seconds": elapsed,
                "rate_per_second": rate,
                "remaining_seconds": remaining,
            }

    def print_progress(self) -> None:
        """Print progress bar to console."""
        progress = self.get_progress()
        percentage = progress["percentage"]
        bar_length = 40
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(
            f"\r{self.description}: [{bar}] "
            f"{percentage:.1f}% ({self.current}/{self.total}) "
            f"ETA: {progress['remaining_seconds']:.0f}s",
            end="", flush=True
        )


# =============================================================================
# Memory Monitor
# =============================================================================

class MemoryMonitor:
    """
    Background memory monitoring with alerts.
    """

    def __init__(
        self,
        check_interval: int = MEMORY_CHECK_INTERVAL_SECONDS,
        max_memory_mb: int = MAX_MEMORY_USAGE_MB
    ) -> None:
        """
        Initialize memory monitor.

        Args:
            check_interval: Seconds between checks
            max_memory_mb: Memory threshold for alerts
        """
        self.check_interval = check_interval
        self.max_memory_mb = max_memory_mb
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._metrics = get_metrics()

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Memory monitor started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Memory monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            memory_mb = self._metrics.update_memory_usage()

            if memory_mb > self.max_memory_mb:
                logger.warning(
                    f"High memory usage: {memory_mb:.2f} MB "
                    f"(threshold: {self.max_memory_mb} MB)"
                )
                # Try to free some memory
                gc.collect()

            time.sleep(self.check_interval)

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}


# =============================================================================
# Convenience Functions
# =============================================================================

def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return MetricsCollector()


def track_api_call(api_name: str, model: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to track API calls.

    Args:
        api_name: Name of the API
        model: Model name for cost calculation

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_metrics()
            start_time = time.time()
            success = True
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                metrics.record_api_call(
                    api_name=api_name,
                    model=model,
                    latency_ms=latency_ms,
                    success=success,
                    error=error
                )

        return wrapper  # type: ignore

    return decorator
