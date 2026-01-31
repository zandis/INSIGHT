"""
Health check module for INSIGHT.

This module provides health checking functionality for:
- External API endpoints
- Internal services
- System resources
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from constants import API_HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_INTERVAL_SECONDS
from logging_config import get_logger

logger = get_logger("health_check")


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    latency_ms: float
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """
    Health checker for external services and internal components.
    """

    def __init__(self) -> None:
        """Initialize the health checker."""
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register("openai", self._check_openai)
        self.register("pubmed", self._check_pubmed)
        self.register("mygene", self._check_mygene)
        self.register("myvariant", self._check_myvariant)
        self.register("disk_space", self._check_disk_space)
        self.register("memory", self._check_memory)

    def register(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult]
    ) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check_fn: Function that returns HealthCheckResult
        """
        with self._lock:
            self._checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")

    def check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.

        Args:
            name: Check name

        Returns:
            Health check result
        """
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message=f"Unknown health check: {name}",
                timestamp=time.time()
            )

        start_time = time.time()
        try:
            result = self._checks[name]()
            result.latency_ms = (time.time() - start_time) * 1000
            result.timestamp = time.time()

            with self._lock:
                self._last_results[name] = result

            return result

        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Health check failed: {e}",
                timestamp=time.time()
            )

            with self._lock:
                self._last_results[name] = result

            return result

    def check_all(self, timeout: float = 30.0) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.

        Args:
            timeout: Maximum time for all checks

        Returns:
            Dictionary of check results
        """
        results = {}

        with ThreadPoolExecutor(max_workers=len(self._checks)) as executor:
            futures = {
                executor.submit(self.check, name): name
                for name in self._checks
            }

            for future in as_completed(futures, timeout=timeout):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=0,
                        message=f"Check timed out or failed: {e}",
                        timestamp=time.time()
                    )

        return results

    def get_overall_status(self) -> Tuple[HealthStatus, Dict[str, HealthCheckResult]]:
        """
        Get overall health status.

        Returns:
            Tuple of (overall status, individual results)
        """
        results = self.check_all()

        # Determine overall status
        statuses = [r.status for r in results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        return overall, results

    def get_last_result(self, name: str) -> Optional[HealthCheckResult]:
        """Get the last result for a check."""
        with self._lock:
            return self._last_results.get(name)

    # ==========================================================================
    # Individual Health Checks
    # ==========================================================================

    def _check_openai(self) -> HealthCheckResult:
        """Check OpenAI API health."""
        import openai

        try:
            # Try a simple models list call
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return HealthCheckResult(
                    name="openai",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    message="OpenAI API key not configured"
                )

            openai.api_key = api_key

            # Simple lightweight API call
            start = time.time()
            models = openai.Model.list()
            latency = (time.time() - start) * 1000

            if models and len(models.data) > 0:
                return HealthCheckResult(
                    name="openai",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message="OpenAI API responding normally",
                    timestamp=time.time(),
                    details={"models_available": len(models.data)}
                )
            else:
                return HealthCheckResult(
                    name="openai",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="OpenAI API responded but no models available",
                    timestamp=time.time()
                )

        except openai.error.RateLimitError:
            return HealthCheckResult(
                name="openai",
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                message="OpenAI rate limit reached",
                timestamp=time.time()
            )

        except openai.error.AuthenticationError:
            return HealthCheckResult(
                name="openai",
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                message="OpenAI authentication failed - invalid API key",
                timestamp=time.time()
            )

        except Exception as e:
            return HealthCheckResult(
                name="openai",
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                message=f"OpenAI API error: {e}",
                timestamp=time.time()
            )

    def _check_pubmed(self) -> HealthCheckResult:
        """Check PubMed API health."""
        try:
            from Bio import Entrez

            Entrez.email = os.environ.get("INSIGHT_EMAIL", "health@check.com")

            start = time.time()
            # Simple search query
            handle = Entrez.esearch(db="pubmed", term="test", retmax=1)
            result = Entrez.read(handle)
            handle.close()
            latency = (time.time() - start) * 1000

            if "Count" in result:
                return HealthCheckResult(
                    name="pubmed",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message="PubMed API responding normally",
                    timestamp=time.time(),
                    details={"test_query_results": result.get("Count", 0)}
                )
            else:
                return HealthCheckResult(
                    name="pubmed",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="PubMed API responded with unexpected format",
                    timestamp=time.time()
                )

        except Exception as e:
            return HealthCheckResult(
                name="pubmed",
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                message=f"PubMed API error: {e}",
                timestamp=time.time()
            )

    def _check_mygene(self) -> HealthCheckResult:
        """Check MyGene API health."""
        try:
            import mygene

            mg = mygene.MyGeneInfo()

            start = time.time()
            # Simple query
            result = mg.query("cdk2", size=1)
            latency = (time.time() - start) * 1000

            if "hits" in result:
                return HealthCheckResult(
                    name="mygene",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message="MyGene API responding normally",
                    timestamp=time.time(),
                    details={"test_query_hits": len(result.get("hits", []))}
                )
            else:
                return HealthCheckResult(
                    name="mygene",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="MyGene API responded with unexpected format",
                    timestamp=time.time()
                )

        except Exception as e:
            return HealthCheckResult(
                name="mygene",
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                message=f"MyGene API error: {e}",
                timestamp=time.time()
            )

    def _check_myvariant(self) -> HealthCheckResult:
        """Check MyVariant API health."""
        try:
            import myvariant

            mv = myvariant.MyVariantInfo()

            start = time.time()
            # Check API metadata
            result = mv.metadata()
            latency = (time.time() - start) * 1000

            if result and "stats" in result:
                return HealthCheckResult(
                    name="myvariant",
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    message="MyVariant API responding normally",
                    timestamp=time.time(),
                    details={"total_variants": result.get("stats", {}).get("total", 0)}
                )
            else:
                return HealthCheckResult(
                    name="myvariant",
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="MyVariant API responded with unexpected format",
                    timestamp=time.time()
                )

        except Exception as e:
            return HealthCheckResult(
                name="myvariant",
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                message=f"MyVariant API error: {e}",
                timestamp=time.time()
            )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        try:
            import shutil

            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024 ** 3)
            used_percent = (used / total) * 100

            if free_gb < 1:
                status = HealthStatus.UNHEALTHY
                message = f"Critical: Only {free_gb:.2f} GB free disk space"
            elif free_gb < 5:
                status = HealthStatus.DEGRADED
                message = f"Warning: Only {free_gb:.2f} GB free disk space"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_gb:.2f} GB free"

            return HealthCheckResult(
                name="disk_space",
                status=status,
                latency_ms=0,
                message=message,
                timestamp=time.time(),
                details={
                    "free_gb": round(free_gb, 2),
                    "used_percent": round(used_percent, 2)
                }
            )

        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message=f"Failed to check disk space: {e}",
                timestamp=time.time()
            )

    def _check_memory(self) -> HealthCheckResult:
        """Check memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            used_percent = memory.percent

            if available_gb < 0.5:
                status = HealthStatus.UNHEALTHY
                message = f"Critical: Only {available_gb:.2f} GB available memory"
            elif available_gb < 1:
                status = HealthStatus.DEGRADED
                message = f"Warning: Only {available_gb:.2f} GB available memory"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory OK: {available_gb:.2f} GB available"

            return HealthCheckResult(
                name="memory",
                status=status,
                latency_ms=0,
                message=message,
                timestamp=time.time(),
                details={
                    "available_gb": round(available_gb, 2),
                    "used_percent": round(used_percent, 2)
                }
            )

        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message=f"Failed to check memory: {e}",
                timestamp=time.time()
            )


class BackgroundHealthMonitor:
    """
    Background health monitoring with periodic checks.
    """

    def __init__(
        self,
        check_interval: int = HEALTH_CHECK_INTERVAL_SECONDS,
        checker: Optional[HealthChecker] = None
    ) -> None:
        """
        Initialize the background monitor.

        Args:
            check_interval: Seconds between checks
            checker: Health checker instance
        """
        self.check_interval = check_interval
        self.checker = checker or HealthChecker()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[Dict[str, HealthCheckResult]], None]] = []

    def add_callback(
        self,
        callback: Callable[[Dict[str, HealthCheckResult]], None]
    ) -> None:
        """Add a callback to be called after each check cycle."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Background health monitor started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Background health monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                results = self.checker.check_all()

                # Log any unhealthy services
                for name, result in results.items():
                    if result.status == HealthStatus.UNHEALTHY:
                        logger.warning(
                            f"Health check failed: {name} - {result.message}"
                        )
                    elif result.status == HealthStatus.DEGRADED:
                        logger.warning(
                            f"Health check degraded: {name} - {result.message}"
                        )

                # Call registered callbacks
                for callback in self._callbacks:
                    try:
                        callback(results)
                    except Exception as e:
                        logger.error(f"Health check callback failed: {e}")

            except Exception as e:
                logger.error(f"Health check cycle failed: {e}")

            time.sleep(self.check_interval)


def get_health_checker() -> HealthChecker:
    """Get a health checker instance."""
    return HealthChecker()


def quick_health_check() -> bool:
    """
    Perform a quick health check of critical services.

    Returns:
        True if all critical services are healthy
    """
    checker = HealthChecker()
    critical_services = ["openai"]

    for service in critical_services:
        result = checker.check(service)
        if result.status == HealthStatus.UNHEALTHY:
            logger.error(f"Critical service unhealthy: {service} - {result.message}")
            return False

    return True
