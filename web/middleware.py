"""
Middleware components for the INSIGHT Web Application.

Provides request/response logging, timing, security headers, and monitoring.
"""

import time
import uuid
from datetime import datetime
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Import logging if available
try:
    from logging_config import get_logger
    logger = get_logger("web.middleware")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all HTTP requests and responses.
    """

    def __init__(self, app: FastAPI, exclude_paths: Optional[list] = None):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
            exclude_paths: Paths to exclude from logging
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")[:100]

        # Log request
        logger.info(
            f"[{request_id}] Request: {request.method} {request.url.path} "
            f"from {client_ip}"
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] Error: {request.method} {request.url.path} "
                f"- {type(e).__name__}: {str(e)} ({duration:.2f}ms)"
            )
            raise

        # Calculate duration
        duration = (time.time() - start_time) * 1000

        # Log response
        log_level = "info" if response.status_code < 400 else "warning" if response.status_code < 500 else "error"
        getattr(logger, log_level)(
            f"[{request_id}] Response: {request.method} {request.url.path} "
            f"- {response.status_code} ({duration:.2f}ms)"
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.2f}ms"

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Content Security Policy (customize as needed)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: ws:;"
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.
    """

    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int = 60,
        burst_size: int = 10
    ):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application
            requests_per_minute: Max requests per minute per IP
            burst_size: Allow burst of requests
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._request_counts: dict = {}
        self._last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit and process request."""
        # Skip rate limiting for internal paths
        if request.url.path.startswith("/api/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Cleanup old entries every minute
        if current_time - self._last_cleanup > 60:
            self._cleanup_old_entries(current_time)
            self._last_cleanup = current_time

        # Check rate limit
        key = f"{client_ip}:{int(current_time // 60)}"

        if key not in self._request_counts:
            self._request_counts[key] = 0

        self._request_counts[key] += 1

        if self._request_counts[key] > self.requests_per_minute + self.burst_size:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"}
            )

        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - self._request_counts[key])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove old rate limit entries."""
        current_minute = int(current_time // 60)
        keys_to_remove = [
            key for key in self._request_counts
            if int(key.split(":")[1]) < current_minute - 1
        ]
        for key in keys_to_remove:
            del self._request_counts[key]


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.
    """

    def __init__(self, app: FastAPI):
        """Initialize metrics middleware."""
        super().__init__(app)
        self.request_count = 0
        self.total_response_time = 0.0
        self.status_counts: dict = {}
        self.endpoint_counts: dict = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics from request/response."""
        start_time = time.time()

        response = await call_next(request)

        # Update metrics
        duration = time.time() - start_time
        self.request_count += 1
        self.total_response_time += duration

        # Count by status
        status_key = f"{response.status_code // 100}xx"
        self.status_counts[status_key] = self.status_counts.get(status_key, 0) + 1

        # Count by endpoint
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in self.endpoint_counts:
            self.endpoint_counts[endpoint] = {"count": 0, "total_time": 0.0}
        self.endpoint_counts[endpoint]["count"] += 1
        self.endpoint_counts[endpoint]["total_time"] += duration

        return response

    def get_metrics(self) -> dict:
        """Get current metrics."""
        avg_response_time = (
            self.total_response_time / self.request_count
            if self.request_count > 0 else 0
        )

        return {
            "total_requests": self.request_count,
            "avg_response_time_ms": avg_response_time * 1000,
            "status_counts": self.status_counts,
            "endpoints": {
                k: {
                    "count": v["count"],
                    "avg_time_ms": (v["total_time"] / v["count"]) * 1000
                }
                for k, v in self.endpoint_counts.items()
            }
        }


def setup_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the application.

    Args:
        app: FastAPI application instance
    """
    # CORS middleware (must be first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=100,
        burst_size=20
    )

    logger.info("Middleware configured successfully")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware for managing request context.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Set up request context."""
        # Add request timestamp
        request.state.timestamp = datetime.utcnow()

        # Add correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id

        response = await call_next(request)

        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id

        return response
