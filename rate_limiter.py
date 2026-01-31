"""
Rate limiting and circuit breaker module for INSIGHT.

This module provides:
- Token bucket rate limiting
- Sliding window rate limiting
- Circuit breaker pattern for external APIs
- Retry with exponential backoff
"""

import functools
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Dict, Optional, TypeVar, Union

from constants import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_HALF_OPEN_REQUESTS,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    MAX_RETRIES,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    RATE_LIMIT_TOKENS_PER_MINUTE,
    RETRY_BASE_DELAY,
    RETRY_EXPONENTIAL_BASE,
    RETRY_MAX_DELAY,
)
from logging_config import get_logger

logger = get_logger("rate_limiter")

F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Rate Limiting
# =============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows bursts up to bucket capacity while maintaining
    average rate over time.
    """

    def __init__(
        self,
        rate: float,
        capacity: Optional[float] = None,
        name: str = "default"
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            rate: Tokens per second to add
            capacity: Maximum bucket capacity (defaults to rate)
            name: Name for logging
        """
        self.rate = rate
        self.capacity = capacity or rate
        self.name = name
        self._tokens = self.capacity
        self._last_update = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    def acquire(self, tokens: float = 1.0, blocking: bool = True) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            blocking: Whether to wait if tokens unavailable

        Returns:
            True if tokens acquired, False if non-blocking and unavailable
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if not blocking:
                return False

            # Calculate wait time
            wait_time = (tokens - self._tokens) / self.rate
            logger.debug(f"Rate limiter {self.name}: waiting {wait_time:.2f}s")

        # Wait outside the lock
        time.sleep(wait_time)

        with self._lock:
            self._refill()
            self._tokens -= tokens
            return True

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking token acquisition."""
        return self.acquire(tokens, blocking=False)


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests in a sliding time window for more
    accurate rate limiting.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        name: str = "default"
    ) -> None:
        """
        Initialize the sliding window limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window duration in seconds
            name: Name for logging
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.name = name
        self._requests: Deque[float] = deque()
        self._lock = threading.Lock()

    def _cleanup(self) -> None:
        """Remove expired requests from the window."""
        cutoff = time.time() - self.window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire permission to make a request.

        Args:
            blocking: Whether to wait if at limit

        Returns:
            True if request allowed
        """
        with self._lock:
            self._cleanup()

            if len(self._requests) < self.max_requests:
                self._requests.append(time.time())
                return True

            if not blocking:
                return False

            # Calculate wait time until oldest request expires
            wait_time = self._requests[0] + self.window_seconds - time.time()

        if wait_time > 0:
            logger.debug(f"Rate limiter {self.name}: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        with self._lock:
            self._cleanup()
            self._requests.append(time.time())
            return True

    def try_acquire(self) -> bool:
        """Non-blocking request acquisition."""
        return self.acquire(blocking=False)

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        with self._lock:
            self._cleanup()
            return max(0, self.max_requests - len(self._requests))


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    state: CircuitState
    failures: int
    successes: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    half_open_successes: int


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when
    a service is unhealthy.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        half_open_requests: int = CIRCUIT_BREAKER_HALF_OPEN_REQUESTS,
        excluded_exceptions: Optional[tuple] = None
    ) -> None:
        """
        Initialize the circuit breaker.

        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening
            recovery_timeout: Seconds before trying half-open
            half_open_requests: Successful requests to close
            excluded_exceptions: Exceptions that don't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        self.excluded_exceptions = excluded_exceptions or ()

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._half_open_successes = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transitions."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_successes = 0
                        logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
            return self._state

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._successes += 1
            self._last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_requests:
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failures = max(0, self._failures - 1)

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if exception should be excluded
        if isinstance(exception, self.excluded_exceptions):
            return

        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery, go back to open
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN")
            elif self._state == CircuitState.CLOSED:
                if self._failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit {self.name}: CLOSED -> OPEN "
                        f"(failures: {self._failures})"
                    )

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If call fails
        """
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit {self.name} is open, failing fast"
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def get_stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return CircuitStats(
                state=self._state,
                failures=self._failures,
                successes=self._successes,
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
                half_open_successes=self._half_open_successes
            )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._half_open_successes = 0
        logger.info(f"Circuit {self.name}: manually reset to CLOSED")


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# =============================================================================
# Retry Logic
# =============================================================================

def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
    exponential_base: float = RETRY_EXPONENTIAL_BASE,
    jitter: bool = True,
    retryable_exceptions: Optional[tuple] = None
) -> Callable[[F], F]:
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter
        retryable_exceptions: Exceptions that trigger retry

    Returns:
        Decorated function
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{max_retries + 1} attempts: {e}"
                        )
                        raise

                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Function {func.__name__} attempt {attempt + 1} "
                        f"failed: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Combined Limiter Manager
# =============================================================================

class RateLimiterManager:
    """
    Manages rate limiters and circuit breakers for different APIs.
    """

    _instance: Optional['RateLimiterManager'] = None

    def __new__(cls) -> 'RateLimiterManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the manager."""
        if self._initialized:
            return

        # Rate limiters
        self._request_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self._token_limiters: Dict[str, TokenBucketRateLimiter] = {}

        # Circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Set up default limiters
        self._setup_default_limiters()

        self._initialized = True
        logger.info("Rate limiter manager initialized")

    def _setup_default_limiters(self) -> None:
        """Set up default rate limiters for known APIs."""
        # OpenAI rate limiting
        self._request_limiters["openai"] = SlidingWindowRateLimiter(
            max_requests=RATE_LIMIT_REQUESTS_PER_MINUTE,
            window_seconds=60,
            name="openai_requests"
        )
        self._token_limiters["openai"] = TokenBucketRateLimiter(
            rate=RATE_LIMIT_TOKENS_PER_MINUTE / 60,
            capacity=RATE_LIMIT_TOKENS_PER_MINUTE,
            name="openai_tokens"
        )

        # PubMed rate limiting (3 requests/second for registered users)
        self._request_limiters["pubmed"] = SlidingWindowRateLimiter(
            max_requests=3,
            window_seconds=1,
            name="pubmed"
        )

        # MyGene rate limiting
        self._request_limiters["mygene"] = SlidingWindowRateLimiter(
            max_requests=10,
            window_seconds=1,
            name="mygene"
        )

        # MyVariant rate limiting
        self._request_limiters["myvariant"] = SlidingWindowRateLimiter(
            max_requests=10,
            window_seconds=1,
            name="myvariant"
        )

        # Circuit breakers
        for api in ["openai", "pubmed", "mygene", "myvariant"]:
            self._circuit_breakers[api] = CircuitBreaker(
                name=api,
                failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                recovery_timeout=CIRCUIT_BREAKER_RECOVERY_TIMEOUT
            )

    def get_request_limiter(self, api: str) -> Optional[SlidingWindowRateLimiter]:
        """Get request rate limiter for an API."""
        return self._request_limiters.get(api)

    def get_token_limiter(self, api: str) -> Optional[TokenBucketRateLimiter]:
        """Get token rate limiter for an API."""
        return self._token_limiters.get(api)

    def get_circuit_breaker(self, api: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for an API."""
        return self._circuit_breakers.get(api)

    def acquire(
        self,
        api: str,
        tokens: Optional[int] = None,
        blocking: bool = True
    ) -> bool:
        """
        Acquire rate limit permission for an API call.

        Args:
            api: API name
            tokens: Number of tokens (for token-based limiting)
            blocking: Whether to wait if at limit

        Returns:
            True if acquired, False if non-blocking and at limit
        """
        # Check circuit breaker first
        circuit = self._circuit_breakers.get(api)
        if circuit and circuit.state == CircuitState.OPEN:
            logger.warning(f"Circuit breaker for {api} is open")
            return False

        # Request rate limiting
        request_limiter = self._request_limiters.get(api)
        if request_limiter:
            if not request_limiter.acquire(blocking=blocking):
                return False

        # Token rate limiting (if applicable)
        if tokens:
            token_limiter = self._token_limiters.get(api)
            if token_limiter:
                if not token_limiter.acquire(tokens, blocking=blocking):
                    return False

        return True

    def record_success(self, api: str) -> None:
        """Record a successful API call."""
        circuit = self._circuit_breakers.get(api)
        if circuit:
            circuit._record_success()

    def record_failure(self, api: str, exception: Exception) -> None:
        """Record a failed API call."""
        circuit = self._circuit_breakers.get(api)
        if circuit:
            circuit._record_failure(exception)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all limiters and breakers."""
        stats: Dict[str, Any] = {
            "request_limiters": {},
            "circuit_breakers": {}
        }

        for name, limiter in self._request_limiters.items():
            stats["request_limiters"][name] = {
                "remaining": limiter.get_remaining(),
                "max_requests": limiter.max_requests,
                "window_seconds": limiter.window_seconds
            }

        for name, circuit in self._circuit_breakers.items():
            circuit_stats = circuit.get_stats()
            stats["circuit_breakers"][name] = {
                "state": circuit_stats.state.value,
                "failures": circuit_stats.failures,
                "successes": circuit_stats.successes
            }

        return stats


def get_rate_limiter_manager() -> RateLimiterManager:
    """Get the global rate limiter manager instance."""
    return RateLimiterManager()


# =============================================================================
# Decorators
# =============================================================================

def rate_limited(api: str, tokens: Optional[int] = None) -> Callable[[F], F]:
    """
    Decorator to apply rate limiting to a function.

    Args:
        api: API name for rate limiting
        tokens: Number of tokens to consume

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_rate_limiter_manager()
            manager.acquire(api, tokens=tokens, blocking=True)

            try:
                result = func(*args, **kwargs)
                manager.record_success(api)
                return result
            except Exception as e:
                manager.record_failure(api, e)
                raise

        return wrapper  # type: ignore

    return decorator


def with_circuit_breaker(api: str) -> Callable[[F], F]:
    """
    Decorator to apply circuit breaker to a function.

    Args:
        api: API name for circuit breaker

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_rate_limiter_manager()
            circuit = manager.get_circuit_breaker(api)

            if circuit:
                return circuit.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
