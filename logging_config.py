"""
Comprehensive logging configuration for INSIGHT.

This module provides structured logging with multiple handlers,
formatters, and log levels for different components.
"""

import logging
import os
import sys
from datetime import datetime
from enum import Enum
from functools import wraps
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from colorama import Fore, Style, init as colorama_init

from constants import LOG_FORMAT, LOG_DATE_FORMAT, DEFAULT_LOG_LEVEL


# Initialize colorama for cross-platform color support
colorama_init(autoreset=True)

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    RESET = Style.RESET_ALL

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        color = self.COLORS.get(record.levelno, self.RESET)

        # Format the base message
        formatted = super().format(record)

        # Add color to the level name
        formatted = formatted.replace(
            record.levelname,
            f"{color}{record.levelname}{self.RESET}"
        )

        return formatted


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON-like log entries."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a structured entry."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return str(log_data)


class InsightLogger:
    """
    Centralized logger for the INSIGHT application.

    Provides multiple logging channels with different formatters
    and handlers for console, file, and structured logging.
    """

    _instance: Optional['InsightLogger'] = None
    _initialized: bool = False

    def __new__(cls) -> 'InsightLogger':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logger if not already initialized."""
        if InsightLogger._initialized:
            return

        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        self._setup_root_logger()
        self._setup_component_loggers()

        InsightLogger._initialized = True

    def _setup_root_logger(self) -> None:
        """Set up the root logger with console and file handlers."""
        self.root_logger = logging.getLogger("insight")
        self.root_logger.setLevel(logging.DEBUG)

        # Prevent propagation to avoid duplicate logs
        self.root_logger.propagate = False

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, DEFAULT_LOG_LEVEL))
        console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, LOG_DATE_FORMAT))

        # Rotating file handler for general logs
        file_handler = RotatingFileHandler(
            self.log_dir / "insight.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

        # Error file handler
        error_handler = RotatingFileHandler(
            self.log_dir / "error.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

        # Add handlers
        self.root_logger.addHandler(console_handler)
        self.root_logger.addHandler(file_handler)
        self.root_logger.addHandler(error_handler)

    def _setup_component_loggers(self) -> None:
        """Set up loggers for specific components."""
        components = ["agents", "api", "utils", "main", "interface", "cache", "metrics"]

        for component in components:
            logger = logging.getLogger(f"insight.{component}")
            logger.setLevel(logging.DEBUG)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific component.

        Args:
            name: The name of the component (e.g., 'agents', 'api')

        Returns:
            A configured logger instance
        """
        return logging.getLogger(f"insight.{name}")

    def set_level(self, level: Union[str, LogLevel]) -> None:
        """
        Set the logging level for all handlers.

        Args:
            level: The logging level (e.g., 'DEBUG', 'INFO')
        """
        if isinstance(level, LogLevel):
            level = level.value

        log_level = getattr(logging, level.upper())

        for handler in self.root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)

    def enable_debug_mode(self) -> None:
        """Enable debug mode with verbose logging."""
        self.set_level(LogLevel.DEBUG)
        self.root_logger.debug("Debug mode enabled")

    def disable_debug_mode(self) -> None:
        """Disable debug mode, return to INFO level."""
        self.set_level(LogLevel.INFO)
        self.root_logger.info("Debug mode disabled")


# Create singleton instance
_logger_instance = InsightLogger()


def get_logger(name: str = "main") -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        name: The name of the component

    Returns:
        A configured logger instance
    """
    return _logger_instance.get_logger(name)


def log_function_call(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    Decorator to log function entry and exit.

    Args:
        logger: Optional logger instance. If not provided, uses the main logger.

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger
            if logger is None:
                logger = get_logger("functions")

            func_name = func.__name__
            logger.debug(f"Entering {func_name}")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Exception in {func_name}: {e}", exc_info=True)
                raise

        return wrapper  # type: ignore

    return decorator


def log_api_call(api_name: str) -> Callable[[F], F]:
    """
    Decorator to log API calls with timing.

    Args:
        api_name: Name of the API being called

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger("api")
            start_time = datetime.now()

            logger.info(f"API call to {api_name} started")

            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"API call to {api_name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.error(f"API call to {api_name} failed after {elapsed:.2f}s: {e}")
                raise

        return wrapper  # type: ignore

    return decorator


def log_performance(threshold_seconds: float = 1.0) -> Callable[[F], F]:
    """
    Decorator to log slow function calls.

    Args:
        threshold_seconds: Log warning if execution exceeds this duration

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger("performance")
            start_time = datetime.now()

            result = func(*args, **kwargs)

            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > threshold_seconds:
                logger.warning(
                    f"Slow function: {func.__name__} took {elapsed:.2f}s "
                    f"(threshold: {threshold_seconds}s)"
                )

            return result

        return wrapper  # type: ignore

    return decorator


class LogContext:
    """Context manager for scoped logging with additional context."""

    def __init__(self, logger: logging.Logger, context_name: str, **extra: Any) -> None:
        """
        Initialize the log context.

        Args:
            logger: The logger to use
            context_name: Name of the context (e.g., 'task_execution')
            **extra: Additional context data to include in logs
        """
        self.logger = logger
        self.context_name = context_name
        self.extra = extra
        self.start_time: Optional[datetime] = None

    def __enter__(self) -> 'LogContext':
        """Enter the context and log start."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.context_name}", extra={"extra_data": self.extra})
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the context and log completion or error."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        if exc_type is None:
            self.logger.info(
                f"Completed {self.context_name} in {elapsed:.2f}s",
                extra={"extra_data": {**self.extra, "duration": elapsed}}
            )
        else:
            self.logger.error(
                f"Failed {self.context_name} after {elapsed:.2f}s: {exc_val}",
                extra={"extra_data": {**self.extra, "duration": elapsed, "error": str(exc_val)}}
            )

        return False  # Don't suppress exceptions


# Suppress noisy third-party loggers
def configure_third_party_logging() -> None:
    """Configure logging levels for third-party libraries."""
    noisy_loggers = [
        "llama_index",
        "openai",
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Initialize third-party logging configuration
configure_third_party_logging()
