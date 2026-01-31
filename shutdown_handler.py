"""
Graceful shutdown handling for INSIGHT.

This module provides:
- Signal handling for clean shutdown
- State persistence on shutdown
- Resource cleanup
"""

import atexit
import signal
import sys
import threading
from typing import Any, Callable, Dict, List, Optional

from logging_config import get_logger

logger = get_logger("shutdown")


class ShutdownHandler:
    """
    Handles graceful shutdown of the application.

    Features:
    - Signal handling (SIGINT, SIGTERM)
    - Cleanup callbacks
    - State persistence
    - Timeout protection
    """

    _instance: Optional['ShutdownHandler'] = None
    _initialized: bool = False

    def __new__(cls) -> 'ShutdownHandler':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the shutdown handler."""
        if ShutdownHandler._initialized:
            return

        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._shutdown_requested = threading.Event()
        self._shutdown_complete = threading.Event()
        self._lock = threading.Lock()
        self._shutdown_timeout = 30  # seconds

        # Register signal handlers
        self._register_signal_handlers()

        # Register atexit handler
        atexit.register(self._atexit_handler)

        ShutdownHandler._initialized = True
        logger.info("Shutdown handler initialized")

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # SIGHUP for Unix systems
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, self._signal_handler)

            logger.debug("Signal handlers registered")
        except Exception as e:
            logger.warning(f"Could not register all signal handlers: {e}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name}, initiating graceful shutdown...")

        # Prevent multiple shutdown attempts
        if self._shutdown_requested.is_set():
            logger.warning("Shutdown already in progress")
            return

        self._shutdown_requested.set()

        # Run cleanup in a thread with timeout
        cleanup_thread = threading.Thread(target=self._run_cleanup)
        cleanup_thread.start()
        cleanup_thread.join(timeout=self._shutdown_timeout)

        if cleanup_thread.is_alive():
            logger.error("Cleanup timed out, forcing exit")

        self._shutdown_complete.set()
        logger.info("Shutdown complete")

        # Exit gracefully
        sys.exit(0)

    def _atexit_handler(self) -> None:
        """Handle normal program exit."""
        if not self._shutdown_requested.is_set():
            logger.debug("Normal exit, running cleanup...")
            self._run_cleanup()

    def _run_cleanup(self) -> None:
        """Execute all cleanup callbacks."""
        with self._lock:
            callbacks = list(self._cleanup_callbacks)

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """
        Register a cleanup callback.

        Callbacks are executed in reverse registration order (LIFO).

        Args:
            callback: Function to call during cleanup
        """
        with self._lock:
            self._cleanup_callbacks.insert(0, callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")

    def unregister_cleanup(self, callback: Callable[[], None]) -> bool:
        """
        Unregister a cleanup callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        with self._lock:
            try:
                self._cleanup_callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested.is_set()

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for shutdown to be requested.

        Args:
            timeout: Maximum time to wait (None for indefinite)

        Returns:
            True if shutdown was requested, False if timeout
        """
        return self._shutdown_requested.wait(timeout=timeout)

    def request_shutdown(self) -> None:
        """Programmatically request shutdown."""
        logger.info("Shutdown requested programmatically")
        self._shutdown_requested.set()
        self._run_cleanup()
        self._shutdown_complete.set()


# =============================================================================
# Convenience Functions
# =============================================================================

_handler: Optional[ShutdownHandler] = None


def get_shutdown_handler() -> ShutdownHandler:
    """Get the global shutdown handler instance."""
    global _handler
    if _handler is None:
        _handler = ShutdownHandler()
    return _handler


def register_cleanup(callback: Callable[[], None]) -> None:
    """
    Register a cleanup callback for shutdown.

    Args:
        callback: Function to call during cleanup
    """
    get_shutdown_handler().register_cleanup(callback)


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return get_shutdown_handler().is_shutdown_requested()


class CleanupContext:
    """
    Context manager for automatic cleanup registration.

    Usage:
        with CleanupContext(cleanup_function):
            # Do work...
    """

    def __init__(self, cleanup_fn: Callable[[], None]) -> None:
        """
        Initialize the context.

        Args:
            cleanup_fn: Function to call on cleanup
        """
        self.cleanup_fn = cleanup_fn

    def __enter__(self) -> 'CleanupContext':
        """Register cleanup on enter."""
        register_cleanup(self.cleanup_fn)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Unregister cleanup on exit."""
        get_shutdown_handler().unregister_cleanup(self.cleanup_fn)
        return False


# =============================================================================
# State Preservation
# =============================================================================

class StatePreserver:
    """
    Preserves application state on shutdown.
    """

    def __init__(self) -> None:
        """Initialize the state preserver."""
        self._state_savers: Dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()

        # Register with shutdown handler
        register_cleanup(self._save_all_states)

    def register_state_saver(
        self,
        name: str,
        saver_fn: Callable[[], None]
    ) -> None:
        """
        Register a state saving function.

        Args:
            name: Identifier for this state saver
            saver_fn: Function that saves state
        """
        with self._lock:
            self._state_savers[name] = saver_fn
        logger.debug(f"Registered state saver: {name}")

    def _save_all_states(self) -> None:
        """Save all registered states."""
        logger.info("Saving application state...")

        with self._lock:
            savers = dict(self._state_savers)

        for name, saver in savers.items():
            try:
                logger.debug(f"Saving state: {name}")
                saver()
            except Exception as e:
                logger.error(f"Failed to save state '{name}': {e}")

        logger.info("State saving complete")


# Global state preserver
_state_preserver: Optional[StatePreserver] = None


def get_state_preserver() -> StatePreserver:
    """Get the global state preserver instance."""
    global _state_preserver
    if _state_preserver is None:
        _state_preserver = StatePreserver()
    return _state_preserver


def register_state_saver(name: str, saver_fn: Callable[[], None]) -> None:
    """
    Register a state saving function.

    Args:
        name: Identifier for this state saver
        saver_fn: Function that saves state
    """
    get_state_preserver().register_state_saver(name, saver_fn)
