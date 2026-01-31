"""
Caching module for INSIGHT.

This module provides caching functionality for embeddings, API results,
and other expensive computations to improve performance.
"""

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from constants import (
    EMBEDDING_CACHE_MAX_SIZE,
    EMBEDDING_CACHE_TTL_SECONDS,
    RESULT_CACHE_MAX_SIZE,
)
from logging_config import get_logger

logger = get_logger("cache")

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    value: T
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if this entry has expired."""
        return (time.time() - self.created_at) > ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache(Generic[T]):
    """
    Thread-safe LRU (Least Recently Used) cache implementation.

    Features:
    - Configurable max size
    - TTL (time-to-live) support
    - Thread-safe operations
    - Statistics tracking
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600,
        name: str = "cache"
    ) -> None:
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries in seconds
            name: Name for logging purposes
        """
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._name = name
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired(self._ttl_seconds):
                del self._cache[key]
                self._misses += 1
                logger.debug(f"Cache miss (expired): {self._name}[{key[:20]}...]")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            self._hits += 1
            logger.debug(f"Cache hit: {self._name}[{key[:20]}...]")
            return entry.value

    def set(self, key: str, value: T) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
        """
        with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]

            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
                logger.debug(f"Cache eviction: {self._name}[{oldest_key[:20]}...]")

            # Add new entry
            entry = CacheEntry(value=value)
            try:
                # Estimate size using JSON (safer than pickle)
                entry.size_bytes = len(json.dumps(value, default=str).encode())
            except (TypeError, ValueError):
                entry.size_bytes = 0

            self._cache[key] = entry
            logger.debug(f"Cache set: {self._name}[{key[:20]}...]")

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The cache key

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            logger.info(f"Cache cleared: {self._name}")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self._ttl_seconds)
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries from {self._name}")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "name": self._name,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "ttl_seconds": self._ttl_seconds,
            }

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache (without updating access time)."""
        with self._lock:
            if key not in self._cache:
                return False
            return not self._cache[key].is_expired(self._ttl_seconds)


class EmbeddingCache:
    """
    Specialized cache for text embeddings.

    Features:
    - Hash-based key generation from text
    - Persistent storage option
    - Deduplication of identical texts
    """

    def __init__(
        self,
        max_size: int = EMBEDDING_CACHE_MAX_SIZE,
        ttl_seconds: float = EMBEDDING_CACHE_TTL_SECONDS,
        persist_path: Optional[str] = None
    ) -> None:
        """
        Initialize the embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: TTL for cache entries
            persist_path: Optional path for persistent storage
        """
        self._cache = LRUCache[List[float]](
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            name="embeddings"
        )
        self._persist_path = persist_path

        if persist_path:
            self._load_from_disk()

    def _hash_text(self, text: str) -> str:
        """Generate a hash key from text."""
        # Normalize text
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text if cached.

        Args:
            text: The text to look up

        Returns:
            Cached embedding or None
        """
        key = self._hash_text(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding for text.

        Args:
            text: The source text
            embedding: The embedding vector
        """
        key = self._hash_text(text)
        self._cache.set(key, embedding)

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], List[float]]
    ) -> List[float]:
        """
        Get cached embedding or compute and cache it.

        Args:
            text: The text to embed
            compute_fn: Function to compute embedding if not cached

        Returns:
            The embedding vector
        """
        cached = self.get(text)
        if cached is not None:
            return cached

        embedding = compute_fn(text)
        self.set(text, embedding)
        return embedding

    def save_to_disk(self) -> None:
        """Persist cache to disk using JSON (safer than pickle)."""
        if not self._persist_path:
            return

        try:
            # Use .json extension for safety
            path = Path(self._persist_path).with_suffix('.json')
            path.parent.mkdir(parents=True, exist_ok=True)

            with self._cache._lock:
                data = {
                    key: {"value": entry.value, "created_at": entry.created_at}
                    for key, entry in self._cache._cache.items()
                }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            logger.info(f"Saved {len(data)} embeddings to {path}")

        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk using JSON."""
        if not self._persist_path:
            return

        # Try JSON first (new format), then fallback to legacy pickle
        json_path = Path(self._persist_path).with_suffix('.json')
        pickle_path = Path(self._persist_path)

        data = None

        # Try JSON format first (secure)
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded cache from JSON: {json_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load JSON cache: {e}")

        # Migrate from legacy pickle if JSON doesn't exist (read-only for migration)
        elif pickle_path.exists() and pickle_path.suffix == '.pkl':
            logger.warning(
                f"Legacy pickle cache found at {pickle_path}. "
                "Please manually review and delete after migration."
            )
            # Skip loading pickle for security - require manual migration
            return

        if data is None:
            return

        loaded = 0
        for key, entry_data in data.items():
            # Validate entry structure
            if not isinstance(entry_data, dict) or "value" not in entry_data:
                continue

            entry = CacheEntry(
                value=entry_data["value"],
                created_at=entry_data.get("created_at", time.time())
            )
            if not entry.is_expired(self._cache._ttl_seconds):
                self._cache._cache[key] = entry
                loaded += 1

        logger.info(f"Loaded {loaded} embeddings from cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class ResultCache:
    """
    Cache for API query results.

    Features:
    - Key based on query parameters
    - Result deduplication
    - Parameter-aware caching
    """

    def __init__(
        self,
        max_size: int = RESULT_CACHE_MAX_SIZE,
        ttl_seconds: float = 3600
    ) -> None:
        """
        Initialize the result cache.

        Args:
            max_size: Maximum number of results to cache
            ttl_seconds: TTL for cache entries
        """
        self._cache = LRUCache[Any](
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            name="results"
        )

    def _make_key(self, tool: str, params: Dict[str, Any]) -> str:
        """Generate a cache key from tool and parameters."""
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{tool}:{params_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, tool: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached result for a query.

        Args:
            tool: The API tool name
            params: Query parameters

        Returns:
            Cached result or None
        """
        key = self._make_key(tool, params)
        return self._cache.get(key)

    def set(self, tool: str, params: Dict[str, Any], result: Any) -> None:
        """
        Cache a query result.

        Args:
            tool: The API tool name
            params: Query parameters
            result: The query result
        """
        key = self._make_key(tool, params)
        self._cache.set(key, result)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class CacheManager:
    """
    Centralized cache manager for the application.

    Provides access to all cache instances and global operations.
    """

    _instance: Optional['CacheManager'] = None

    def __new__(cls) -> 'CacheManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the cache manager."""
        if self._initialized:
            return

        self.embeddings = EmbeddingCache(
            persist_path=".cache/embeddings.pkl"
        )
        self.results = ResultCache()
        self._initialized = True

        logger.info("Cache manager initialized")

    def clear_all(self) -> None:
        """Clear all caches."""
        self.embeddings._cache.clear()
        self.results._cache.clear()
        logger.info("All caches cleared")

    def cleanup_expired(self) -> Dict[str, int]:
        """
        Clean up expired entries from all caches.

        Returns:
            Dict with cleanup counts per cache
        """
        return {
            "embeddings": self.embeddings._cache.cleanup_expired(),
            "results": self.results._cache.cleanup_expired(),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all caches."""
        return {
            "embeddings": self.embeddings.get_stats(),
            "results": self.results.get_stats(),
        }

    def save_all(self) -> None:
        """Persist all caches that support it."""
        self.embeddings.save_to_disk()
        logger.info("All caches saved")


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return CacheManager()


# Decorator for caching function results
def cached(
    cache_name: str = "default",
    ttl_seconds: float = 3600,
    key_fn: Optional[Callable[..., str]] = None
) -> Callable:
    """
    Decorator to cache function results.

    Args:
        cache_name: Name of the cache to use
        ttl_seconds: TTL for cached results
        key_fn: Optional function to generate cache key from args

    Returns:
        Decorated function
    """
    _cache = LRUCache[Any](max_size=1000, ttl_seconds=ttl_seconds, name=cache_name)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"
                key = hashlib.sha256(key.encode()).hexdigest()

            # Check cache
            result = _cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            _cache.set(key, result)
            return result

        return wrapper

    return decorator
