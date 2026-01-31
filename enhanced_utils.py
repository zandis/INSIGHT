"""
Enhanced Utilities for INSIGHT.

This module provides 20 new advanced functions for improved
performance, reliability, and functionality:

1. async_api_call - Async API wrapper for parallel execution
2. deduplicate_results - Semantic deduplication engine
3. batch_queries - Smart query batching
4. optimize_query - Query optimization engine
5. hierarchical_summarize - Multi-level summarization
6. suggest_cheaper_model - Cost optimization
7. build_task_dependency_graph - Task dependency analysis
8. validate_response_schema - Response validation
9. DataLineageTracker - Data provenance tracking
10. execute_tasks_concurrent - Parallel task execution
11. AdaptiveRateLimiter - Smart rate limiting
12. find_similar_queries - Query fuzzy matching
13. optimize_prompt_tokens - Prompt optimization
14. score_result_quality - Result quality scoring
15. SessionCheckpoint - Checkpointing system
16. update_index_incrementally - Incremental indexing
17. suggest_error_recovery - Error recovery suggestions
18. InsightProfiler - Performance profiling
19. handle_streaming_response - Streaming handler
20. parse_structured_output - Structured output parser
"""

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import (
    Any, AsyncIterator, Callable, Dict, Iterator, List,
    Optional, Set, Tuple, TypeVar, Union
)

import numpy as np

try:
    from logging_config import get_logger
    logger = get_logger("enhanced_utils")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# =============================================================================
# 1. Async API Wrapper
# =============================================================================

T = TypeVar('T')


async def async_api_call(
    func: Callable[..., T],
    *args,
    timeout: float = 30.0,
    **kwargs
) -> T:
    """
    Execute a blocking API call asynchronously.

    Args:
        func: The blocking function to call
        *args: Positional arguments for the function
        timeout: Maximum time to wait (seconds)
        **kwargs: Keyword arguments for the function

    Returns:
        The function result

    Raises:
        asyncio.TimeoutError: If call exceeds timeout
    """
    loop = asyncio.get_event_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: func(*args, **kwargs)),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"API call to {func.__name__} timed out after {timeout}s")
        raise


async def parallel_api_calls(
    calls: List[Tuple[Callable, tuple, dict]],
    max_concurrent: int = 5
) -> List[Any]:
    """
    Execute multiple API calls in parallel with concurrency limit.

    Args:
        calls: List of (function, args, kwargs) tuples
        max_concurrent: Maximum concurrent calls

    Returns:
        List of results in same order as calls
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_call(func, args, kwargs, idx):
        async with semaphore:
            try:
                result = await async_api_call(func, *args, **kwargs)
                return (idx, result, None)
            except Exception as e:
                return (idx, None, e)

    tasks = [
        limited_call(func, args, kwargs, i)
        for i, (func, args, kwargs) in enumerate(calls)
    ]

    results = await asyncio.gather(*tasks)

    # Sort by original index and extract results
    sorted_results = sorted(results, key=lambda x: x[0])
    return [r[1] if r[2] is None else r[2] for r in sorted_results]


# =============================================================================
# 2. Result Deduplication Engine
# =============================================================================

def deduplicate_results(
    results: List[Dict[str, Any]],
    threshold: float = 0.85,
    key_field: str = "content"
) -> List[Dict[str, Any]]:
    """
    Remove semantically similar results using MinHash.

    Args:
        results: List of result dictionaries
        threshold: Similarity threshold (0-1)
        key_field: Field to use for comparison

    Returns:
        Deduplicated list of results
    """
    if not results:
        return results

    def get_shingles(text: str, k: int = 3) -> Set[str]:
        """Create k-shingles from text."""
        text = text.lower()
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def minhash_signature(shingles: Set[str], num_hashes: int = 100) -> List[int]:
        """Generate MinHash signature."""
        signature = []
        for i in range(num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                h = hash((shingle, i)) % (2**32)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from signatures."""
        return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

    # Compute signatures
    signatures = []
    for result in results:
        text = str(result.get(key_field, ""))
        shingles = get_shingles(text)
        sig = minhash_signature(shingles) if shingles else []
        signatures.append(sig)

    # Find duplicates
    duplicates = set()
    for i in range(len(results)):
        if i in duplicates:
            continue
        for j in range(i + 1, len(results)):
            if j in duplicates:
                continue
            if signatures[i] and signatures[j]:
                sim = jaccard_similarity(signatures[i], signatures[j])
                if sim >= threshold:
                    duplicates.add(j)
                    logger.debug(f"Marked result {j} as duplicate of {i} (sim={sim:.2f})")

    # Return non-duplicates
    deduplicated = [r for i, r in enumerate(results) if i not in duplicates]
    logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)}")

    return deduplicated


# =============================================================================
# 3. Smart Query Batching
# =============================================================================

def batch_queries(
    queries: List[str],
    api_name: str,
    max_batch_size: int = 10,
    similarity_threshold: float = 0.7
) -> List[List[str]]:
    """
    Group similar queries for batch processing.

    Args:
        queries: List of query strings
        api_name: Target API name
        max_batch_size: Maximum queries per batch
        similarity_threshold: Threshold for grouping

    Returns:
        List of query batches
    """
    if not queries:
        return []

    # Simple word overlap similarity
    def word_overlap(q1: str, q2: str) -> float:
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    # Cluster queries
    batches = []
    remaining = list(range(len(queries)))

    while remaining:
        # Start new batch with first remaining query
        batch_indices = [remaining.pop(0)]
        batch_center = queries[batch_indices[0]]

        # Add similar queries to batch
        i = 0
        while i < len(remaining) and len(batch_indices) < max_batch_size:
            idx = remaining[i]
            if word_overlap(batch_center, queries[idx]) >= similarity_threshold:
                batch_indices.append(idx)
                remaining.pop(i)
            else:
                i += 1

        batches.append([queries[idx] for idx in batch_indices])

    logger.info(f"Batched {len(queries)} queries into {len(batches)} batches")
    return batches


# =============================================================================
# 4. Query Optimization Engine
# =============================================================================

def optimize_query(
    query: str,
    objective: str,
    context: str = "",
    api_name: str = "pubmed"
) -> str:
    """
    Optimize a query for better API results.

    Args:
        query: Original query string
        objective: Research objective
        context: Additional context
        api_name: Target API

    Returns:
        Optimized query string
    """
    # API-specific optimizations
    optimizations = {
        "pubmed": _optimize_pubmed_query,
        "mygene": _optimize_mygene_query,
        "myvariant": _optimize_myvariant_query
    }

    optimizer = optimizations.get(api_name.lower(), lambda q, o, c: q)
    optimized = optimizer(query, objective, context)

    if optimized != query:
        logger.debug(f"Optimized query: '{query[:50]}...' -> '{optimized[:50]}...'")

    return optimized


def _optimize_pubmed_query(query: str, objective: str, context: str) -> str:
    """Optimize query for PubMed API."""
    # Add relevant MeSH terms if not present
    mesh_additions = []

    # Check for common topics and add MeSH terms
    topic_mesh = {
        "cancer": "[MeSH Terms]",
        "diabetes": "[MeSH Terms]",
        "heart": "cardiovascular diseases[MeSH Terms]",
        "brain": "nervous system diseases[MeSH Terms]",
    }

    query_lower = query.lower()
    for topic, mesh in topic_mesh.items():
        if topic in query_lower and mesh not in query_lower:
            # Don't add if already specific enough
            pass

    # Add date filter for recent research
    if "recent" in query_lower or "latest" in query_lower:
        query = re.sub(r'\b(recent|latest)\b', '', query, flags=re.IGNORECASE)
        query = f"({query.strip()}) AND (\"last 5 years\"[PDat])"

    # Remove overly common words
    stop_words = {"the", "a", "an", "in", "on", "of", "for", "to"}
    words = query.split()
    filtered_words = [w for w in words if w.lower() not in stop_words or "[" in w]

    return " ".join(filtered_words)


def _optimize_mygene_query(query: str, objective: str, context: str) -> str:
    """Optimize query for MyGene API."""
    # Extract potential gene symbols (uppercase words 2-10 chars)
    gene_pattern = r'\b([A-Z][A-Z0-9]{1,9})\b'
    genes = re.findall(gene_pattern, query)

    if genes:
        # Focus on gene symbols
        return " OR ".join(genes)

    return query


def _optimize_myvariant_query(query: str, objective: str, context: str) -> str:
    """Optimize query for MyVariant API."""
    # Standardize variant formats
    # rs ID format
    rs_match = re.search(r'rs\d+', query, re.IGNORECASE)
    if rs_match:
        return rs_match.group().lower()

    # HGVS format
    hgvs_match = re.search(r'chr\d+:g\.\d+[ACGT]>[ACGT]', query, re.IGNORECASE)
    if hgvs_match:
        return hgvs_match.group()

    return query


# =============================================================================
# 5. Hierarchical Summarization
# =============================================================================

def hierarchical_summarize(
    texts: List[str],
    levels: int = 3,
    max_tokens_per_level: int = 500
) -> Dict[int, str]:
    """
    Create multi-level summaries from texts.

    Args:
        texts: List of text strings to summarize
        levels: Number of summary levels (1=most detailed)
        max_tokens_per_level: Approximate max tokens per level

    Returns:
        Dictionary mapping level -> summary
    """
    if not texts:
        return {}

    # Combine texts
    combined = "\n\n".join(texts)

    # Estimate token count (rough approximation)
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    summaries = {}

    # Level 1: Detailed summary (all key points)
    level1_points = []
    for text in texts[:20]:  # Limit to prevent explosion
        # Extract key sentences
        sentences = re.split(r'[.!?]+', text)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        level1_points.extend(key_sentences)

    summaries[1] = "\n".join(f"- {p}" for p in level1_points[:30])

    # Level 2: Condensed summary
    if levels >= 2:
        # Group similar points
        summaries[2] = "\n".join(f"- {p}" for p in level1_points[:15])

    # Level 3: Executive summary
    if levels >= 3:
        summaries[3] = "\n".join(f"- {p}" for p in level1_points[:5])

    logger.info(f"Created {len(summaries)} summary levels")
    return summaries


# =============================================================================
# 6. Cost Optimizer
# =============================================================================

# Model pricing (per 1K tokens, approximate)
MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.001, "output": 0.002},
    "text-davinci-003": {"input": 0.02, "output": 0.02},  # Deprecated
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
}


def suggest_cheaper_model(
    current_model: str,
    task_type: str,
    required_quality: str = "medium"
) -> Tuple[str, float]:
    """
    Suggest a cheaper model that can handle the task.

    Args:
        current_model: Currently used model
        task_type: Type of task (summarization, extraction, generation)
        required_quality: Required quality level (low, medium, high)

    Returns:
        Tuple of (suggested_model, estimated_savings_percent)
    """
    task_model_map = {
        "summarization": {
            "low": "gpt-4o-mini",
            "medium": "gpt-3.5-turbo",
            "high": "gpt-4o"
        },
        "extraction": {
            "low": "gpt-4o-mini",
            "medium": "gpt-4o-mini",
            "high": "gpt-3.5-turbo"
        },
        "generation": {
            "low": "gpt-3.5-turbo",
            "medium": "gpt-4o",
            "high": "gpt-4-turbo"
        },
        "analysis": {
            "low": "gpt-3.5-turbo",
            "medium": "gpt-4o",
            "high": "gpt-4-turbo"
        }
    }

    suggested = task_model_map.get(task_type, {}).get(required_quality, current_model)

    # Calculate savings
    current_cost = MODEL_PRICING.get(current_model, {}).get("input", 0.01)
    suggested_cost = MODEL_PRICING.get(suggested, {}).get("input", 0.01)

    savings = ((current_cost - suggested_cost) / current_cost * 100) if current_cost > 0 else 0

    logger.info(f"Model suggestion: {current_model} -> {suggested} ({savings:.1f}% savings)")
    return suggested, max(0, savings)


# =============================================================================
# 7. Task Dependency Graph
# =============================================================================

@dataclass
class TaskNode:
    """Node in the task dependency graph."""
    task_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0
    estimated_time: float = 1.0


def build_task_dependency_graph(
    tasks: List[str],
    objective: str
) -> Dict[str, TaskNode]:
    """
    Analyze tasks and build a dependency graph.

    Args:
        tasks: List of task descriptions
        objective: Research objective

    Returns:
        Dictionary mapping task_id to TaskNode
    """
    graph = {}

    # Identify keywords that indicate dependencies
    dependency_keywords = {
        "based on": -1,  # Depends on previous
        "after": -1,
        "using results from": -1,
        "following": -1,
        "then": -1,
        "first": None,  # Should be first
        "finally": 999,  # Should be last
    }

    for i, task in enumerate(tasks):
        task_id = f"task_{i}"
        task_lower = task.lower()

        # Detect dependencies
        dependencies = []
        priority = i  # Default priority is order

        for keyword, dep_offset in dependency_keywords.items():
            if keyword in task_lower:
                if dep_offset == -1 and i > 0:
                    dependencies.append(f"task_{i-1}")
                elif dep_offset is None:
                    priority = -100
                elif dep_offset == 999:
                    priority = 100 + i

        # Detect tool usage
        tool_keywords = ["PUBMED", "MYGENE", "MYVARIANT"]
        uses_tool = any(tool in task.upper() for tool in tool_keywords)

        graph[task_id] = TaskNode(
            task_id=task_id,
            description=task,
            dependencies=dependencies,
            priority=priority,
            estimated_time=2.0 if uses_tool else 1.0
        )

    logger.info(f"Built dependency graph with {len(graph)} nodes")
    return graph


def get_execution_order(graph: Dict[str, TaskNode]) -> List[str]:
    """
    Get optimal task execution order respecting dependencies.

    Args:
        graph: Task dependency graph

    Returns:
        Ordered list of task IDs
    """
    # Topological sort with priority
    in_degree = {tid: 0 for tid in graph}
    for node in graph.values():
        for dep in node.dependencies:
            if dep in in_degree:
                in_degree[node.task_id] += 1

    # Start with tasks that have no dependencies
    available = [(graph[tid].priority, tid) for tid, degree in in_degree.items() if degree == 0]
    import heapq
    heapq.heapify(available)

    order = []
    while available:
        _, current = heapq.heappop(available)
        order.append(current)

        # Reduce in-degree for dependent tasks
        for tid, node in graph.items():
            if current in node.dependencies:
                in_degree[tid] -= 1
                if in_degree[tid] == 0:
                    heapq.heappush(available, (node.priority, tid))

    return order


# =============================================================================
# 8. Response Schema Validator
# =============================================================================

def validate_response_schema(
    response: Any,
    api_name: str,
    schema: Optional[Dict] = None
) -> Tuple[bool, List[str]]:
    """
    Validate API response against expected schema.

    Args:
        response: API response to validate
        api_name: Name of the API
        schema: Optional custom schema

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Default schemas for known APIs
    default_schemas = {
        "pubmed": {
            "type": "dict",
            "required_keys": ["IdList"],
            "optional_keys": ["Count", "RetMax", "RetStart"]
        },
        "mygene": {
            "type": "dict",
            "required_keys": ["hits"],
            "item_schema": {"required_keys": ["_id"]}
        },
        "myvariant": {
            "type": "dict",
            "required_keys": ["_id"]
        }
    }

    schema = schema or default_schemas.get(api_name.lower(), {})

    if not schema:
        return True, []

    # Type check
    expected_type = schema.get("type", "any")
    if expected_type == "dict" and not isinstance(response, dict):
        errors.append(f"Expected dict, got {type(response).__name__}")
        return False, errors
    elif expected_type == "list" and not isinstance(response, list):
        errors.append(f"Expected list, got {type(response).__name__}")
        return False, errors

    # Required keys check
    if isinstance(response, dict):
        for key in schema.get("required_keys", []):
            if key not in response:
                errors.append(f"Missing required key: {key}")

    # Item schema check for lists
    if isinstance(response, dict) and "hits" in response and "item_schema" in schema:
        item_schema = schema["item_schema"]
        for i, item in enumerate(response.get("hits", [])[:5]):
            for key in item_schema.get("required_keys", []):
                if key not in item:
                    errors.append(f"Item {i} missing required key: {key}")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning(f"Schema validation failed for {api_name}: {errors}")

    return is_valid, errors


# =============================================================================
# 9. Data Lineage Tracker
# =============================================================================

@dataclass
class LineageRecord:
    """Record of data lineage."""
    result_id: str
    source_api: str
    query: str
    timestamp: datetime
    parent_ids: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)


class DataLineageTracker:
    """
    Track the provenance and lineage of research data.
    """

    def __init__(self):
        """Initialize the tracker."""
        self._records: Dict[str, LineageRecord] = {}

    def track_source(
        self,
        result_id: str,
        source_api: str,
        query: str,
        parent_ids: Optional[List[str]] = None
    ) -> LineageRecord:
        """
        Record a new data source.

        Args:
            result_id: Unique ID for the result
            source_api: Name of the source API
            query: Query that produced this result
            parent_ids: IDs of parent results (if derived)

        Returns:
            Created LineageRecord
        """
        record = LineageRecord(
            result_id=result_id,
            source_api=source_api,
            query=query,
            timestamp=datetime.utcnow(),
            parent_ids=parent_ids or []
        )
        self._records[result_id] = record
        logger.debug(f"Tracked lineage for {result_id} from {source_api}")
        return record

    def track_transformation(
        self,
        result_id: str,
        transformation: str
    ) -> None:
        """Record a transformation applied to a result."""
        if result_id in self._records:
            self._records[result_id].transformations.append(transformation)

    def get_lineage(self, result_id: str) -> Optional[LineageRecord]:
        """Get lineage record for a result."""
        return self._records.get(result_id)

    def get_full_provenance(self, result_id: str) -> List[LineageRecord]:
        """Get complete provenance chain for a result."""
        provenance = []
        visited = set()

        def traverse(rid: str):
            if rid in visited or rid not in self._records:
                return
            visited.add(rid)
            record = self._records[rid]
            provenance.append(record)
            for parent_id in record.parent_ids:
                traverse(parent_id)

        traverse(result_id)
        return provenance

    def export_lineage(self) -> Dict[str, Any]:
        """Export all lineage data."""
        return {
            rid: {
                "source_api": r.source_api,
                "query": r.query,
                "timestamp": r.timestamp.isoformat(),
                "parent_ids": r.parent_ids,
                "transformations": r.transformations
            }
            for rid, r in self._records.items()
        }


# =============================================================================
# 10. Concurrent Task Executor
# =============================================================================

async def execute_tasks_concurrent(
    tasks: List[Tuple[str, Callable, Dict]],
    max_concurrent: int = 3,
    fail_fast: bool = False
) -> Dict[str, Any]:
    """
    Execute multiple tasks concurrently with dependency awareness.

    Args:
        tasks: List of (task_id, callable, kwargs) tuples
        max_concurrent: Maximum concurrent tasks
        fail_fast: Stop all on first failure

    Returns:
        Dictionary mapping task_id to result or error
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    errors = {}
    stop_event = asyncio.Event()

    async def run_task(task_id: str, func: Callable, kwargs: Dict):
        if stop_event.is_set():
            return

        async with semaphore:
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: func(**kwargs))
                results[task_id] = result
                logger.debug(f"Task {task_id} completed successfully")
            except Exception as e:
                errors[task_id] = e
                logger.error(f"Task {task_id} failed: {e}")
                if fail_fast:
                    stop_event.set()

    await asyncio.gather(
        *[run_task(tid, func, kwargs) for tid, func, kwargs in tasks],
        return_exceptions=True
    )

    return {"results": results, "errors": errors}


# =============================================================================
# 11. Adaptive Rate Limiter
# =============================================================================

class AdaptiveRateLimiter:
    """
    Rate limiter that adapts based on API response patterns.
    """

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 100.0
    ):
        """
        Initialize the adaptive rate limiter.

        Args:
            initial_rate: Starting requests per second
            min_rate: Minimum rate
            max_rate: Maximum rate
        """
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self._tokens = initial_rate
        self._last_update = time.time()
        self._success_streak = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.

        Returns:
            True if acquired, False if rate limited
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(
                self.current_rate,
                self._tokens + elapsed * self.current_rate
            )
            self._last_update = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    async def record_response(self, status_code: int) -> None:
        """
        Adjust rate based on response.

        Args:
            status_code: HTTP status code of response
        """
        async with self._lock:
            if status_code == 429:  # Rate limited
                self.current_rate = max(self.min_rate, self.current_rate * 0.5)
                self._success_streak = 0
                logger.warning(f"Rate limited, reducing rate to {self.current_rate:.1f}/s")

            elif status_code >= 500:  # Server error
                self.current_rate = max(self.min_rate, self.current_rate * 0.75)
                self._success_streak = 0

            elif 200 <= status_code < 300:  # Success
                self._success_streak += 1
                if self._success_streak >= 10:
                    self.current_rate = min(self.max_rate, self.current_rate * 1.1)
                    self._success_streak = 0
                    logger.debug(f"Increasing rate to {self.current_rate:.1f}/s")

    def get_current_rate(self) -> float:
        """Get current rate limit."""
        return self.current_rate


# =============================================================================
# 12. Query Fuzzy Matcher
# =============================================================================

def find_similar_queries(
    new_query: str,
    query_history: List[str],
    threshold: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Find similar queries from history.

    Args:
        new_query: New query to match
        query_history: List of previous queries
        threshold: Minimum similarity threshold

    Returns:
        List of (similar_query, similarity_score) tuples
    """
    def normalized_levenshtein(s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein similarity."""
        if not s1 or not s2:
            return 0.0

        len1, len2 = len(s1), len(s2)
        if len1 < len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        # Initialize matrix
        current_row = list(range(len2 + 1))

        for i in range(1, len1 + 1):
            previous_row = current_row
            current_row = [i] + [0] * len2

            for j in range(1, len2 + 1):
                add = previous_row[j] + 1
                delete = current_row[j - 1] + 1
                change = previous_row[j - 1] + (s1[i-1] != s2[j-1])
                current_row[j] = min(add, delete, change)

        distance = current_row[len2]
        return 1 - (distance / max(len1, len2))

    def word_overlap_similarity(q1: str, q2: str) -> float:
        """Calculate word overlap similarity."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    similar = []
    new_query_lower = new_query.lower()

    for hist_query in query_history:
        # Combine multiple similarity metrics
        lev_sim = normalized_levenshtein(new_query_lower, hist_query.lower())
        word_sim = word_overlap_similarity(new_query, hist_query)
        combined_sim = (lev_sim + word_sim) / 2

        if combined_sim >= threshold:
            similar.append((hist_query, combined_sim))

    # Sort by similarity descending
    similar.sort(key=lambda x: x[1], reverse=True)

    if similar:
        logger.debug(f"Found {len(similar)} similar queries for '{new_query[:30]}...'")

    return similar


# =============================================================================
# 13. Prompt Token Optimizer
# =============================================================================

def optimize_prompt_tokens(
    prompt: str,
    max_tokens: int,
    preserve_structure: bool = True
) -> str:
    """
    Reduce prompt token count while preserving meaning.

    Args:
        prompt: Original prompt
        max_tokens: Target maximum tokens
        preserve_structure: Keep formatting structure

    Returns:
        Optimized prompt
    """
    # Rough token estimation (1 token ≈ 4 chars)
    estimated_tokens = len(prompt) // 4

    if estimated_tokens <= max_tokens:
        return prompt

    optimized = prompt

    # Step 1: Remove excessive whitespace
    optimized = re.sub(r'\n\s*\n', '\n\n', optimized)
    optimized = re.sub(r'[ \t]+', ' ', optimized)

    # Step 2: Remove filler phrases
    filler_phrases = [
        "please note that",
        "it is important to",
        "as you can see",
        "in order to",
        "make sure to",
        "keep in mind that",
        "it should be noted that",
    ]
    for phrase in filler_phrases:
        optimized = re.sub(phrase, '', optimized, flags=re.IGNORECASE)

    # Step 3: Abbreviate common patterns
    abbreviations = [
        (r'\bfor example\b', 'e.g.'),
        (r'\bthat is\b', 'i.e.'),
        (r'\band so on\b', 'etc.'),
        (r'\bin other words\b', 'i.e.'),
    ]
    for pattern, replacement in abbreviations:
        optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)

    # Step 4: Truncate if still too long
    estimated_tokens = len(optimized) // 4
    if estimated_tokens > max_tokens:
        target_chars = max_tokens * 4
        if preserve_structure:
            # Try to truncate at sentence boundary
            sentences = re.split(r'(?<=[.!?])\s+', optimized)
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) < target_chars:
                    truncated += sentence + " "
                else:
                    break
            optimized = truncated.strip() + "..."
        else:
            optimized = optimized[:target_chars] + "..."

    logger.debug(f"Optimized prompt from ~{len(prompt)//4} to ~{len(optimized)//4} tokens")
    return optimized


# =============================================================================
# 14. Result Quality Scorer
# =============================================================================

def score_result_quality(
    result: Dict[str, Any],
    criteria: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Score the quality of a research result.

    Args:
        result: Result dictionary to score
        criteria: Custom scoring criteria weights

    Returns:
        Tuple of (overall_score, individual_scores)
    """
    default_criteria = {
        "completeness": 0.25,
        "relevance": 0.30,
        "recency": 0.20,
        "credibility": 0.25
    }
    criteria = criteria or default_criteria

    scores = {}

    # Completeness: Check for required fields
    required_fields = ["content", "source", "title"]
    present_fields = sum(1 for f in required_fields if result.get(f))
    scores["completeness"] = present_fields / len(required_fields)

    # Relevance: Length and keyword density
    content = str(result.get("content", ""))
    if content:
        # Longer, more detailed content scores higher (up to a point)
        length_score = min(len(content) / 1000, 1.0)
        scores["relevance"] = length_score
    else:
        scores["relevance"] = 0.0

    # Recency: Check for date information
    date_str = result.get("date", result.get("publication_date", ""))
    if date_str:
        try:
            # Try to parse date
            if isinstance(date_str, str):
                year = int(re.search(r'20\d{2}', date_str).group())
                current_year = datetime.now().year
                years_old = current_year - year
                scores["recency"] = max(0, 1 - (years_old / 10))
            else:
                scores["recency"] = 0.5
        except (ValueError, AttributeError):
            scores["recency"] = 0.5
    else:
        scores["recency"] = 0.5

    # Credibility: Check source information
    source = result.get("source", "")
    credible_sources = ["pubmed", "ncbi", "nih", "nature", "science", "cell"]
    if any(cs in source.lower() for cs in credible_sources):
        scores["credibility"] = 1.0
    elif source:
        scores["credibility"] = 0.6
    else:
        scores["credibility"] = 0.3

    # Calculate weighted overall score
    overall = sum(scores.get(k, 0) * v for k, v in criteria.items())

    return overall, scores


# =============================================================================
# 15. Session Checkpointing
# =============================================================================

@dataclass
class Checkpoint:
    """Session checkpoint data."""
    checkpoint_id: str
    label: str
    timestamp: datetime
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionCheckpoint:
    """
    Manage session checkpoints for recovery.
    """

    def __init__(self, storage_path: str = ".checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            storage_path: Directory for storing checkpoints
        """
        self.storage_path = storage_path
        self._checkpoints: Dict[str, Checkpoint] = {}

        import os
        os.makedirs(storage_path, exist_ok=True)

    def create_checkpoint(
        self,
        session_state: Dict[str, Any],
        label: str = "auto"
    ) -> str:
        """
        Create a new checkpoint.

        Args:
            session_state: Current session state to save
            label: Label for the checkpoint

        Returns:
            Checkpoint ID
        """
        checkpoint_id = hashlib.md5(
            f"{datetime.utcnow().isoformat()}{label}".encode()
        ).hexdigest()[:12]

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            label=label,
            timestamp=datetime.utcnow(),
            state=session_state,
            metadata={"created_by": "auto"}
        )

        self._checkpoints[checkpoint_id] = checkpoint

        # Save to disk
        self._save_checkpoint(checkpoint)

        logger.info(f"Created checkpoint {checkpoint_id} ({label})")
        return checkpoint_id

    def restore_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Restore session from checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            Restored session state or None
        """
        checkpoint = self._checkpoints.get(checkpoint_id)

        if not checkpoint:
            checkpoint = self._load_checkpoint(checkpoint_id)

        if checkpoint:
            logger.info(f"Restored from checkpoint {checkpoint_id}")
            return checkpoint.state

        logger.warning(f"Checkpoint {checkpoint_id} not found")
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        for cp in self._checkpoints.values():
            checkpoints.append({
                "id": cp.checkpoint_id,
                "label": cp.label,
                "timestamp": cp.timestamp.isoformat(),
                "metadata": cp.metadata
            })
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)

    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk."""
        import os
        filepath = os.path.join(self.storage_path, f"{checkpoint.checkpoint_id}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "label": checkpoint.label,
                    "timestamp": checkpoint.timestamp.isoformat(),
                    "state": checkpoint.state,
                    "metadata": checkpoint.metadata
                }, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from disk."""
        import os
        filepath = os.path.join(self.storage_path, f"{checkpoint_id}.json")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return Checkpoint(
                    checkpoint_id=data["checkpoint_id"],
                    label=data["label"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    state=data["state"],
                    metadata=data.get("metadata", {})
                )
        except (FileNotFoundError, json.JSONDecodeError):
            return None


# =============================================================================
# 16. Incremental Index Updater
# =============================================================================

def update_index_incrementally(
    index: Any,
    new_documents: List[Dict[str, str]],
    batch_size: int = 10
) -> int:
    """
    Add new documents to index without full rebuild.

    Args:
        index: LlamaIndex instance
        new_documents: List of {"text": str, "id": str} dicts
        batch_size: Documents to process at once

    Returns:
        Number of documents added
    """
    from llama_index.core import Document

    added = 0

    for i in range(0, len(new_documents), batch_size):
        batch = new_documents[i:i + batch_size]

        for doc_dict in batch:
            try:
                doc = Document(
                    text=doc_dict.get("text", ""),
                    doc_id=doc_dict.get("id", str(added)),
                    metadata=doc_dict.get("metadata", {})
                )
                index.insert(doc)
                added += 1
            except Exception as e:
                logger.warning(f"Failed to add document: {e}")

    logger.info(f"Added {added} documents to index incrementally")
    return added


# =============================================================================
# 17. Error Recovery Suggester
# =============================================================================

def suggest_error_recovery(
    error: Exception,
    task: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze error and suggest recovery actions.

    Args:
        error: The exception that occurred
        task: Task that failed
        context: Additional context

    Returns:
        Recovery suggestions
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()

    suggestions = {
        "error_type": error_type,
        "recoverable": True,
        "actions": [],
        "modified_task": None,
        "wait_time": 0
    }

    # Rate limiting errors
    if "rate" in error_msg or "429" in error_msg or "too many" in error_msg:
        suggestions["actions"] = [
            "Wait before retrying",
            "Reduce request frequency",
            "Use cached results if available"
        ]
        suggestions["wait_time"] = 60
        suggestions["modified_task"] = task

    # Timeout errors
    elif "timeout" in error_msg or "timed out" in error_msg:
        suggestions["actions"] = [
            "Retry with longer timeout",
            "Simplify the query",
            "Break task into smaller parts"
        ]
        suggestions["modified_task"] = _simplify_task(task)

    # Connection errors
    elif "connection" in error_msg or "network" in error_msg:
        suggestions["actions"] = [
            "Check network connectivity",
            "Retry after brief wait",
            "Try alternative API endpoint"
        ]
        suggestions["wait_time"] = 5

    # Authentication errors
    elif "auth" in error_msg or "401" in error_msg or "403" in error_msg:
        suggestions["actions"] = [
            "Verify API credentials",
            "Check API key permissions",
            "Regenerate API token"
        ]
        suggestions["recoverable"] = False

    # Invalid input
    elif "invalid" in error_msg or "bad request" in error_msg or "400" in error_msg:
        suggestions["actions"] = [
            "Validate input parameters",
            "Check query format",
            "Review API documentation"
        ]
        suggestions["modified_task"] = _sanitize_task(task)

    # Not found
    elif "not found" in error_msg or "404" in error_msg:
        suggestions["actions"] = [
            "Verify resource exists",
            "Try alternative search terms",
            "Check spelling and formatting"
        ]
        suggestions["modified_task"] = _broaden_task(task)

    # Generic error
    else:
        suggestions["actions"] = [
            "Review error details",
            "Check system logs",
            "Retry with modified parameters"
        ]

    logger.info(f"Recovery suggestions for {error_type}: {suggestions['actions']}")
    return suggestions


def _simplify_task(task: str) -> str:
    """Simplify a task description."""
    # Remove complex phrases
    simplified = re.sub(r'\b(comprehensive|detailed|extensive|thorough)\b', '', task, flags=re.IGNORECASE)
    simplified = re.sub(r'\s+', ' ', simplified).strip()
    return simplified


def _sanitize_task(task: str) -> str:
    """Sanitize task for safe API queries."""
    # Remove special characters
    sanitized = re.sub(r'[^\w\s\-:.]', '', task)
    return sanitized.strip()


def _broaden_task(task: str) -> str:
    """Broaden a task to get more results."""
    # Remove specific identifiers
    broadened = re.sub(r'\b(rs\d+|chr\d+:[gp]\.\d+\w+)\b', '', task)
    broadened = re.sub(r'\s+', ' ', broadened).strip()
    return broadened


# =============================================================================
# 18. Performance Profiler
# =============================================================================

class InsightProfiler:
    """
    Profile INSIGHT workflow performance.
    """

    def __init__(self):
        """Initialize the profiler."""
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, float] = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self._start_times[operation] = time.time()

    def stop_timer(self, operation: str) -> float:
        """
        Stop timing and record duration.

        Returns:
            Duration in seconds
        """
        if operation not in self._start_times:
            return 0.0

        duration = time.time() - self._start_times[operation]
        self._timings[operation].append(duration)
        self._counts[operation] += 1
        del self._start_times[operation]

        return duration

    def record_timing(self, operation: str, duration: float) -> None:
        """Manually record a timing."""
        self._timings[operation].append(duration)
        self._counts[operation] += 1

    def profile_workflow(self) -> Dict[str, Any]:
        """
        Get comprehensive workflow profile.

        Returns:
            Profiling statistics
        """
        profile = {}

        for operation, timings in self._timings.items():
            if timings:
                profile[operation] = {
                    "count": self._counts[operation],
                    "total_time": sum(timings),
                    "avg_time": sum(timings) / len(timings),
                    "min_time": min(timings),
                    "max_time": max(timings)
                }

        return profile

    def identify_bottlenecks(
        self,
        threshold_percent: float = 20.0
    ) -> List[Tuple[str, float, str]]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_percent: Minimum % of total time to flag

        Returns:
            List of (operation, percentage, suggestion) tuples
        """
        total_time = sum(sum(t) for t in self._timings.values())
        if total_time == 0:
            return []

        bottlenecks = []

        for operation, timings in self._timings.items():
            op_time = sum(timings)
            percentage = (op_time / total_time) * 100

            if percentage >= threshold_percent:
                suggestion = self._get_optimization_suggestion(operation, timings)
                bottlenecks.append((operation, percentage, suggestion))

        # Sort by percentage descending
        bottlenecks.sort(key=lambda x: x[1], reverse=True)

        return bottlenecks

    def _get_optimization_suggestion(
        self,
        operation: str,
        timings: List[float]
    ) -> str:
        """Get optimization suggestion for an operation."""
        avg_time = sum(timings) / len(timings)
        op_lower = operation.lower()

        if "api" in op_lower or "fetch" in op_lower:
            if avg_time > 2.0:
                return "Consider caching API responses or batching requests"
            return "API calls are reasonably fast"

        if "embed" in op_lower:
            if avg_time > 0.5:
                return "Enable embedding cache to reduce redundant computations"
            return "Embedding performance is acceptable"

        if "index" in op_lower:
            return "Use incremental index updates instead of full rebuilds"

        if "llm" in op_lower or "gpt" in op_lower:
            if avg_time > 10.0:
                return "Consider using a faster model or reducing prompt size"
            return "LLM response times are within normal range"

        return "Review operation for optimization opportunities"

    def get_summary(self) -> str:
        """Get human-readable performance summary."""
        profile = self.profile_workflow()
        total_time = sum(p["total_time"] for p in profile.values())

        lines = [
            "Performance Summary",
            "=" * 40,
            f"Total tracked time: {total_time:.2f}s",
            "",
            "Operation Breakdown:",
        ]

        for op, stats in sorted(profile.items(), key=lambda x: x[1]["total_time"], reverse=True):
            pct = (stats["total_time"] / total_time * 100) if total_time > 0 else 0
            lines.append(
                f"  {op}: {stats['total_time']:.2f}s ({pct:.1f}%) "
                f"[avg: {stats['avg_time']*1000:.0f}ms, n={stats['count']}]"
            )

        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            lines.extend(["", "Bottlenecks Identified:"])
            for op, pct, suggestion in bottlenecks:
                lines.append(f"  - {op} ({pct:.1f}%): {suggestion}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all profiling data."""
        self._timings.clear()
        self._counts.clear()
        self._start_times.clear()


def profile_function(profiler: InsightProfiler, operation_name: str):
    """
    Decorator to profile a function.

    Args:
        profiler: InsightProfiler instance
        operation_name: Name for the operation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler.start_timer(operation_name)
            try:
                return func(*args, **kwargs)
            finally:
                profiler.stop_timer(operation_name)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler.start_timer(operation_name)
            try:
                return await func(*args, **kwargs)
            finally:
                profiler.stop_timer(operation_name)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# =============================================================================
# 19. Streaming Response Handler
# =============================================================================

async def handle_streaming_response(
    stream: AsyncIterator[str],
    callback: Optional[Callable[[str], None]] = None,
    buffer_size: int = 100
) -> str:
    """
    Handle streaming API responses efficiently.

    Args:
        stream: Async iterator of response chunks
        callback: Optional callback for each chunk
        buffer_size: Chars to buffer before callback

    Returns:
        Complete response string
    """
    full_response = []
    buffer = ""

    async for chunk in stream:
        full_response.append(chunk)
        buffer += chunk

        if callback and len(buffer) >= buffer_size:
            callback(buffer)
            buffer = ""

    # Final callback for remaining buffer
    if callback and buffer:
        callback(buffer)

    return "".join(full_response)


def handle_streaming_response_sync(
    stream: Iterator[str],
    callback: Optional[Callable[[str], None]] = None,
    buffer_size: int = 100
) -> str:
    """
    Handle streaming responses synchronously.

    Args:
        stream: Iterator of response chunks
        callback: Optional callback for each chunk
        buffer_size: Chars to buffer before callback

    Returns:
        Complete response string
    """
    full_response = []
    buffer = ""

    for chunk in stream:
        full_response.append(chunk)
        buffer += chunk

        if callback and len(buffer) >= buffer_size:
            callback(buffer)
            buffer = ""

    if callback and buffer:
        callback(buffer)

    return "".join(full_response)


# =============================================================================
# 20. Structured Output Parser
# =============================================================================

def parse_structured_output(
    response: str,
    format_spec: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Parse and validate structured output from LLM responses.

    Args:
        response: Raw LLM response
        format_spec: Expected format specification

    Returns:
        Parsed and validated output
    """
    result = {}

    # Try JSON parsing first
    json_match = re.search(r'\{[^{}]*\}|\[[^\[\]]*\]', response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                result = parsed
            elif isinstance(parsed, list):
                result = {"items": parsed}
        except json.JSONDecodeError:
            pass

    # Try structured format parsing
    if not result:
        for field_name, field_spec in format_spec.items():
            field_type = field_spec.get("type", "string")
            pattern = field_spec.get("pattern")

            if pattern:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1) if match.groups() else match.group()
                    result[field_name] = _convert_type(value, field_type)
            else:
                # Try to extract by field name
                name_pattern = rf'{field_name}\s*[:=]\s*(.+?)(?:\n|$)'
                match = re.search(name_pattern, response, re.IGNORECASE)
                if match:
                    result[field_name] = _convert_type(match.group(1).strip(), field_type)

    # Apply defaults for missing fields
    for field_name, field_spec in format_spec.items():
        if field_name not in result and "default" in field_spec:
            result[field_name] = field_spec["default"]

    # Validate required fields
    errors = []
    for field_name, field_spec in format_spec.items():
        if field_spec.get("required") and field_name not in result:
            errors.append(f"Missing required field: {field_name}")

    if errors:
        logger.warning(f"Structured output parsing errors: {errors}")
        result["_parsing_errors"] = errors

    return result


def _convert_type(value: str, target_type: str) -> Any:
    """Convert string value to target type."""
    value = value.strip()

    if target_type == "int":
        try:
            return int(re.sub(r'[^\d-]', '', value))
        except ValueError:
            return 0

    elif target_type == "float":
        try:
            return float(re.sub(r'[^\d.-]', '', value))
        except ValueError:
            return 0.0

    elif target_type == "bool":
        return value.lower() in ("true", "yes", "1", "on")

    elif target_type == "list":
        # Try to parse as comma-separated
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]

    return value


# =============================================================================
# Global instances
# =============================================================================

_profiler: Optional[InsightProfiler] = None
_lineage_tracker: Optional[DataLineageTracker] = None
_checkpoint_manager: Optional[SessionCheckpoint] = None


def get_profiler() -> InsightProfiler:
    """Get global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = InsightProfiler()
    return _profiler


def get_lineage_tracker() -> DataLineageTracker:
    """Get global lineage tracker instance."""
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = DataLineageTracker()
    return _lineage_tracker


def get_checkpoint_manager(storage_path: str = ".checkpoints") -> SessionCheckpoint:
    """Get global checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = SessionCheckpoint(storage_path)
    return _checkpoint_manager
