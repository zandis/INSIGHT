"""
Workflow Enhancements for INSIGHT.

This module integrates the enhanced utilities into the main workflow,
providing improved performance, reliability, and functionality.
"""

import asyncio
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from logging_config import get_logger
    logger = get_logger("workflow")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from enhanced_utils import (
    AdaptiveRateLimiter,
    DataLineageTracker,
    InsightProfiler,
    SessionCheckpoint,
    batch_queries,
    deduplicate_results,
    execute_tasks_concurrent,
    find_similar_queries,
    get_checkpoint_manager,
    get_lineage_tracker,
    get_profiler,
    optimize_query,
    score_result_quality,
    suggest_error_recovery,
    validate_response_schema,
)


class EnhancedWorkflow:
    """
    Enhanced research workflow with integrated utilities.
    """

    def __init__(
        self,
        api_key: str,
        enable_profiling: bool = True,
        enable_checkpointing: bool = True,
        enable_lineage: bool = True
    ):
        """
        Initialize the enhanced workflow.

        Args:
            api_key: OpenAI API key
            enable_profiling: Enable performance profiling
            enable_checkpointing: Enable session checkpoints
            enable_lineage: Enable data lineage tracking
        """
        self.api_key = api_key
        self._profiler = get_profiler() if enable_profiling else None
        self._checkpointer = get_checkpoint_manager() if enable_checkpointing else None
        self._lineage = get_lineage_tracker() if enable_lineage else None

        # Rate limiters for each API
        self._rate_limiters = {
            "pubmed": AdaptiveRateLimiter(initial_rate=3.0, min_rate=0.5, max_rate=10.0),
            "mygene": AdaptiveRateLimiter(initial_rate=10.0, min_rate=1.0, max_rate=30.0),
            "myvariant": AdaptiveRateLimiter(initial_rate=10.0, min_rate=1.0, max_rate=30.0),
            "openai": AdaptiveRateLimiter(initial_rate=20.0, min_rate=1.0, max_rate=60.0),
        }

        # Query history for deduplication
        self._query_history: Dict[str, List[str]] = defaultdict(list)

        # Results cache
        self._results_cache: Dict[str, Any] = {}

    async def run_research(
        self,
        objective: str,
        tools: List[str],
        max_iterations: int = 5,
        checkpoint_interval: int = 2
    ) -> Dict[str, Any]:
        """
        Run enhanced research workflow.

        Args:
            objective: Research objective
            tools: List of tools to use
            max_iterations: Maximum iterations
            checkpoint_interval: Create checkpoint every N iterations

        Returns:
            Research results
        """
        session_state = {
            "objective": objective,
            "tools": tools,
            "iteration": 0,
            "results": [],
            "summaries": [],
            "completed_tasks": []
        }

        if self._profiler:
            self._profiler.start_timer("total_workflow")

        try:
            for iteration in range(max_iterations):
                session_state["iteration"] = iteration + 1

                if self._profiler:
                    self._profiler.start_timer(f"iteration_{iteration}")

                # Run iteration
                iteration_results = await self._run_iteration(
                    objective, tools, session_state
                )

                # Update state
                session_state["results"].extend(iteration_results.get("results", []))
                session_state["summaries"].append(iteration_results.get("summary", ""))
                session_state["completed_tasks"].extend(
                    iteration_results.get("completed_tasks", [])
                )

                if self._profiler:
                    self._profiler.stop_timer(f"iteration_{iteration}")

                # Checkpoint
                if self._checkpointer and (iteration + 1) % checkpoint_interval == 0:
                    self._checkpointer.create_checkpoint(
                        session_state,
                        label=f"iteration_{iteration + 1}"
                    )

                logger.info(f"Completed iteration {iteration + 1}/{max_iterations}")

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            # Emergency checkpoint
            if self._checkpointer:
                self._checkpointer.create_checkpoint(
                    session_state,
                    label="error_recovery"
                )
            raise

        finally:
            if self._profiler:
                self._profiler.stop_timer("total_workflow")

        # Deduplicate final results
        session_state["results"] = deduplicate_results(session_state["results"])

        # Score results
        scored_results = []
        for result in session_state["results"]:
            score, details = score_result_quality(result)
            result["quality_score"] = score
            result["quality_details"] = details
            scored_results.append(result)

        # Sort by quality
        scored_results.sort(key=lambda r: r.get("quality_score", 0), reverse=True)
        session_state["results"] = scored_results

        # Add profiling summary
        if self._profiler:
            session_state["performance"] = self._profiler.profile_workflow()
            session_state["bottlenecks"] = self._profiler.identify_bottlenecks()

        return session_state

    async def _run_iteration(
        self,
        objective: str,
        tools: List[str],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single research iteration."""
        results = []
        completed_tasks = []

        # Generate tasks (placeholder - would use boss_agent)
        tasks = await self._generate_tasks(objective, tools, state)

        # Batch and optimize queries
        for task in tasks:
            api_name = self._detect_api(task)

            if api_name:
                # Optimize query
                optimized_query = optimize_query(
                    task, objective, str(state.get("summaries", []))[-1000:], api_name
                )

                # Check for similar previous queries
                similar = find_similar_queries(
                    optimized_query,
                    self._query_history[api_name],
                    threshold=0.9
                )

                if similar:
                    logger.info(f"Found similar previous query, reusing results")
                    cached_key = f"{api_name}:{similar[0][0]}"
                    if cached_key in self._results_cache:
                        results.append(self._results_cache[cached_key])
                        continue

                # Execute query with rate limiting
                try:
                    limiter = self._rate_limiters.get(api_name)
                    if limiter:
                        while not await limiter.acquire():
                            await asyncio.sleep(0.1)

                    result = await self._execute_query(api_name, optimized_query)

                    if limiter:
                        await limiter.record_response(200)

                    # Validate response
                    is_valid, errors = validate_response_schema(result, api_name)
                    if not is_valid:
                        logger.warning(f"Invalid response from {api_name}: {errors}")

                    # Track lineage
                    if self._lineage:
                        self._lineage.track_source(
                            result_id=f"{api_name}_{len(results)}",
                            source_api=api_name,
                            query=optimized_query
                        )

                    # Cache result
                    cache_key = f"{api_name}:{optimized_query}"
                    self._results_cache[cache_key] = result
                    self._query_history[api_name].append(optimized_query)

                    results.append(result)
                    completed_tasks.append(task)

                except Exception as e:
                    # Error recovery
                    recovery = suggest_error_recovery(e, task, state)
                    logger.warning(f"Query failed, recovery: {recovery['actions']}")

                    if recovery.get("wait_time"):
                        await asyncio.sleep(recovery["wait_time"])

                    if recovery.get("modified_task"):
                        # Retry with modified task
                        pass

        return {
            "results": results,
            "completed_tasks": completed_tasks,
            "summary": self._generate_summary(results)
        }

    async def _generate_tasks(
        self,
        objective: str,
        tools: List[str],
        state: Dict[str, Any]
    ) -> List[str]:
        """Generate tasks for current iteration."""
        # Placeholder - would integrate with boss_agent
        tasks = []
        for tool in tools:
            tasks.append(f"{tool}: Search for information related to {objective}")
        return tasks

    def _detect_api(self, task: str) -> Optional[str]:
        """Detect which API a task is targeting."""
        task_upper = task.upper()
        if "PUBMED" in task_upper:
            return "pubmed"
        elif "MYGENE" in task_upper:
            return "mygene"
        elif "MYVARIANT" in task_upper:
            return "myvariant"
        return None

    async def _execute_query(self, api_name: str, query: str) -> Dict[str, Any]:
        """Execute a query against the specified API."""
        if self._profiler:
            self._profiler.start_timer(f"api_{api_name}")

        try:
            # Placeholder - would call actual API wrappers
            result = {"api": api_name, "query": query, "data": [], "content": query}
            return result
        finally:
            if self._profiler:
                self._profiler.stop_timer(f"api_{api_name}")

    def _generate_summary(self, results: List[Dict]) -> str:
        """Generate summary from results."""
        if not results:
            return "No results obtained."
        return f"Obtained {len(results)} results from research queries."

    def get_performance_report(self) -> str:
        """Get performance profiling report."""
        if self._profiler:
            return self._profiler.get_summary()
        return "Profiling not enabled"

    def get_lineage_report(self) -> Dict[str, Any]:
        """Get data lineage report."""
        if self._lineage:
            return self._lineage.export_lineage()
        return {}

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        if self._checkpointer:
            return self._checkpointer.list_checkpoints()
        return []

    async def restore_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[Dict[str, Any]]:
        """Restore workflow from checkpoint."""
        if self._checkpointer:
            return self._checkpointer.restore_from_checkpoint(checkpoint_id)
        return None


class ParallelResearchExecutor:
    """
    Execute multiple research queries in parallel.
    """

    def __init__(self, max_concurrent: int = 3):
        """
        Initialize parallel executor.

        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self._profiler = get_profiler()

    async def execute_parallel(
        self,
        queries: List[Tuple[str, str, Dict]],
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        Execute queries in parallel.

        Args:
            queries: List of (task_id, api_name, query_params) tuples
            fail_fast: Stop on first failure

        Returns:
            Results dictionary
        """
        self._profiler.start_timer("parallel_execution")

        # Convert to format expected by execute_tasks_concurrent
        tasks = [
            (
                task_id,
                self._create_query_func(api_name),
                query_params
            )
            for task_id, api_name, query_params in queries
        ]

        results = await execute_tasks_concurrent(
            tasks,
            max_concurrent=self.max_concurrent,
            fail_fast=fail_fast
        )

        self._profiler.stop_timer("parallel_execution")

        return results

    def _create_query_func(self, api_name: str) -> Callable:
        """Create query function for API."""
        async def query_func(**kwargs):
            # Placeholder - would call actual API
            await asyncio.sleep(0.1)  # Simulate API call
            return {"api": api_name, **kwargs}
        return query_func


class SmartQueryBatcher:
    """
    Intelligently batch similar queries.
    """

    def __init__(self, max_batch_size: int = 10):
        """
        Initialize query batcher.

        Args:
            max_batch_size: Maximum queries per batch
        """
        self.max_batch_size = max_batch_size

    def batch_for_api(
        self,
        queries: List[str],
        api_name: str
    ) -> List[List[str]]:
        """
        Batch queries for a specific API.

        Args:
            queries: List of query strings
            api_name: Target API name

        Returns:
            List of query batches
        """
        return batch_queries(
            queries,
            api_name,
            max_batch_size=self.max_batch_size
        )


# Factory functions for easy access
def create_enhanced_workflow(
    api_key: str,
    **options
) -> EnhancedWorkflow:
    """
    Create an enhanced workflow instance.

    Args:
        api_key: OpenAI API key
        **options: Additional options

    Returns:
        EnhancedWorkflow instance
    """
    return EnhancedWorkflow(api_key, **options)


def create_parallel_executor(max_concurrent: int = 3) -> ParallelResearchExecutor:
    """
    Create a parallel executor instance.

    Args:
        max_concurrent: Maximum concurrent tasks

    Returns:
        ParallelResearchExecutor instance
    """
    return ParallelResearchExecutor(max_concurrent)
