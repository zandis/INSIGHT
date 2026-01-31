"""
Example Plugin for INSIGHT.

This demonstrates how to create custom plugins.
"""

from typing import Any, Dict, List
from plugins import DataSourcePlugin, ExportPlugin, HookPlugin


class ExampleDataSource(DataSourcePlugin):
    """
    Example data source plugin that queries a custom API.
    """

    @property
    def name(self) -> str:
        return "Example Data Source"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Example plugin demonstrating custom data sources"

    @property
    def author(self) -> str:
        return "INSIGHT Team"

    @property
    def source_name(self) -> str:
        return "EXAMPLE"

    def get_tool_description(self) -> str:
        return """
        Query Example API. This is useful for finding example data.
        If you wish to use this tool, say 'EXAMPLE:' followed by your query.
        Example: 'EXAMPLE: find sample research data'
        """

    async def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a query against the example data source."""
        # In a real plugin, this would call an external API
        return {
            "query": query,
            "results": [
                {"id": 1, "title": "Example Result 1", "data": "Sample data"},
                {"id": 2, "title": "Example Result 2", "data": "More sample data"},
            ],
            "total": 2,
            "source": "example_api"
        }

    def on_load(self) -> None:
        """Called when the plugin is loaded."""
        print(f"Example Data Source plugin loaded (v{self.version})")


class ExampleExporter(ExportPlugin):
    """
    Example export plugin that creates a custom format.
    """

    @property
    def name(self) -> str:
        return "Example Exporter"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Example plugin demonstrating custom export formats"

    @property
    def author(self) -> str:
        return "INSIGHT Team"

    @property
    def format_name(self) -> str:
        return "EXAMPLE"

    @property
    def file_extension(self) -> str:
        return ".example"

    @property
    def mime_type(self) -> str:
        return "application/x-example"

    async def export(
        self,
        session_data: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> bytes:
        """Export to custom format."""
        output = f"""
# INSIGHT Research Export
# Format: Example Custom Format
# Session: {session_data.get('id', 'unknown')}

Objective: {session_data.get('objective', 'N/A')}
Status: {session_data.get('status', 'N/A')}

## Results ({len(results)} items)
"""
        for i, result in enumerate(results, 1):
            output += f"\n### Result {i}\n"
            output += f"Task: {result.get('task', 'N/A')}\n"
            output += f"Summary: {result.get('summary', 'N/A')}\n"

        return output.encode('utf-8')


class ExampleHooks(HookPlugin):
    """
    Example hooks plugin that logs lifecycle events.
    """

    @property
    def name(self) -> str:
        return "Example Hooks"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Example plugin demonstrating lifecycle hooks"

    @property
    def author(self) -> str:
        return "INSIGHT Team"

    async def on_research_start(
        self,
        session_id: str,
        objective: str,
        tools: List[str]
    ) -> None:
        """Called when research starts."""
        print(f"[HOOK] Research started: {session_id}")
        print(f"[HOOK] Objective: {objective[:50]}...")

    async def on_task_complete(
        self,
        session_id: str,
        task: str,
        result: Any
    ) -> None:
        """Called when a task completes."""
        print(f"[HOOK] Task completed in session {session_id}: {task[:50]}...")

    async def on_research_complete(
        self,
        session_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Called when research completes."""
        print(f"[HOOK] Research completed: {session_id}")
        print(f"[HOOK] Results count: {len(results.get('results', []))}")

    async def on_research_error(
        self,
        session_id: str,
        error: Exception
    ) -> None:
        """Called when research encounters an error."""
        print(f"[HOOK] Research error in {session_id}: {error}")
