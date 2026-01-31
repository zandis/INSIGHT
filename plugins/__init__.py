"""
INSIGHT Plugin System.

Provides extensibility through a plugin architecture that allows:
- Custom data sources
- Custom export formats
- Custom post-processing
- Custom tools for the research agents
"""

import importlib
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

try:
    from logging_config import get_logger
    logger = get_logger("plugins")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# =============================================================================
# Plugin Base Classes
# =============================================================================

class PluginBase(ABC):
    """Base class for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    def description(self) -> str:
        """Plugin description."""
        return ""

    @property
    def author(self) -> str:
        """Plugin author."""
        return ""

    def on_load(self) -> None:
        """Called when the plugin is loaded."""
        pass

    def on_unload(self) -> None:
        """Called when the plugin is unloaded."""
        pass


class DataSourcePlugin(PluginBase):
    """
    Plugin for custom data sources.

    Implement this to add new APIs or databases to query.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the data source (e.g., 'DRUGBANK')."""
        pass

    @abstractmethod
    async def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a query against the data source.

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            Query results
        """
        pass

    @abstractmethod
    def get_tool_description(self) -> str:
        """
        Get the tool description for the boss agent.

        Returns:
            Description string for task planning
        """
        pass


class ExportPlugin(PluginBase):
    """
    Plugin for custom export formats.

    Implement this to add new export formats.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of the export format (e.g., 'WORD')."""
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension (e.g., '.docx')."""
        pass

    @property
    def mime_type(self) -> str:
        """MIME type for the export."""
        return "application/octet-stream"

    @abstractmethod
    async def export(
        self,
        session_data: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> bytes:
        """
        Export research results to the custom format.

        Args:
            session_data: Research session metadata
            results: List of research results

        Returns:
            Exported content as bytes
        """
        pass


class PostProcessorPlugin(PluginBase):
    """
    Plugin for post-processing results.

    Implement this to add custom processing steps after research completes.
    """

    @property
    @abstractmethod
    def processor_name(self) -> str:
        """Name of the processor."""
        pass

    @abstractmethod
    async def process(
        self,
        session_data: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process the research results.

        Args:
            session_data: Research session metadata
            results: List of research results

        Returns:
            Processed results
        """
        pass


class HookPlugin(PluginBase):
    """
    Plugin for lifecycle hooks.

    Implement this to execute custom code at various points.
    """

    async def on_research_start(
        self,
        session_id: str,
        objective: str,
        tools: List[str]
    ) -> None:
        """Called when research starts."""
        pass

    async def on_task_start(
        self,
        session_id: str,
        task: str
    ) -> None:
        """Called when a task starts."""
        pass

    async def on_task_complete(
        self,
        session_id: str,
        task: str,
        result: Any
    ) -> None:
        """Called when a task completes."""
        pass

    async def on_research_complete(
        self,
        session_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Called when research completes."""
        pass

    async def on_research_error(
        self,
        session_id: str,
        error: Exception
    ) -> None:
        """Called when research encounters an error."""
        pass


# =============================================================================
# Plugin Manager
# =============================================================================

@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    instance: PluginBase
    enabled: bool = True
    load_error: Optional[str] = None


class PluginManager:
    """
    Manages plugin discovery, loading, and execution.
    """

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        Initialize the plugin manager.

        Args:
            plugin_dirs: Directories to search for plugins
        """
        self._plugins: Dict[str, PluginInfo] = {}
        self._data_sources: Dict[str, DataSourcePlugin] = {}
        self._export_formats: Dict[str, ExportPlugin] = {}
        self._post_processors: Dict[str, PostProcessorPlugin] = {}
        self._hooks: List[HookPlugin] = []

        # Default plugin directory
        self._plugin_dirs = plugin_dirs or [
            os.path.join(os.path.dirname(__file__), "installed")
        ]

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin directories.

        Returns:
            List of discovered plugin module names
        """
        discovered = []

        for plugin_dir in self._plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue

            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)

                # Check for Python files
                if item.endswith(".py") and not item.startswith("_"):
                    discovered.append(item[:-3])

                # Check for package directories
                elif os.path.isdir(item_path):
                    init_file = os.path.join(item_path, "__init__.py")
                    if os.path.exists(init_file):
                        discovered.append(item)

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def load_plugin(self, module_name: str) -> Optional[PluginInfo]:
        """
        Load a plugin by module name.

        Args:
            module_name: Python module name

        Returns:
            PluginInfo if loaded successfully
        """
        try:
            # Import the module
            module = importlib.import_module(f"plugins.installed.{module_name}")

            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PluginBase) and obj is not PluginBase:
                    if obj not in (DataSourcePlugin, ExportPlugin,
                                   PostProcessorPlugin, HookPlugin):
                        plugin_classes.append(obj)

            if not plugin_classes:
                logger.warning(f"No plugin classes found in {module_name}")
                return None

            # Instantiate the first plugin class found
            plugin_class = plugin_classes[0]
            instance = plugin_class()

            # Determine plugin type
            if isinstance(instance, DataSourcePlugin):
                plugin_type = "data_source"
                self._data_sources[instance.source_name] = instance
            elif isinstance(instance, ExportPlugin):
                plugin_type = "export"
                self._export_formats[instance.format_name] = instance
            elif isinstance(instance, PostProcessorPlugin):
                plugin_type = "post_processor"
                self._post_processors[instance.processor_name] = instance
            elif isinstance(instance, HookPlugin):
                plugin_type = "hook"
                self._hooks.append(instance)
            else:
                plugin_type = "generic"

            # Create plugin info
            info = PluginInfo(
                name=instance.name,
                version=instance.version,
                description=instance.description,
                author=instance.author,
                plugin_type=plugin_type,
                instance=instance
            )

            self._plugins[instance.name] = info

            # Call on_load
            instance.on_load()

            logger.info(f"Loaded plugin: {instance.name} v{instance.version}")
            return info

        except Exception as e:
            logger.error(f"Failed to load plugin {module_name}: {e}")
            return PluginInfo(
                name=module_name,
                version="unknown",
                description="",
                author="",
                plugin_type="unknown",
                instance=None,
                enabled=False,
                load_error=str(e)
            )

    def load_all_plugins(self) -> Dict[str, PluginInfo]:
        """
        Discover and load all available plugins.

        Returns:
            Dictionary of loaded plugins
        """
        discovered = self.discover_plugins()

        for module_name in discovered:
            self.load_plugin(module_name)

        return self._plugins

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if unloaded
        """
        info = self._plugins.get(plugin_name)
        if not info:
            return False

        # Call on_unload
        if info.instance:
            info.instance.on_unload()

        # Remove from registries
        if isinstance(info.instance, DataSourcePlugin):
            self._data_sources.pop(info.instance.source_name, None)
        elif isinstance(info.instance, ExportPlugin):
            self._export_formats.pop(info.instance.format_name, None)
        elif isinstance(info.instance, PostProcessorPlugin):
            self._post_processors.pop(info.instance.processor_name, None)
        elif isinstance(info.instance, HookPlugin):
            if info.instance in self._hooks:
                self._hooks.remove(info.instance)

        del self._plugins[plugin_name]
        logger.info(f"Unloaded plugin: {plugin_name}")
        return True

    def get_plugin(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return list(self._plugins.values())

    def get_data_source(self, source_name: str) -> Optional[DataSourcePlugin]:
        """Get a data source plugin by name."""
        return self._data_sources.get(source_name)

    def get_export_format(self, format_name: str) -> Optional[ExportPlugin]:
        """Get an export plugin by format name."""
        return self._export_formats.get(format_name)

    def get_available_data_sources(self) -> List[str]:
        """Get list of available data source names."""
        return list(self._data_sources.keys())

    def get_available_export_formats(self) -> List[str]:
        """Get list of available export format names."""
        return list(self._export_formats.keys())

    # Hook execution methods

    async def execute_research_start_hooks(
        self,
        session_id: str,
        objective: str,
        tools: List[str]
    ) -> None:
        """Execute all research start hooks."""
        for hook in self._hooks:
            try:
                await hook.on_research_start(session_id, objective, tools)
            except Exception as e:
                logger.error(f"Hook error in on_research_start: {e}")

    async def execute_task_start_hooks(
        self,
        session_id: str,
        task: str
    ) -> None:
        """Execute all task start hooks."""
        for hook in self._hooks:
            try:
                await hook.on_task_start(session_id, task)
            except Exception as e:
                logger.error(f"Hook error in on_task_start: {e}")

    async def execute_task_complete_hooks(
        self,
        session_id: str,
        task: str,
        result: Any
    ) -> None:
        """Execute all task complete hooks."""
        for hook in self._hooks:
            try:
                await hook.on_task_complete(session_id, task, result)
            except Exception as e:
                logger.error(f"Hook error in on_task_complete: {e}")

    async def execute_research_complete_hooks(
        self,
        session_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Execute all research complete hooks."""
        for hook in self._hooks:
            try:
                await hook.on_research_complete(session_id, results)
            except Exception as e:
                logger.error(f"Hook error in on_research_complete: {e}")

    async def execute_research_error_hooks(
        self,
        session_id: str,
        error: Exception
    ) -> None:
        """Execute all research error hooks."""
        for hook in self._hooks:
            try:
                await hook.on_research_error(session_id, error)
            except Exception as e:
                logger.error(f"Hook error in on_research_error: {e}")

    async def run_post_processors(
        self,
        session_data: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run all post processors on results.

        Args:
            session_data: Research session metadata
            results: Research results

        Returns:
            Processed results
        """
        processed = results

        for processor in self._post_processors.values():
            try:
                processed = await processor.process(session_data, processed)
            except Exception as e:
                logger.error(f"Post-processor error: {e}")

        return processed


# Singleton instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def initialize_plugins(plugin_dirs: Optional[List[str]] = None) -> PluginManager:
    """
    Initialize the plugin system.

    Args:
        plugin_dirs: Custom plugin directories

    Returns:
        Configured PluginManager
    """
    global _plugin_manager
    _plugin_manager = PluginManager(plugin_dirs)
    _plugin_manager.load_all_plugins()
    return _plugin_manager
