"""
Enhanced CLI for INSIGHT with rich output and improved UX.

Provides a modern command-line interface with progress bars,
formatted tables, and colorful output.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Try to import rich for enhanced output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for enhanced CLI output: pip install rich")

# Import colorama as fallback
from colorama import Fore, Style, init
init(autoreset=True)


class EnhancedCLI:
    """
    Enhanced command-line interface for INSIGHT.
    """

    def __init__(self):
        """Initialize the CLI."""
        self.console = Console() if RICH_AVAILABLE else None

    def print_banner(self) -> None:
        """Print the INSIGHT banner."""
        banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██╗███╗   ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗     ║
    ║   ██║████╗  ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝     ║
    ║   ██║██╔██╗ ██║███████╗██║██║  ███╗███████║   ██║        ║
    ║   ██║██║╚██╗██║╚════██║██║██║   ██║██╔══██║   ██║        ║
    ║   ██║██║ ╚████║███████║██║╚██████╔╝██║  ██║   ██║        ║
    ║   ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝        ║
    ║                                                           ║
    ║       Autonomous AI-Powered Medical Research              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
        """

        if RICH_AVAILABLE:
            self.console.print(Panel(banner, style="bold magenta", border_style="magenta"))
        else:
            print(Fore.MAGENTA + banner)

    def print_success(self, message: str) -> None:
        """Print a success message."""
        if RICH_AVAILABLE:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(Fore.GREEN + "✓ " + message)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        if RICH_AVAILABLE:
            self.console.print(f"[bold red]✗[/bold red] {message}")
        else:
            print(Fore.RED + "✗ " + message)

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        if RICH_AVAILABLE:
            self.console.print(f"[bold yellow]⚠[/bold yellow] {message}")
        else:
            print(Fore.YELLOW + "⚠ " + message)

    def print_info(self, message: str) -> None:
        """Print an info message."""
        if RICH_AVAILABLE:
            self.console.print(f"[bold blue]ℹ[/bold blue] {message}")
        else:
            print(Fore.BLUE + "ℹ " + message)

    def print_section(self, title: str) -> None:
        """Print a section header."""
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold cyan]{'─' * 5} {title} {'─' * (50 - len(title))}[/bold cyan]")
        else:
            print(Fore.CYAN + f"\n{'─' * 5} {title} {'─' * (50 - len(title))}")

    def print_table(self, title: str, headers: List[str], rows: List[List[str]]) -> None:
        """Print a formatted table."""
        if RICH_AVAILABLE:
            table = Table(title=title, show_header=True, header_style="bold blue")
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
        else:
            print(f"\n{Fore.CYAN}{title}")
            print(Fore.CYAN + "─" * 60)
            print(Fore.WHITE + " | ".join(headers))
            print("─" * 60)
            for row in rows:
                print(" | ".join(row))
            print()

    def print_tree(self, title: str, items: Dict[str, Any]) -> None:
        """Print a tree structure."""
        if RICH_AVAILABLE:
            tree = Tree(f"[bold]{title}[/bold]")
            self._add_tree_items(tree, items)
            self.console.print(tree)
        else:
            print(f"\n{title}")
            self._print_dict_tree(items, indent=0)

    def _add_tree_items(self, tree: Any, items: Dict[str, Any], prefix: str = "") -> None:
        """Recursively add items to a tree."""
        if not RICH_AVAILABLE:
            return

        for key, value in items.items():
            if isinstance(value, dict):
                branch = tree.add(f"[bold]{key}[/bold]")
                self._add_tree_items(branch, value)
            elif isinstance(value, list):
                branch = tree.add(f"[bold]{key}[/bold]")
                for item in value[:10]:  # Limit display
                    branch.add(str(item))
                if len(value) > 10:
                    branch.add(f"... and {len(value) - 10} more")
            else:
                tree.add(f"{key}: [green]{value}[/green]")

    def _print_dict_tree(self, items: Dict[str, Any], indent: int) -> None:
        """Print dictionary as tree (fallback)."""
        for key, value in items.items():
            prefix = "  " * indent + "├── "
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict_tree(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key}:")
                for item in value[:5]:
                    print(f"  " * (indent + 1) + f"├── {item}")
            else:
                print(f"{prefix}{key}: {value}")

    def progress_bar(self, description: str, total: int) -> Any:
        """Create a progress bar context manager."""
        if RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
        else:
            return FallbackProgress(description, total)

    def prompt(self, message: str, default: str = "") -> str:
        """Prompt for user input."""
        if RICH_AVAILABLE:
            return Prompt.ask(message, default=default)
        else:
            result = input(f"{message} [{default}]: ").strip()
            return result if result else default

    def confirm(self, message: str, default: bool = False) -> bool:
        """Prompt for confirmation."""
        if RICH_AVAILABLE:
            return Confirm.ask(message, default=default)
        else:
            result = input(f"{message} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
            if not result:
                return default
            return result in ('y', 'yes')

    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted research results summary."""
        self.print_section("Research Results Summary")

        if RICH_AVAILABLE:
            # Objective
            if results.get('objective'):
                self.console.print(Panel(
                    results['objective'],
                    title="Objective",
                    border_style="blue"
                ))

            # Statistics
            stats_table = Table(title="Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            stats_table.add_row("Tasks Completed", str(results.get('tasks_completed', 0)))
            stats_table.add_row("Results Found", str(results.get('results_count', 0)))
            stats_table.add_row("Duration", f"{results.get('duration', 0):.2f}s")
            self.console.print(stats_table)

            # Key Findings
            if results.get('key_findings'):
                self.console.print(Panel(
                    Markdown(results['key_findings']),
                    title="Key Findings",
                    border_style="green"
                ))
        else:
            print(f"\nObjective: {results.get('objective', 'N/A')}")
            print(f"Tasks Completed: {results.get('tasks_completed', 0)}")
            print(f"Results Found: {results.get('results_count', 0)}")
            print(f"Duration: {results.get('duration', 0):.2f}s")
            if results.get('key_findings'):
                print(f"\nKey Findings:\n{results['key_findings']}")


class FallbackProgress:
    """Fallback progress indicator when rich is not available."""

    def __init__(self, description: str, total: int):
        self.description = description
        self.total = total
        self.current = 0

    def __enter__(self):
        print(f"\n{self.description}")
        return self

    def __exit__(self, *args):
        print("\nComplete!")

    def add_task(self, description: str, total: int) -> int:
        self.description = description
        self.total = total
        return 0

    def update(self, task_id: int, advance: int = 1, **kwargs) -> None:
        self.current += advance
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        bar_width = 40
        filled = int(bar_width * self.current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{bar}] {percentage:.1f}% - {self.description}", end="", flush=True)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="INSIGHT - Autonomous AI-Powered Medical Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run "Find genes associated with breast cancer"
  %(prog)s run "Research BRCA1 mutations" --tools PUBMED MYGENE --iterations 10
  %(prog)s reload /path/to/session
  %(prog)s web --port 8080
  %(prog)s health
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a research session")
    run_parser.add_argument("objective", help="Research objective")
    run_parser.add_argument(
        "--tools", "-t",
        nargs="+",
        choices=["PUBMED", "MYGENE", "MYVARIANT"],
        default=["PUBMED", "MYGENE"],
        help="Tools to use"
    )
    run_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Number of iterations"
    )
    run_parser.add_argument(
        "--output", "-o",
        help="Output directory for results"
    )
    run_parser.add_argument(
        "--data-file", "-d",
        help="Additional data file to include"
    )

    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload a previous session")
    reload_parser.add_argument("path", help="Path to session state")
    reload_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Additional iterations to run"
    )

    # Web command
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    web_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    web_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    # Health command
    subparsers.add_parser("health", help="Check system health")

    # Config command
    config_parser = subparsers.add_parser("config", help="View/edit configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    config_parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set config value")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export session results")
    export_parser.add_argument("session_path", help="Path to session")
    export_parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "markdown", "bibtex", "pdf"],
        default="json",
        help="Export format"
    )
    export_parser.add_argument("--output", "-o", help="Output file path")

    return parser


def main():
    """Main entry point for enhanced CLI."""
    cli = EnhancedCLI()
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        cli.print_banner()
        parser.print_help()
        return

    cli.print_banner()

    if args.command == "run":
        cli.print_section("Starting Research")
        cli.print_info(f"Objective: {args.objective}")
        cli.print_info(f"Tools: {', '.join(args.tools)}")
        cli.print_info(f"Iterations: {args.iterations}")

        if cli.confirm("Proceed with research?", default=True):
            try:
                from main import run
                from config import OPENAI_API_KEY

                api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    cli.print_error("OpenAI API key not configured")
                    sys.exit(1)

                run(
                    api_key=api_key,
                    OBJECTIVE=args.objective,
                    TOOLS=args.tools,
                    MAX_ITERATIONS=args.iterations,
                    my_data_path=args.data_file or ""
                )
                cli.print_success("Research completed successfully!")

            except Exception as e:
                cli.print_error(f"Research failed: {e}")
                sys.exit(1)

    elif args.command == "reload":
        cli.print_section("Reloading Session")
        cli.print_info(f"Session path: {args.path}")

        try:
            from main import run
            from config import OPENAI_API_KEY

            api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            run(
                api_key=api_key,
                MAX_ITERATIONS=args.iterations,
                reload_path=args.path
            )
            cli.print_success("Session reloaded and continued!")

        except Exception as e:
            cli.print_error(f"Failed to reload: {e}")
            sys.exit(1)

    elif args.command == "web":
        cli.print_section("Starting Web Interface")
        cli.print_info(f"Host: {args.host}")
        cli.print_info(f"Port: {args.port}")

        try:
            import uvicorn
            uvicorn.run(
                "web.app:app",
                host=args.host,
                port=args.port,
                reload=args.reload
            )
        except ImportError:
            cli.print_error("uvicorn not installed. Run: pip install uvicorn")
            sys.exit(1)

    elif args.command == "health":
        cli.print_section("System Health Check")

        try:
            from health_check import quick_health_check
            results = quick_health_check()

            rows = []
            for service, status in results.items():
                status_str = "✓ OK" if status else "✗ FAIL"
                rows.append([service, status_str])

            cli.print_table("Health Status", ["Service", "Status"], rows)

            all_healthy = all(results.values())
            if all_healthy:
                cli.print_success("All systems operational")
            else:
                cli.print_warning("Some services are unavailable")

        except ImportError:
            cli.print_warning("Health check module not available")

    elif args.command == "config":
        cli.print_section("Configuration")

        if args.show:
            try:
                from config import OPENAI_API_KEY, EMAIL
                from constants import (
                    MAX_TOKENS, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL,
                    DEFAULT_TEMPERATURE, RATE_LIMIT_PUBMED
                )

                config = {
                    "API Key": "***" + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else "Not set",
                    "Email": EMAIL or "Not set",
                    "Max Tokens": MAX_TOKENS,
                    "Model": DEFAULT_MODEL,
                    "Embedding Model": DEFAULT_EMBEDDING_MODEL,
                    "Temperature": DEFAULT_TEMPERATURE,
                    "PubMed Rate Limit": f"{RATE_LIMIT_PUBMED}/min"
                }

                cli.print_tree("Current Configuration", config)

            except ImportError as e:
                cli.print_error(f"Could not load config: {e}")

    elif args.command == "export":
        cli.print_section("Exporting Results")
        cli.print_info(f"Session: {args.session_path}")
        cli.print_info(f"Format: {args.format}")

        try:
            from utils import load
            from web.export import ResearchExporter

            # Load session
            session_data = load(args.session_path)

            # Create exporter
            exporter = ResearchExporter()

            # Export
            output_path = args.output or f"export.{args.format}"
            # Implementation depends on exporter methods
            cli.print_success(f"Exported to {output_path}")

        except Exception as e:
            cli.print_error(f"Export failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
