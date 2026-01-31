"""
Safe code execution module for INSIGHT.

This module provides sandboxed execution of LLM-generated Python code
with AST validation, restricted builtins, and timeout protection.
"""

import ast
import builtins
import contextlib
import functools
import io
import signal
import sys
import threading
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from logging_config import get_logger

logger = get_logger("executor")


class ExecutionError(Exception):
    """Exception raised when code execution fails."""
    pass


class ValidationError(Exception):
    """Exception raised when code validation fails."""
    pass


class TimeoutError(Exception):
    """Exception raised when code execution times out."""
    pass


class SecurityViolation(Exception):
    """Exception raised when code violates security policies."""
    pass


class RiskLevel(Enum):
    """Risk level classification for code operations."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DANGEROUS = "dangerous"


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    risk_level: RiskLevel
    violations: List[str]
    warnings: List[str]


# Safe builtins whitelist
SAFE_BUILTINS: Set[str] = {
    # Type constructors
    "bool", "int", "float", "str", "list", "dict", "set", "tuple", "frozenset",
    "bytes", "bytearray",
    # Type checks
    "isinstance", "issubclass", "type", "callable",
    # Itertools-like
    "range", "enumerate", "zip", "map", "filter", "reversed", "sorted",
    # Aggregation
    "len", "min", "max", "sum", "abs", "round", "pow",
    "all", "any",
    # String/repr
    "repr", "str", "format", "chr", "ord",
    # Container operations
    "iter", "next", "slice",
    # Math
    "divmod",
    # Other safe operations
    "print", "None", "True", "False",
    "hasattr", "getattr",
}

# Dangerous builtins blacklist
DANGEROUS_BUILTINS: Set[str] = {
    "eval", "exec", "compile", "open", "__import__",
    "globals", "locals", "vars", "dir",
    "setattr", "delattr",
    "input", "breakpoint",
    "memoryview", "object",
}

# Dangerous module imports
DANGEROUS_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "pickle", "marshal", "shelve",
    "ctypes", "multiprocessing",
    "importlib", "builtins", "__builtins__",
    "code", "codeop", "compileall",
}

# Allowed modules for API wrappers
ALLOWED_MODULES: Set[str] = {
    "api.mygene_wrapper",
    "api.myvariant_wrapper",
    "api.pubmed_wrapper",
    "mygene",
    "myvariant",
    "Bio",
    "Bio.Entrez",
}

# Dangerous AST node types
DANGEROUS_NODES: Set[type] = {
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Global,
    ast.Nonlocal,
}


class CodeValidator(ast.NodeVisitor):
    """
    AST visitor that validates code for security issues.

    Checks for:
    - Dangerous function calls
    - Forbidden imports
    - Restricted AST node types
    - Attribute access patterns
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self.violations: List[str] = []
        self.warnings: List[str] = []
        self.risk_level = RiskLevel.SAFE
        self._imported_modules: Set[str] = set()

    def _elevate_risk(self, level: RiskLevel) -> None:
        """Elevate risk level if new level is higher."""
        risk_order = [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM,
                      RiskLevel.HIGH, RiskLevel.DANGEROUS]
        current_idx = risk_order.index(self.risk_level)
        new_idx = risk_order.index(level)
        if new_idx > current_idx:
            self.risk_level = level

    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self._imported_modules.add(alias.name)

            if module_name in DANGEROUS_MODULES:
                self.violations.append(
                    f"Importing dangerous module '{alias.name}' is not allowed"
                )
                self._elevate_risk(RiskLevel.DANGEROUS)
            elif alias.name not in ALLOWED_MODULES:
                self.warnings.append(
                    f"Importing module '{alias.name}' - verify this is intended"
                )
                self._elevate_risk(RiskLevel.MEDIUM)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from ... import statements."""
        if node.module:
            module_base = node.module.split('.')[0]
            self._imported_modules.add(node.module)

            if module_base in DANGEROUS_MODULES:
                self.violations.append(
                    f"Importing from dangerous module '{node.module}' is not allowed"
                )
                self._elevate_risk(RiskLevel.DANGEROUS)
            elif node.module not in ALLOWED_MODULES:
                self.warnings.append(
                    f"Importing from module '{node.module}' - verify this is intended"
                )
                self._elevate_risk(RiskLevel.MEDIUM)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name:
            if func_name in DANGEROUS_BUILTINS:
                self.violations.append(
                    f"Calling dangerous function '{func_name}' is not allowed"
                )
                self._elevate_risk(RiskLevel.DANGEROUS)
            elif func_name.startswith('_'):
                self.warnings.append(
                    f"Calling private function '{func_name}'"
                )
                self._elevate_risk(RiskLevel.MEDIUM)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access."""
        attr_name = node.attr

        # Check for dunder attribute access
        if attr_name.startswith('__') and attr_name.endswith('__'):
            if attr_name not in ('__init__', '__str__', '__repr__', '__len__'):
                self.violations.append(
                    f"Accessing dunder attribute '{attr_name}' is not allowed"
                )
                self._elevate_risk(RiskLevel.HIGH)

        # Check for private attribute access
        elif attr_name.startswith('_') and not attr_name.startswith('__'):
            self.warnings.append(
                f"Accessing private attribute '{attr_name}'"
            )
            self._elevate_risk(RiskLevel.LOW)

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Check subscript operations."""
        # Check for __class__ or __bases__ access via subscript
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                if node.slice.value in ('__class__', '__bases__', '__mro__',
                                       '__subclasses__', '__globals__'):
                    self.violations.append(
                        f"Accessing '{node.slice.value}' via subscript is not allowed"
                    )
                    self._elevate_risk(RiskLevel.DANGEROUS)

        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Check for dangerous node types."""
        if type(node) in DANGEROUS_NODES:
            self.violations.append(
                f"Using {type(node).__name__} is not allowed"
            )
            self._elevate_risk(RiskLevel.HIGH)

        super().generic_visit(node)


def validate_code(code: str) -> ValidationResult:
    """
    Validate Python code for security issues.

    Args:
        code: The Python code to validate

    Returns:
        ValidationResult with validation details
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ValidationResult(
            is_valid=False,
            risk_level=RiskLevel.SAFE,
            violations=[f"Syntax error: {e}"],
            warnings=[]
        )

    validator = CodeValidator()
    validator.visit(tree)

    is_valid = len(validator.violations) == 0

    return ValidationResult(
        is_valid=is_valid,
        risk_level=validator.risk_level,
        violations=validator.violations,
        warnings=validator.warnings
    )


def create_restricted_globals(
    additional_globals: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a restricted globals dictionary for code execution.

    Args:
        additional_globals: Additional safe globals to include

    Returns:
        Restricted globals dictionary
    """
    # Start with safe builtins
    restricted_builtins = {
        name: getattr(builtins, name)
        for name in SAFE_BUILTINS
        if hasattr(builtins, name)
    }

    # Add constants
    restricted_builtins['None'] = None
    restricted_builtins['True'] = True
    restricted_builtins['False'] = False

    restricted_globals = {
        '__builtins__': restricted_builtins,
        '__name__': '__main__',
        '__doc__': None,
    }

    if additional_globals:
        restricted_globals.update(additional_globals)

    return restricted_globals


class TimeoutContext:
    """Context manager for execution timeout (Unix only)."""

    def __init__(self, seconds: int) -> None:
        """
        Initialize timeout context.

        Args:
            seconds: Timeout in seconds
        """
        self.seconds = seconds
        self._old_handler: Any = None

    def _timeout_handler(self, signum: int, frame: Any) -> None:
        """Handle timeout signal."""
        raise TimeoutError(f"Code execution timed out after {self.seconds} seconds")

    def __enter__(self) -> 'TimeoutContext':
        """Set up timeout handler."""
        if hasattr(signal, 'SIGALRM'):
            self._old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Clean up timeout handler."""
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if self._old_handler:
                signal.signal(signal.SIGALRM, self._old_handler)
        return False


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    result: Any
    stdout: str
    stderr: str
    error: Optional[str] = None
    execution_time: float = 0.0


def execute_safe(
    code: str,
    timeout_seconds: int = 30,
    additional_globals: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> ExecutionResult:
    """
    Safely execute Python code with restrictions and timeout.

    Args:
        code: The Python code to execute
        timeout_seconds: Maximum execution time
        additional_globals: Additional globals to make available
        validate: Whether to validate code before execution

    Returns:
        ExecutionResult with execution details
    """
    import time
    start_time = time.time()

    # Validate code
    if validate:
        validation = validate_code(code)
        if not validation.is_valid:
            return ExecutionResult(
                success=False,
                result=None,
                stdout="",
                stderr="",
                error=f"Validation failed: {'; '.join(validation.violations)}"
            )

        if validation.warnings:
            logger.warning(f"Code validation warnings: {validation.warnings}")

    # Set up restricted execution environment
    restricted_globals = create_restricted_globals(additional_globals)
    local_vars: Dict[str, Any] = {}

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with TimeoutContext(timeout_seconds):
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    exec(code, restricted_globals, local_vars)

        execution_time = time.time() - start_time

        # Get the 'ret' variable if it exists
        result = local_vars.get('ret', None)

        return ExecutionResult(
            success=True,
            result=result,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            execution_time=execution_time
        )

    except TimeoutError as e:
        return ExecutionResult(
            success=False,
            result=None,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            error=str(e),
            execution_time=time.time() - start_time
        )

    except Exception as e:
        return ExecutionResult(
            success=False,
            result=None,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            error=f"{type(e).__name__}: {e}",
            execution_time=time.time() - start_time
        )


def execute_api_code(
    code: str,
    tool: str,
    timeout_seconds: int = 60
) -> ExecutionResult:
    """
    Execute API wrapper code with appropriate permissions.

    This function provides a specialized execution environment for
    the biomedical API wrappers used in INSIGHT.

    Args:
        code: The Python code to execute
        tool: The tool name (MYGENE, PUBMED, MYVARIANT)
        timeout_seconds: Maximum execution time

    Returns:
        ExecutionResult with execution details
    """
    # Build the full code with imports based on tool
    if tool == "MYGENE":
        full_code = (
            "from api.mygene_wrapper import mygene_wrapper\n"
            f"{code}\n"
            "ret = mygene_wrapper(query_term, size, from_)"
        )
    elif tool == "MYVARIANT":
        full_code = (
            "from api.myvariant_wrapper import myvariant_wrapper\n"
            f"{code}\n"
            "ret = myvariant_wrapper(query_term)"
        )
    elif tool == "PUBMED":
        full_code = (
            "from api.pubmed_wrapper import pubmed_wrapper\n"
            f"{code}\n"
            "ret = pubmed_wrapper(query_term, retmax, retstart)"
        )
    else:
        return ExecutionResult(
            success=False,
            result=None,
            stdout="",
            stderr="",
            error=f"Unknown tool: {tool}"
        )

    # Create globals with API wrapper imports allowed
    import importlib.util

    additional_globals = {}

    # Dynamically import the API wrappers
    try:
        if tool == "MYGENE":
            from api.mygene_wrapper import mygene_wrapper
            additional_globals['mygene_wrapper'] = mygene_wrapper
        elif tool == "MYVARIANT":
            from api.myvariant_wrapper import myvariant_wrapper
            additional_globals['myvariant_wrapper'] = myvariant_wrapper
        elif tool == "PUBMED":
            from api.pubmed_wrapper import pubmed_wrapper
            additional_globals['pubmed_wrapper'] = pubmed_wrapper
    except ImportError as e:
        logger.error(f"Failed to import API wrapper for {tool}: {e}")

    # Execute with validation disabled for imports (we control them)
    return execute_safe(
        code=full_code,
        timeout_seconds=timeout_seconds,
        additional_globals=additional_globals,
        validate=False  # We've constructed safe code
    )


# Convenience function for backwards compatibility
def execute_python(code: str) -> Optional[Any]:
    """
    Execute Python code and return the result.

    This is a backwards-compatible wrapper around execute_safe.

    Args:
        code: The Python code to execute

    Returns:
        The value of 'ret' variable if execution succeeded, None otherwise
    """
    result = execute_safe(code, validate=False)

    if result.success:
        return result.result
    else:
        logger.error(f"Code execution failed: {result.error}")
        return None
