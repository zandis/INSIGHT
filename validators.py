"""
Data validation module for INSIGHT.

This module provides validation schemas and utilities for:
- API responses
- User input
- Configuration data
- File paths and content
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from constants import (
    AVAILABLE_TOOLS,
    HGVS_PATTERN,
    MAX_TOKENS,
    RESULT_CUTOFF,
    RSID_PATTERN,
    VALID_EMAIL_PATTERN,
)
from logging_config import get_logger

logger = get_logger("validators")

T = TypeVar('T')


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(f"{field or 'Validation'}: {message}")


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    issues: List[ValidationIssue]

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


# =============================================================================
# Base Validator Classes
# =============================================================================

class Validator:
    """Base validator class."""

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a value.

        Args:
            value: Value to validate

        Returns:
            ValidationResult
        """
        raise NotImplementedError

    def __call__(self, value: Any) -> ValidationResult:
        """Allow calling validator as function."""
        return self.validate(value)


class CompositeValidator(Validator):
    """Validator that combines multiple validators."""

    def __init__(self, *validators: Validator):
        """
        Initialize with child validators.

        Args:
            *validators: Validators to combine
        """
        self.validators = validators

    def validate(self, value: Any) -> ValidationResult:
        """Run all validators and combine results."""
        all_issues: List[ValidationIssue] = []

        for validator in self.validators:
            result = validator.validate(value)
            all_issues.extend(result.issues)

        is_valid = not any(
            i.severity == ValidationSeverity.ERROR for i in all_issues
        )

        return ValidationResult(is_valid=is_valid, issues=all_issues)


# =============================================================================
# Type Validators
# =============================================================================

class TypeValidator(Validator):
    """Validate value is of expected type."""

    def __init__(
        self,
        expected_type: Type,
        field_name: str = "value",
        allow_none: bool = False
    ):
        self.expected_type = expected_type
        self.field_name = field_name
        self.allow_none = allow_none

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if value is None:
            if not self.allow_none:
                issues.append(ValidationIssue(
                    field=self.field_name,
                    message=f"Value cannot be None, expected {self.expected_type.__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=value
                ))
        elif not isinstance(value, self.expected_type):
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Expected {self.expected_type.__name__}, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)


class StringValidator(Validator):
    """Validate string values."""

    def __init__(
        self,
        field_name: str = "value",
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[re.Pattern] = None,
        allow_empty: bool = True,
        allow_none: bool = False
    ):
        self.field_name = field_name
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.allow_empty = allow_empty
        self.allow_none = allow_none

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if value is None:
            if not self.allow_none:
                issues.append(ValidationIssue(
                    field=self.field_name,
                    message="Value cannot be None",
                    severity=ValidationSeverity.ERROR,
                    value=value
                ))
            return ValidationResult(is_valid=len(issues) == 0, issues=issues)

        if not isinstance(value, str):
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Expected string, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        if not self.allow_empty and len(value) == 0:
            issues.append(ValidationIssue(
                field=self.field_name,
                message="Value cannot be empty",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        if self.min_length is not None and len(value) < self.min_length:
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Length must be at least {self.min_length}, got {len(value)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        if self.max_length is not None and len(value) > self.max_length:
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Length must be at most {self.max_length}, got {len(value)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        if self.pattern is not None and not self.pattern.match(value):
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Value does not match required pattern",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)


class NumberValidator(Validator):
    """Validate numeric values."""

    def __init__(
        self,
        field_name: str = "value",
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allow_float: bool = True,
        allow_none: bool = False
    ):
        self.field_name = field_name
        self.min_value = min_value
        self.max_value = max_value
        self.allow_float = allow_float
        self.allow_none = allow_none

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if value is None:
            if not self.allow_none:
                issues.append(ValidationIssue(
                    field=self.field_name,
                    message="Value cannot be None",
                    severity=ValidationSeverity.ERROR,
                    value=value
                ))
            return ValidationResult(is_valid=len(issues) == 0, issues=issues)

        if not isinstance(value, (int, float)):
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Expected number, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        if not self.allow_float and isinstance(value, float):
            issues.append(ValidationIssue(
                field=self.field_name,
                message="Value must be an integer, not float",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        if self.min_value is not None and value < self.min_value:
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Value must be at least {self.min_value}, got {value}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        if self.max_value is not None and value > self.max_value:
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Value must be at most {self.max_value}, got {value}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)


class ListValidator(Validator):
    """Validate list values."""

    def __init__(
        self,
        field_name: str = "value",
        item_validator: Optional[Validator] = None,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        allow_empty: bool = True,
        allow_none: bool = False
    ):
        self.field_name = field_name
        self.item_validator = item_validator
        self.min_items = min_items
        self.max_items = max_items
        self.allow_empty = allow_empty
        self.allow_none = allow_none

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if value is None:
            if not self.allow_none:
                issues.append(ValidationIssue(
                    field=self.field_name,
                    message="Value cannot be None",
                    severity=ValidationSeverity.ERROR,
                    value=value
                ))
            return ValidationResult(is_valid=len(issues) == 0, issues=issues)

        if not isinstance(value, list):
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"Expected list, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        if not self.allow_empty and len(value) == 0:
            issues.append(ValidationIssue(
                field=self.field_name,
                message="List cannot be empty",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        if self.min_items is not None and len(value) < self.min_items:
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"List must have at least {self.min_items} items",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        if self.max_items is not None and len(value) > self.max_items:
            issues.append(ValidationIssue(
                field=self.field_name,
                message=f"List must have at most {self.max_items} items",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        # Validate each item
        if self.item_validator:
            for i, item in enumerate(value):
                result = self.item_validator.validate(item)
                for issue in result.issues:
                    issue.field = f"{self.field_name}[{i}].{issue.field}"
                    issues.append(issue)

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)


# =============================================================================
# Domain-Specific Validators
# =============================================================================

class ObjectiveValidator(Validator):
    """Validate research objective."""

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        # Type check
        if not isinstance(value, str):
            issues.append(ValidationIssue(
                field="objective",
                message=f"Expected string, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        # Empty check
        if not value or not value.strip():
            issues.append(ValidationIssue(
                field="objective",
                message="Objective cannot be empty",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        # Length check
        if len(value) < 10:
            issues.append(ValidationIssue(
                field="objective",
                message="Objective should be more descriptive (at least 10 characters)",
                severity=ValidationSeverity.WARNING,
                value=value
            ))

        if len(value) > 5000:
            issues.append(ValidationIssue(
                field="objective",
                message="Objective is too long (max 5000 characters)",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        # Content checks
        if not any(c.isalpha() for c in value):
            issues.append(ValidationIssue(
                field="objective",
                message="Objective should contain alphabetic characters",
                severity=ValidationSeverity.WARNING,
                value=value
            ))

        return ValidationResult(
            is_valid=not any(i.severity == ValidationSeverity.ERROR for i in issues),
            issues=issues
        )


class ToolValidator(Validator):
    """Validate tool selection."""

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if not isinstance(value, list):
            issues.append(ValidationIssue(
                field="tools",
                message=f"Expected list, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        if len(value) == 0:
            issues.append(ValidationIssue(
                field="tools",
                message="At least one tool must be selected",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        for tool in value:
            if tool not in AVAILABLE_TOOLS:
                issues.append(ValidationIssue(
                    field="tools",
                    message=f"Unknown tool: {tool}. Available: {AVAILABLE_TOOLS}",
                    severity=ValidationSeverity.ERROR,
                    value=tool
                ))

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)


class TaskValidator(Validator):
    """Validate task string."""

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if not isinstance(value, str):
            issues.append(ValidationIssue(
                field="task",
                message=f"Expected string, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        if not value.strip():
            issues.append(ValidationIssue(
                field="task",
                message="Task cannot be empty",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        # Check for tool prefix
        has_tool_prefix = any(
            value.upper().startswith(f"{tool}:")
            for tool in AVAILABLE_TOOLS
        )

        if not has_tool_prefix:
            issues.append(ValidationIssue(
                field="task",
                message=f"Task should start with a tool prefix (e.g., 'PUBMED:', 'MYGENE:')",
                severity=ValidationSeverity.WARNING,
                value=value
            ))

        return ValidationResult(
            is_valid=not any(i.severity == ValidationSeverity.ERROR for i in issues),
            issues=issues
        )


class FilePathValidator(Validator):
    """Validate file paths."""

    def __init__(
        self,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allowed_extensions: Optional[List[str]] = None
    ):
        self.must_exist = must_exist
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir
        self.allowed_extensions = allowed_extensions

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if not isinstance(value, (str, Path)):
            issues.append(ValidationIssue(
                field="path",
                message=f"Expected string or Path, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        path = Path(value)

        if self.must_exist and not path.exists():
            issues.append(ValidationIssue(
                field="path",
                message=f"Path does not exist: {value}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        if path.exists():
            if self.must_be_file and not path.is_file():
                issues.append(ValidationIssue(
                    field="path",
                    message=f"Path is not a file: {value}",
                    severity=ValidationSeverity.ERROR,
                    value=value
                ))

            if self.must_be_dir and not path.is_dir():
                issues.append(ValidationIssue(
                    field="path",
                    message=f"Path is not a directory: {value}",
                    severity=ValidationSeverity.ERROR,
                    value=value
                ))

        if self.allowed_extensions:
            ext = path.suffix.lower()
            if ext not in [e.lower() for e in self.allowed_extensions]:
                issues.append(ValidationIssue(
                    field="path",
                    message=f"Invalid file extension: {ext}. Allowed: {self.allowed_extensions}",
                    severity=ValidationSeverity.ERROR,
                    value=value
                ))

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)


class GeneticVariantValidator(Validator):
    """Validate genetic variant identifiers."""

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if not isinstance(value, str):
            issues.append(ValidationIssue(
                field="variant",
                message=f"Expected string, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        value = value.strip()

        # Check for valid formats
        is_rsid = RSID_PATTERN.match(value) is not None
        is_hgvs = HGVS_PATTERN.match(value) is not None

        if not is_rsid and not is_hgvs:
            issues.append(ValidationIssue(
                field="variant",
                message="Invalid variant format. Use rsID (e.g., rs123456) or HGVS (e.g., chr1:g.12345A>G)",
                severity=ValidationSeverity.WARNING,
                value=value
            ))

        return ValidationResult(
            is_valid=not any(i.severity == ValidationSeverity.ERROR for i in issues),
            issues=issues
        )


class EmailValidator(Validator):
    """Validate email addresses."""

    def validate(self, value: Any) -> ValidationResult:
        issues = []

        if not isinstance(value, str):
            issues.append(ValidationIssue(
                field="email",
                message=f"Expected string, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return ValidationResult(is_valid=False, issues=issues)

        if not VALID_EMAIL_PATTERN.match(value):
            issues.append(ValidationIssue(
                field="email",
                message=f"Invalid email address: {value}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_objective(objective: str) -> ValidationResult:
    """Validate a research objective."""
    return ObjectiveValidator().validate(objective)


def validate_tools(tools: List[str]) -> ValidationResult:
    """Validate tool selection."""
    return ToolValidator().validate(tools)


def validate_task(task: str) -> ValidationResult:
    """Validate a task string."""
    return TaskValidator().validate(task)


def validate_file_path(
    path: str,
    must_exist: bool = False,
    allowed_extensions: Optional[List[str]] = None
) -> ValidationResult:
    """Validate a file path."""
    return FilePathValidator(
        must_exist=must_exist,
        must_be_file=True,
        allowed_extensions=allowed_extensions
    ).validate(path)


def validate_or_raise(
    value: Any,
    validator: Validator,
    error_class: Type[Exception] = ValidationError
) -> None:
    """
    Validate a value and raise exception if invalid.

    Args:
        value: Value to validate
        validator: Validator to use
        error_class: Exception class to raise

    Raises:
        error_class: If validation fails
    """
    result = validator.validate(value)

    if not result.is_valid:
        error_messages = [
            f"{issue.field}: {issue.message}"
            for issue in result.errors
        ]
        raise error_class("\n".join(error_messages))


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize a string for safe use.

    Args:
        value: String to sanitize
        max_length: Maximum length

    Returns:
        Sanitized string
    """
    # Remove null bytes
    value = value.replace('\x00', '')

    # Trim to max length
    value = value[:max_length]

    # Remove control characters (except newlines and tabs)
    value = ''.join(
        c for c in value
        if c.isprintable() or c in '\n\t'
    )

    return value.strip()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe filesystem use.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = os.path.basename(filename)

    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

    # Remove leading dots
    filename = filename.lstrip('.')

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext

    return filename or "unnamed"
