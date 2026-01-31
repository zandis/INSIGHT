"""
Configuration validation module for INSIGHT.

This module provides comprehensive validation of configuration settings
at startup, ensuring all required values are present and valid.
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from constants import (
    AVAILABLE_TOOLS,
    DEFAULT_API_TIMEOUT,
    DEFAULT_CHAT_MODEL,
    MAX_TOKENS,
    RESULT_CUTOFF,
    VALID_EMAIL_PATTERN,
)


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    field_name: Optional[str] = None


@dataclass
class InsightConfig:
    """
    Configuration dataclass for INSIGHT application.

    Provides validated, type-safe access to all configuration values.
    """

    # API Keys
    openai_api_key: str = ""
    email: str = ""

    # Model Configuration
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_model: str = "text-embedding-ada-002"
    completion_model: str = "text-davinci-003"

    # Processing Configuration
    max_tokens: int = MAX_TOKENS
    result_cutoff: int = RESULT_CUTOFF
    max_iterations: int = 10
    default_top_k: int = 50

    # Timeout Configuration
    api_timeout: int = DEFAULT_API_TIMEOUT
    embedding_timeout: int = 60
    pubmed_timeout: int = 30

    # Tool Configuration
    enabled_tools: List[str] = field(default_factory=lambda: list(AVAILABLE_TOOLS))

    # Output Configuration
    output_dir: str = "out"
    log_dir: str = "logs"

    # Feature Flags
    enable_caching: bool = True
    enable_rate_limiting: bool = True
    enable_metrics: bool = True
    enable_health_checks: bool = True
    debug_mode: bool = False

    # Rate Limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._load_from_environment()

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # API Keys (prefer env vars over config file)
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

        # Email
        if not self.email:
            self.email = os.environ.get("INSIGHT_EMAIL", "help@insightmaker.ai")

        # Debug mode
        if os.environ.get("INSIGHT_DEBUG", "").lower() in ("1", "true", "yes"):
            self.debug_mode = True

        # Model overrides
        if os.environ.get("INSIGHT_CHAT_MODEL"):
            self.chat_model = os.environ["INSIGHT_CHAT_MODEL"]

        # Timeout overrides
        if os.environ.get("INSIGHT_API_TIMEOUT"):
            try:
                self.api_timeout = int(os.environ["INSIGHT_API_TIMEOUT"])
            except ValueError:
                pass


class ConfigValidator:
    """
    Validates configuration settings for the INSIGHT application.

    Performs comprehensive validation of all configuration values,
    ensuring they meet requirements and are properly formatted.
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self._errors: List[str] = []
        self._warnings: List[str] = []

    def validate(self, config: InsightConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all configuration settings.

        Args:
            config: The configuration to validate

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self._errors = []
        self._warnings = []

        # Run all validation checks
        self._validate_api_key(config)
        self._validate_email(config)
        self._validate_models(config)
        self._validate_numeric_settings(config)
        self._validate_tools(config)
        self._validate_directories(config)
        self._validate_timeouts(config)
        self._validate_rate_limits(config)

        return len(self._errors) == 0, self._errors, self._warnings

    def validate_or_raise(self, config: InsightConfig) -> None:
        """
        Validate configuration and raise exception if invalid.

        Args:
            config: The configuration to validate

        Raises:
            ConfigError: If validation fails
        """
        is_valid, errors, warnings = self.validate(config)

        if warnings:
            # Log warnings but don't fail
            for warning in warnings:
                print(f"Config Warning: {warning}")

        if not is_valid:
            raise ConfigError(errors)

    def _validate_api_key(self, config: InsightConfig) -> None:
        """Validate the OpenAI API key."""
        if not config.openai_api_key:
            self._errors.append(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide it in config.py"
            )
        elif not config.openai_api_key.startswith("sk-"):
            self._warnings.append(
                "OpenAI API key does not start with 'sk-'. This may be invalid."
            )
        elif len(config.openai_api_key) < 20:
            self._errors.append(
                "OpenAI API key appears to be too short. Please check the value."
            )

    def _validate_email(self, config: InsightConfig) -> None:
        """Validate the email address (required for PubMed API)."""
        if not config.email:
            self._warnings.append(
                "Email is not set. This is required for PubMed API access."
            )
        elif not VALID_EMAIL_PATTERN.match(config.email):
            self._errors.append(
                f"Email '{config.email}' is not a valid email address."
            )

    def _validate_models(self, config: InsightConfig) -> None:
        """Validate model names."""
        valid_chat_models = {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
        }

        valid_embedding_models = {
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        }

        if config.chat_model not in valid_chat_models:
            self._warnings.append(
                f"Chat model '{config.chat_model}' is not a recognized model. "
                f"Valid options: {valid_chat_models}"
            )

        if config.embedding_model not in valid_embedding_models:
            self._warnings.append(
                f"Embedding model '{config.embedding_model}' is not a recognized model. "
                f"Valid options: {valid_embedding_models}"
            )

    def _validate_numeric_settings(self, config: InsightConfig) -> None:
        """Validate numeric configuration values."""
        if config.max_tokens <= 0:
            self._errors.append(f"max_tokens must be positive, got {config.max_tokens}")

        if config.max_tokens > 128000:
            self._warnings.append(
                f"max_tokens ({config.max_tokens}) exceeds typical model limits"
            )

        if config.result_cutoff <= 0:
            self._errors.append(f"result_cutoff must be positive, got {config.result_cutoff}")

        if config.max_iterations <= 0:
            self._errors.append(f"max_iterations must be positive, got {config.max_iterations}")

        if config.max_iterations > 100:
            self._warnings.append(
                f"max_iterations ({config.max_iterations}) is very high, "
                "which may result in long execution times and high costs"
            )

        if config.default_top_k <= 0:
            self._errors.append(f"default_top_k must be positive, got {config.default_top_k}")

    def _validate_tools(self, config: InsightConfig) -> None:
        """Validate tool configuration."""
        if not config.enabled_tools:
            self._errors.append("At least one tool must be enabled")

        invalid_tools = set(config.enabled_tools) - set(AVAILABLE_TOOLS)
        if invalid_tools:
            self._errors.append(
                f"Invalid tools specified: {invalid_tools}. "
                f"Available tools: {AVAILABLE_TOOLS}"
            )

    def _validate_directories(self, config: InsightConfig) -> None:
        """Validate directory configuration."""
        # Check output directory
        output_path = Path(config.output_dir)
        if output_path.exists() and not output_path.is_dir():
            self._errors.append(
                f"Output path '{config.output_dir}' exists but is not a directory"
            )

        # Check log directory
        log_path = Path(config.log_dir)
        if log_path.exists() and not log_path.is_dir():
            self._errors.append(
                f"Log path '{config.log_dir}' exists but is not a directory"
            )

    def _validate_timeouts(self, config: InsightConfig) -> None:
        """Validate timeout settings."""
        if config.api_timeout <= 0:
            self._errors.append(f"api_timeout must be positive, got {config.api_timeout}")

        if config.api_timeout > 600:
            self._warnings.append(
                f"api_timeout ({config.api_timeout}s) is very long. "
                "Consider reducing to avoid hung connections."
            )

        if config.embedding_timeout <= 0:
            self._errors.append(
                f"embedding_timeout must be positive, got {config.embedding_timeout}"
            )

        if config.pubmed_timeout <= 0:
            self._errors.append(
                f"pubmed_timeout must be positive, got {config.pubmed_timeout}"
            )

    def _validate_rate_limits(self, config: InsightConfig) -> None:
        """Validate rate limiting settings."""
        if config.enable_rate_limiting:
            if config.requests_per_minute <= 0:
                self._errors.append(
                    f"requests_per_minute must be positive, got {config.requests_per_minute}"
                )

            if config.tokens_per_minute <= 0:
                self._errors.append(
                    f"tokens_per_minute must be positive, got {config.tokens_per_minute}"
                )


def load_config() -> InsightConfig:
    """
    Load and validate configuration.

    Returns:
        Validated InsightConfig instance

    Raises:
        ConfigError: If configuration is invalid
    """
    # Import from config.py if available
    try:
        from config import OPENAI_API_KEY, EMAIL
        config = InsightConfig(
            openai_api_key=OPENAI_API_KEY or "",
            email=EMAIL or ""
        )
    except ImportError:
        config = InsightConfig()

    # Validate
    validator = ConfigValidator()
    validator.validate_or_raise(config)

    return config


def get_config() -> InsightConfig:
    """
    Get the current configuration, loading it if necessary.

    Returns:
        InsightConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


# Global config instance
_config_instance: Optional[InsightConfig] = None


# Convenience function for quick validation
def validate_startup() -> bool:
    """
    Perform startup validation checks.

    Returns:
        True if validation passes, False otherwise
    """
    try:
        load_config()
        return True
    except ConfigError as e:
        print(f"Startup validation failed: {e}")
        return False
