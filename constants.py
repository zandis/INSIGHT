"""
Constants and configuration values for INSIGHT.

This module centralizes all magic numbers, default values, and configuration
constants used throughout the application.
"""

from enum import Enum
from typing import Final


# =============================================================================
# Token and Model Configuration
# =============================================================================

MAX_TOKENS: Final[int] = 4097
ADA_EMBEDDING_MAX_SIZE: Final[int] = 8191
DEFAULT_CHUNK_SIZE: Final[int] = 4000
DEFAULT_MAX_TOKENS_RESPONSE: Final[int] = 6000

# Result size limits
RESULT_CUTOFF: Final[int] = 20000
MAX_GENERIF_ENTRIES: Final[int] = 20

# =============================================================================
# Model Names
# =============================================================================

class ModelName(str, Enum):
    """Supported OpenAI model names."""
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    TEXT_DAVINCI_003 = "text-davinci-003"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


DEFAULT_CHAT_MODEL: Final[str] = ModelName.GPT_35_TURBO_16K.value
DEFAULT_COMPLETION_MODEL: Final[str] = ModelName.TEXT_DAVINCI_003.value
DEFAULT_EMBEDDING_MODEL: Final[str] = ModelName.TEXT_EMBEDDING_ADA_002.value

# =============================================================================
# API Configuration
# =============================================================================

# Retry configuration
MAX_RETRIES: Final[int] = 5
RETRY_BASE_DELAY: Final[float] = 1.0
RETRY_MAX_DELAY: Final[float] = 60.0
RETRY_EXPONENTIAL_BASE: Final[float] = 2.0

# Rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 60
RATE_LIMIT_TOKENS_PER_MINUTE: Final[int] = 90000

# Timeouts (in seconds)
DEFAULT_API_TIMEOUT: Final[int] = 120
EMBEDDING_TIMEOUT: Final[int] = 60
PUBMED_TIMEOUT: Final[int] = 30
MYGENE_TIMEOUT: Final[int] = 30
MYVARIANT_TIMEOUT: Final[int] = 30

# =============================================================================
# Tool Names
# =============================================================================

class ToolName(str, Enum):
    """Available research tools."""
    MYGENE = "MYGENE"
    PUBMED = "PUBMED"
    MYVARIANT = "MYVARIANT"


AVAILABLE_TOOLS: Final[list] = [ToolName.MYGENE.value, ToolName.PUBMED.value, ToolName.MYVARIANT.value]

# =============================================================================
# File and Directory Configuration
# =============================================================================

OUTPUT_DIR: Final[str] = "out"
INDEX_FILENAME: Final[str] = "index.json"
STATE_FILENAME: Final[str] = "state.json"
KEY_FINDINGS_PREFIX: Final[str] = "key_findings"
EXECUTIVE_SUMMARY_FILENAME: Final[str] = "executive_summary.txt"
API_CALL_FILENAME: Final[str] = "api_call.txt"

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# =============================================================================
# Cache Configuration
# =============================================================================

EMBEDDING_CACHE_MAX_SIZE: Final[int] = 10000
EMBEDDING_CACHE_TTL_SECONDS: Final[int] = 86400  # 24 hours
RESULT_CACHE_MAX_SIZE: Final[int] = 1000

# =============================================================================
# Query Configuration
# =============================================================================

DEFAULT_TOP_K: Final[int] = 50
KEY_RESULTS_TOP_K: Final[int] = 20
DEFAULT_SIMILARITY_TOP_K: Final[int] = 50

# Default queries for key results compilation
KEY_RESULT_QUERIES: Final[list] = [
    "Give a brief high level summary of all the data.",
    "Briefly list all the main points that the data covers.",
    "Generate several creative hypotheses given the data.",
    "What are some high level research directions to explore further given the data?",
]

# =============================================================================
# Temperature Settings
# =============================================================================

DETERMINISTIC_TEMPERATURE: Final[float] = 0.0
CREATIVE_TEMPERATURE: Final[float] = 0.7
DATA_CLEANING_TEMPERATURE: Final[float] = 0.1

# =============================================================================
# Health Check Configuration
# =============================================================================

HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 300  # 5 minutes
API_HEALTH_CHECK_TIMEOUT: Final[int] = 10

# =============================================================================
# Circuit Breaker Configuration
# =============================================================================

CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = 60  # seconds
CIRCUIT_BREAKER_HALF_OPEN_REQUESTS: Final[int] = 3

# =============================================================================
# Memory Configuration
# =============================================================================

MAX_MEMORY_USAGE_MB: Final[int] = 2048
MEMORY_CHECK_INTERVAL_SECONDS: Final[int] = 60

# =============================================================================
# Validation Patterns
# =============================================================================

import re

VALID_EMAIL_PATTERN: Final[re.Pattern] = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

VALID_PATH_PATTERN: Final[re.Pattern] = re.compile(
    r'^[a-zA-Z0-9_\-./\\]+$'
)

# Genetic variant patterns
RSID_PATTERN: Final[re.Pattern] = re.compile(r'^rs\d+$', re.IGNORECASE)
HGVS_PATTERN: Final[re.Pattern] = re.compile(r'^chr\d+:g\.\d+[ACGT]>[ACGT]$', re.IGNORECASE)
