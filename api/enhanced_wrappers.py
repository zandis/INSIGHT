"""
Enhanced API wrappers with retry logic, rate limiting, and error handling.

This module provides production-ready wrappers for:
- PubMed API
- MyGene API
- MyVariant API
"""

import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import backoff

# Import rate limiting and metrics
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import get_logger
from rate_limiter import (
    CircuitOpenError,
    get_rate_limiter_manager,
    rate_limited,
    retry_with_backoff,
    with_circuit_breaker,
)
from metrics import get_metrics, track_api_call

logger = get_logger("api.wrappers")

F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Custom Exceptions
# =============================================================================

class APIError(Exception):
    """Base exception for API errors."""
    pass


class PubMedError(APIError):
    """Exception for PubMed API errors."""
    pass


class MyGeneError(APIError):
    """Exception for MyGene API errors."""
    pass


class MyVariantError(APIError):
    """Exception for MyVariant API errors."""
    pass


class RateLimitExceeded(APIError):
    """Exception when rate limit is exceeded."""
    pass


class APITimeoutError(APIError):
    """Exception when API call times out."""
    pass


# =============================================================================
# Response Classes
# =============================================================================

@dataclass
class APIResponse:
    """Standard API response wrapper."""
    success: bool
    data: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    cached: bool = False


# =============================================================================
# PubMed Wrapper
# =============================================================================

class PubMedWrapper:
    """
    Enhanced PubMed API wrapper with retry and rate limiting.
    """

    def __init__(self, email: Optional[str] = None):
        """
        Initialize the PubMed wrapper.

        Args:
            email: Email for Entrez API (required by NCBI)
        """
        from Bio import Entrez

        self.email = email or os.environ.get("INSIGHT_EMAIL", "user@example.com")
        Entrez.email = self.email
        self._metrics = get_metrics()
        self._rate_manager = get_rate_limiter_manager()

    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(Exception,)
    )
    @rate_limited("pubmed")
    def search(
        self,
        query: str,
        retmax: int = 10,
        retstart: int = 0
    ) -> APIResponse:
        """
        Search PubMed for articles.

        Args:
            query: Search query
            retmax: Maximum results to return
            retstart: Starting offset

        Returns:
            APIResponse with article IDs
        """
        from Bio import Entrez

        start_time = time.time()

        try:
            logger.info(f"PubMed search: '{query}' (max={retmax}, start={retstart})")

            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=retmax,
                retstart=retstart
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            pubmed_ids = search_results.get("IdList", [])
            latency = (time.time() - start_time) * 1000

            self._metrics.record_api_call(
                api_name="pubmed",
                latency_ms=latency,
                success=True
            )

            logger.info(f"PubMed search returned {len(pubmed_ids)} results")

            return APIResponse(
                success=True,
                data=pubmed_ids,
                latency_ms=latency
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._metrics.record_api_call(
                api_name="pubmed",
                latency_ms=latency,
                success=False,
                error=str(e)
            )
            logger.error(f"PubMed search failed: {e}")
            raise PubMedError(f"PubMed search failed: {e}")

    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(Exception,)
    )
    @rate_limited("pubmed")
    def fetch_abstracts(
        self,
        pubmed_ids: List[str]
    ) -> APIResponse:
        """
        Fetch abstracts for given PubMed IDs.

        Args:
            pubmed_ids: List of PubMed IDs

        Returns:
            APIResponse with abstracts XML
        """
        from Bio import Entrez

        start_time = time.time()

        try:
            if not pubmed_ids:
                return APIResponse(
                    success=True,
                    data="",
                    latency_ms=0
                )

            logger.info(f"Fetching abstracts for {len(pubmed_ids)} articles")

            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pubmed_ids,
                rettype="abstract"
            )
            abstracts = fetch_handle.read()
            fetch_handle.close()

            latency = (time.time() - start_time) * 1000

            self._metrics.record_api_call(
                api_name="pubmed",
                latency_ms=latency,
                success=True
            )

            return APIResponse(
                success=True,
                data=abstracts,
                latency_ms=latency
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._metrics.record_api_call(
                api_name="pubmed",
                latency_ms=latency,
                success=False,
                error=str(e)
            )
            logger.error(f"PubMed fetch failed: {e}")
            raise PubMedError(f"PubMed fetch failed: {e}")

    def search_and_fetch(
        self,
        query: str,
        retmax: int = 10,
        retstart: int = 0
    ) -> str:
        """
        Combined search and fetch operation.

        Args:
            query: Search query
            retmax: Maximum results
            retstart: Starting offset

        Returns:
            Abstracts XML string
        """
        search_response = self.search(query, retmax, retstart)

        if not search_response.success:
            return ""

        fetch_response = self.fetch_abstracts(search_response.data)

        return fetch_response.data if fetch_response.success else ""


# =============================================================================
# MyGene Wrapper
# =============================================================================

class MyGeneWrapper:
    """
    Enhanced MyGene API wrapper with retry and rate limiting.
    """

    def __init__(self):
        """Initialize the MyGene wrapper."""
        import mygene
        self._mg = mygene.MyGeneInfo()
        self._metrics = get_metrics()
        self._rate_manager = get_rate_limiter_manager()

    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(Exception,)
    )
    @rate_limited("mygene")
    def query(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        species: str = "human"
    ) -> APIResponse:
        """
        Query MyGene for genes.

        Args:
            query: Search query
            size: Maximum results
            from_: Starting offset
            species: Species filter

        Returns:
            APIResponse with gene hits
        """
        start_time = time.time()

        try:
            logger.info(f"MyGene query: '{query}' (size={size}, from={from_})")

            fields = "symbol,name,entrezgene,ensemblgene"
            result = self._mg.query(
                query,
                fields=fields,
                species=species,
                size=size,
                from_=from_
            )

            hits = result.get('hits', [])
            latency = (time.time() - start_time) * 1000

            self._metrics.record_api_call(
                api_name="mygene",
                latency_ms=latency,
                success=True
            )

            logger.info(f"MyGene query returned {len(hits)} hits")

            return APIResponse(
                success=True,
                data=hits,
                latency_ms=latency
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._metrics.record_api_call(
                api_name="mygene",
                latency_ms=latency,
                success=False,
                error=str(e)
            )
            logger.error(f"MyGene query failed: {e}")
            raise MyGeneError(f"MyGene query failed: {e}")

    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(Exception,)
    )
    @rate_limited("mygene")
    def get_gene(
        self,
        gene_id: str,
        fields: Optional[List[str]] = None
    ) -> APIResponse:
        """
        Get detailed gene information.

        Args:
            gene_id: Gene ID
            fields: Fields to retrieve

        Returns:
            APIResponse with gene details
        """
        start_time = time.time()

        try:
            if fields is None:
                fields = [
                    'name', 'symbol', 'type_of_gene', 'genomic_pos_hg19',
                    'refseq', 'taxid', 'generif', 'summary', 'pathway'
                ]

            logger.debug(f"MyGene get_gene: {gene_id}")

            result = self._mg.getgene(gene_id, fields=fields)
            latency = (time.time() - start_time) * 1000

            self._metrics.record_api_call(
                api_name="mygene",
                latency_ms=latency,
                success=True
            )

            return APIResponse(
                success=True,
                data=result,
                latency_ms=latency
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._metrics.record_api_call(
                api_name="mygene",
                latency_ms=latency,
                success=False,
                error=str(e)
            )
            logger.error(f"MyGene get_gene failed: {e}")
            raise MyGeneError(f"MyGene get_gene failed: {e}")

    def query_and_get_details(
        self,
        query: str,
        size: int = 10,
        from_: int = 0
    ) -> List[Dict]:
        """
        Combined query and detailed info fetch.

        Args:
            query: Search query
            size: Maximum results
            from_: Starting offset

        Returns:
            List of gene info dictionaries
        """
        query_response = self.query(query, size, from_)

        if not query_response.success:
            return []

        gene_info_list = []
        for gene in query_response.data:
            try:
                gene_response = self.get_gene(gene['_id'])
                if gene_response.success:
                    gene_info_list.append(gene_response.data)
            except Exception as e:
                logger.warning(f"Failed to get details for gene {gene.get('_id')}: {e}")
                continue

        return gene_info_list


# =============================================================================
# MyVariant Wrapper
# =============================================================================

class MyVariantWrapper:
    """
    Enhanced MyVariant API wrapper with retry and rate limiting.
    """

    def __init__(self):
        """Initialize the MyVariant wrapper."""
        import myvariant
        self._mv = myvariant.MyVariantInfo()
        self._metrics = get_metrics()
        self._rate_manager = get_rate_limiter_manager()

    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(Exception,)
    )
    @rate_limited("myvariant")
    def get_variant(self, variant_id: str) -> APIResponse:
        """
        Get variant information.

        Args:
            variant_id: Variant identifier (rsID, HGVS, etc.)

        Returns:
            APIResponse with variant data
        """
        start_time = time.time()

        try:
            logger.info(f"MyVariant query: {variant_id}")

            result = self._mv.getvariant(variant_id)
            latency = (time.time() - start_time) * 1000

            self._metrics.record_api_call(
                api_name="myvariant",
                latency_ms=latency,
                success=True
            )

            logger.debug(f"MyVariant returned result for {variant_id}")

            return APIResponse(
                success=True,
                data=result,
                latency_ms=latency
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._metrics.record_api_call(
                api_name="myvariant",
                latency_ms=latency,
                success=False,
                error=str(e)
            )
            logger.error(f"MyVariant get_variant failed: {e}")
            raise MyVariantError(f"MyVariant get_variant failed: {e}")

    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(Exception,)
    )
    @rate_limited("myvariant")
    def query(
        self,
        query: str,
        size: int = 10
    ) -> APIResponse:
        """
        Query MyVariant for variants.

        Args:
            query: Search query
            size: Maximum results

        Returns:
            APIResponse with variant hits
        """
        start_time = time.time()

        try:
            logger.info(f"MyVariant query: '{query}' (size={size})")

            result = self._mv.query(query, size=size)
            hits = result.get('hits', [])
            latency = (time.time() - start_time) * 1000

            self._metrics.record_api_call(
                api_name="myvariant",
                latency_ms=latency,
                success=True
            )

            logger.info(f"MyVariant query returned {len(hits)} hits")

            return APIResponse(
                success=True,
                data=hits,
                latency_ms=latency
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._metrics.record_api_call(
                api_name="myvariant",
                latency_ms=latency,
                success=False,
                error=str(e)
            )
            logger.error(f"MyVariant query failed: {e}")
            raise MyVariantError(f"MyVariant query failed: {e}")


# =============================================================================
# Backwards-Compatible Wrapper Functions
# =============================================================================

_pubmed_wrapper: Optional[PubMedWrapper] = None
_mygene_wrapper: Optional[MyGeneWrapper] = None
_myvariant_wrapper: Optional[MyVariantWrapper] = None


def pubmed_wrapper(query_term: str, retmax: int, retstart: int) -> str:
    """
    Backwards-compatible PubMed wrapper function.

    Args:
        query_term: Search query
        retmax: Maximum results
        retstart: Starting offset

    Returns:
        Abstracts XML string
    """
    global _pubmed_wrapper
    if _pubmed_wrapper is None:
        _pubmed_wrapper = PubMedWrapper()

    return _pubmed_wrapper.search_and_fetch(query_term, retmax, retstart)


def mygene_wrapper(query_term: str, size: int, from_: int) -> List[Dict]:
    """
    Backwards-compatible MyGene wrapper function.

    Args:
        query_term: Search query
        size: Maximum results
        from_: Starting offset

    Returns:
        List of gene info dictionaries
    """
    global _mygene_wrapper
    if _mygene_wrapper is None:
        _mygene_wrapper = MyGeneWrapper()

    return _mygene_wrapper.query_and_get_details(query_term, size, from_)


def myvariant_wrapper(query_term: str) -> Dict:
    """
    Backwards-compatible MyVariant wrapper function.

    Args:
        query_term: Variant identifier

    Returns:
        Variant data dictionary
    """
    global _myvariant_wrapper
    if _myvariant_wrapper is None:
        _myvariant_wrapper = MyVariantWrapper()

    response = _myvariant_wrapper.get_variant(query_term)
    return response.data if response.success else {}
