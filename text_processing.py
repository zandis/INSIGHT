"""
Text processing module for INSIGHT.

This module provides utilities for:
- Smart text chunking for token management
- Text deduplication
- Semantic text processing
- Token counting and management
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import tiktoken

from constants import ADA_EMBEDDING_MAX_SIZE, DEFAULT_CHUNK_SIZE, MAX_TOKENS
from logging_config import get_logger

logger = get_logger("text_processing")


# =============================================================================
# Token Counting
# =============================================================================

class TokenCounter:
    """
    Efficient token counter with caching.
    """

    _encoders: Dict[str, tiktoken.Encoding] = {}

    @classmethod
    def get_encoder(cls, encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
        """
        Get a cached encoder.

        Args:
            encoding_name: Name of the encoding

        Returns:
            Tiktoken encoder
        """
        if encoding_name not in cls._encoders:
            cls._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
        return cls._encoders[encoding_name]

    @classmethod
    def count_tokens(cls, text: str, encoding_name: str = "cl100k_base") -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens in
            encoding_name: Encoding to use

        Returns:
            Number of tokens
        """
        encoder = cls.get_encoder(encoding_name)
        return len(encoder.encode(text))

    @classmethod
    def truncate_to_tokens(
        cls,
        text: str,
        max_tokens: int,
        encoding_name: str = "cl100k_base"
    ) -> str:
        """
        Truncate text to a maximum number of tokens.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            encoding_name: Encoding to use

        Returns:
            Truncated text
        """
        encoder = cls.get_encoder(encoding_name)
        tokens = encoder.encode(text)

        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return encoder.decode(truncated_tokens)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        string: Text to count tokens in
        encoding_name: Encoding to use

    Returns:
        Number of tokens
    """
    return TokenCounter.count_tokens(string, encoding_name)


# =============================================================================
# Text Chunking
# =============================================================================

@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    start_index: int
    end_index: int
    token_count: int
    chunk_id: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """
    Smart text chunker that respects semantic boundaries.

    Features:
    - Respects sentence and paragraph boundaries
    - Configurable chunk size and overlap
    - Preserves important content structure
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Paragraph breaks
    PARAGRAPH_BREAK = re.compile(r'\n\s*\n')

    def __init__(
        self,
        max_tokens: int = DEFAULT_CHUNK_SIZE,
        overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Tokens to overlap between chunks
            encoding_name: Tiktoken encoding to use
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding_name = encoding_name

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunk text into semantically meaningful pieces.

        Args:
            text: Text to chunk

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        # First, try to split by paragraphs
        paragraphs = self.PARAGRAPH_BREAK.split(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        start_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = TokenCounter.count_tokens(para, self.encoding_name)

            # If single paragraph exceeds max, split by sentences
            if para_tokens > self.max_tokens:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        start_index=start_index,
                        end_index=start_index + len(chunk_text),
                        token_count=current_tokens,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    start_index += len(chunk_text) + 2  # +2 for paragraph break

                # Split paragraph by sentences
                sentence_chunks = self._chunk_by_sentences(para)
                for sc in sentence_chunks:
                    sc.chunk_id = chunk_id
                    chunks.append(sc)
                    chunk_id += 1

                current_chunk = []
                current_tokens = 0
                continue

            # Check if adding this paragraph exceeds max
            if current_tokens + para_tokens > self.max_tokens:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        start_index=start_index,
                        end_index=start_index + len(chunk_text),
                        token_count=current_tokens,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    start_index += len(chunk_text) + 2

                # Handle overlap
                if self.overlap_tokens > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(
                        '\n\n'.join(current_chunk),
                        self.overlap_tokens
                    )
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = TokenCounter.count_tokens(
                        overlap_text, self.encoding_name
                    ) if overlap_text else 0
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(para)
            current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_index=start_index,
                end_index=start_index + len(chunk_text),
                token_count=current_tokens,
                chunk_id=chunk_id
            ))

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def _chunk_by_sentences(self, text: str) -> List[TextChunk]:
        """Split text by sentences when paragraph is too large."""
        sentences = self.SENTENCE_ENDINGS.split(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        start_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_tokens = TokenCounter.count_tokens(sentence, self.encoding_name)

            # Single sentence too large - force split
            if sent_tokens > self.max_tokens:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        start_index=start_index,
                        end_index=start_index + len(chunk_text),
                        token_count=current_tokens,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    start_index += len(chunk_text) + 1

                # Force truncate the sentence
                truncated = TokenCounter.truncate_to_tokens(
                    sentence, self.max_tokens, self.encoding_name
                )
                chunks.append(TextChunk(
                    text=truncated,
                    start_index=start_index,
                    end_index=start_index + len(truncated),
                    token_count=self.max_tokens,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
                start_index += len(truncated) + 1
                current_chunk = []
                current_tokens = 0
                continue

            if current_tokens + sent_tokens > self.max_tokens:
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=start_index + len(chunk_text),
                    token_count=current_tokens,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
                start_index += len(chunk_text) + 1
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += sent_tokens

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_index=start_index,
                end_index=start_index + len(chunk_text),
                token_count=current_tokens,
                chunk_id=chunk_id
            ))

        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get the last N tokens of text for overlap."""
        encoder = TokenCounter.get_encoder(self.encoding_name)
        tokens = encoder.encode(text)

        if len(tokens) <= overlap_tokens:
            return text

        overlap_tokens_list = tokens[-overlap_tokens:]
        return encoder.decode(overlap_tokens_list)


# =============================================================================
# Text Deduplication
# =============================================================================

class TextDeduplicator:
    """
    Deduplicate text based on content similarity.

    Features:
    - Exact duplicate detection
    - Near-duplicate detection using MinHash
    - Configurable similarity threshold
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the deduplicator.

        Args:
            similarity_threshold: Minimum Jaccard similarity to consider duplicate
        """
        self.similarity_threshold = similarity_threshold
        self._seen_hashes: Set[str] = set()
        self._seen_shingles: List[Set[str]] = []

    def _hash_text(self, text: str) -> str:
        """Generate hash for exact matching."""
        normalized = self._normalize(text)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _get_shingles(self, text: str, n: int = 3) -> Set[str]:
        """Get n-gram shingles for near-duplicate detection."""
        normalized = self._normalize(text)
        words = normalized.split()

        if len(words) < n:
            return {normalized}

        return {
            ' '.join(words[i:i+n])
            for i in range(len(words) - n + 1)
        }

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a duplicate of previously seen text.

        Args:
            text: Text to check

        Returns:
            True if duplicate
        """
        # Check exact duplicate
        text_hash = self._hash_text(text)
        if text_hash in self._seen_hashes:
            return True

        # Check near duplicate
        shingles = self._get_shingles(text)
        for seen_shingles in self._seen_shingles:
            similarity = self._jaccard_similarity(shingles, seen_shingles)
            if similarity >= self.similarity_threshold:
                logger.debug(f"Near-duplicate found (similarity: {similarity:.2f})")
                return True

        return False

    def add(self, text: str) -> bool:
        """
        Add text to the deduplicator.

        Args:
            text: Text to add

        Returns:
            True if added (not duplicate), False if duplicate
        """
        if self.is_duplicate(text):
            return False

        text_hash = self._hash_text(text)
        self._seen_hashes.add(text_hash)
        self._seen_shingles.append(self._get_shingles(text))

        return True

    def deduplicate(self, texts: List[str]) -> List[str]:
        """
        Deduplicate a list of texts.

        Args:
            texts: List of texts to deduplicate

        Returns:
            List of unique texts
        """
        unique = []
        for text in texts:
            if self.add(text):
                unique.append(text)

        logger.info(f"Deduplicated {len(texts)} texts to {len(unique)} unique")
        return unique

    def clear(self) -> None:
        """Clear all seen texts."""
        self._seen_hashes.clear()
        self._seen_shingles.clear()


# =============================================================================
# Text Processing Utilities
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace('\x00', '')

    # Normalize whitespace
    text = ' '.join(text.split())

    # Remove control characters (except newlines)
    text = ''.join(
        c for c in text
        if c.isprintable() or c == '\n'
    )

    return text.strip()


def extract_key_sentences(
    text: str,
    max_sentences: int = 5,
    min_sentence_length: int = 20
) -> List[str]:
    """
    Extract key sentences from text.

    Simple extraction based on position and length.

    Args:
        text: Text to extract from
        max_sentences: Maximum sentences to extract
        min_sentence_length: Minimum sentence length

    Returns:
        List of key sentences
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter by length
    valid_sentences = [
        s.strip() for s in sentences
        if len(s.strip()) >= min_sentence_length
    ]

    # Take first and last sentences plus some from middle
    if len(valid_sentences) <= max_sentences:
        return valid_sentences

    result = [valid_sentences[0]]  # First sentence

    # Add some from the middle
    middle_count = max_sentences - 2
    if middle_count > 0:
        step = len(valid_sentences) // (middle_count + 1)
        for i in range(1, middle_count + 1):
            idx = i * step
            if idx < len(valid_sentences) - 1:
                result.append(valid_sentences[idx])

    result.append(valid_sentences[-1])  # Last sentence

    return result


def truncate_text_smart(
    text: str,
    max_tokens: int,
    preserve_start: bool = True,
    preserve_end: bool = True,
    encoding_name: str = "cl100k_base"
) -> str:
    """
    Smart text truncation that preserves important content.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        preserve_start: Whether to preserve the start
        preserve_end: Whether to preserve the end
        encoding_name: Token encoding to use

    Returns:
        Truncated text
    """
    token_count = TokenCounter.count_tokens(text, encoding_name)

    if token_count <= max_tokens:
        return text

    if preserve_start and preserve_end:
        # Preserve both start and end
        half_tokens = max_tokens // 2
        start_text = TokenCounter.truncate_to_tokens(text, half_tokens, encoding_name)

        # Get end portion
        encoder = TokenCounter.get_encoder(encoding_name)
        tokens = encoder.encode(text)
        end_tokens = tokens[-half_tokens:]
        end_text = encoder.decode(end_tokens)

        return f"{start_text}\n\n[...truncated...]\n\n{end_text}"

    elif preserve_start:
        return TokenCounter.truncate_to_tokens(text, max_tokens, encoding_name)

    elif preserve_end:
        encoder = TokenCounter.get_encoder(encoding_name)
        tokens = encoder.encode(text)
        end_tokens = tokens[-max_tokens:]
        return encoder.decode(end_tokens)

    else:
        return TokenCounter.truncate_to_tokens(text, max_tokens, encoding_name)


def prepare_for_embedding(
    text: str,
    max_tokens: int = ADA_EMBEDDING_MAX_SIZE
) -> str:
    """
    Prepare text for embedding by cleaning and truncating.

    Args:
        text: Text to prepare
        max_tokens: Maximum tokens for embedding model

    Returns:
        Prepared text
    """
    # Clean the text
    text = clean_text(text)

    # Replace newlines with spaces for embedding
    text = text.replace('\n', ' ')

    # Truncate if necessary
    if TokenCounter.count_tokens(text) > max_tokens:
        text = truncate_text_smart(
            text,
            max_tokens,
            preserve_start=True,
            preserve_end=True
        )

    return text


# =============================================================================
# Batch Processing
# =============================================================================

def batch_texts(
    texts: List[str],
    max_batch_tokens: int = 8000,
    encoding_name: str = "cl100k_base"
) -> List[List[str]]:
    """
    Batch texts for efficient API calls.

    Args:
        texts: List of texts to batch
        max_batch_tokens: Maximum tokens per batch
        encoding_name: Token encoding to use

    Returns:
        List of batches (each batch is a list of texts)
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        text_tokens = TokenCounter.count_tokens(text, encoding_name)

        if current_tokens + text_tokens > max_batch_tokens:
            if current_batch:
                batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens

    if current_batch:
        batches.append(current_batch)

    logger.debug(f"Created {len(batches)} batches from {len(texts)} texts")
    return batches
