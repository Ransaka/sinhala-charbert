"""
Sinhala-Aware Sentence Splitting and Text Chunking Utilities.
Provides robust segmentation for long documents (Wikipedia articles, paragraphs)
to fit within Transformer context length limits during pre-training and downstream inference.
"""

import re
from typing import List, Tuple


# Regex pattern matching sentence delimiters:
# - Full stop (.) followed by whitespace, quotes, or EOF
# - Question mark (?), exclamation mark (!), danda (।), or newlines
SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<=[.?!।\n])\s+'
)

# Clause delimiters for sub-splitting overly long sentences
CLAUSE_SPLIT_PATTERN = re.compile(
    r'(?<=[,;:—\-])\s+'
)


def split_sentences_with_spans(
    text: str,
    max_words_per_chunk: int = 50,
) -> List[Tuple[str, int, int]]:
    """
    Splits input text into sentence chunks and returns their exact (text, start_idx, end_idx) spans.

    Parameters
    ----------
    text : str
        Input Sinhala text (document, paragraph, or sentence).
    max_words_per_chunk : int
        Approximate maximum word count per chunk. Sentences longer than this
        will be further partitioned at clause boundaries or whitespace.

    Returns
    -------
    List[Tuple[str, int, int]]
        List of tuples: (chunk_text, start_char_offset, end_char_offset).
    """
    if not text or not text.strip():
        return []

    # Step 1: Find primary sentence boundaries
    # Using regex search iter to locate start and end indices of sentences
    spans: List[Tuple[int, int]] = []
    curr_start = 0

    for match in SENTENCE_SPLIT_PATTERN.finditer(text):
        end = match.start()
        # Find non-whitespace slice within [curr_start, end]
        seg = text[curr_start:end]
        if seg.strip():
            # Adjust start/end to strip leading/trailing whitespace
            l_strip = len(seg) - len(seg.lstrip())
            r_strip = len(seg) - len(seg.rstrip())
            spans.append((curr_start + l_strip, end - r_strip))
        curr_start = match.end()

    # Add final segment
    if curr_start < len(text):
        seg = text[curr_start:]
        if seg.strip():
            l_strip = len(seg) - len(seg.lstrip())
            r_strip = len(seg) - len(seg.rstrip())
            spans.append((curr_start + l_strip, len(text) - r_strip))

    if not spans:
        # Fallback if no delimiter matched but text is non-empty
        l_strip = len(text) - len(text.lstrip())
        r_strip = len(text) - len(text.rstrip())
        spans.append((l_strip, len(text) - r_strip))

    # Step 2: Check for overly long sentences and sub-split if necessary
    final_spans: List[Tuple[str, int, int]] = []

    for s_start, s_end in spans:
        sent_str = text[s_start:s_end]
        words = sent_str.split()

        if len(words) <= max_words_per_chunk:
            final_spans.append((sent_str, s_start, s_end))
        else:
            # Sub-split long sentence at clause boundaries
            sub_spans = _sub_split_long_sentence(text, s_start, s_end, max_words_per_chunk)
            final_spans.extend(sub_spans)

    return final_spans


def _sub_split_long_sentence(
    text: str,
    s_start: int,
    s_end: int,
    max_words_per_chunk: int,
) -> List[Tuple[str, int, int]]:
    """Sub-partitions a long sentence across clause boundaries or whitespace."""
    sent_text = text[s_start:s_end]
    results: List[Tuple[str, int, int]] = []

    # Try clause split first
    clause_spans: List[Tuple[int, int]] = []
    curr_start = s_start

    for match in CLAUSE_SPLIT_PATTERN.finditer(sent_text):
        abs_end = s_start + match.start()
        seg = text[curr_start:abs_end]
        if seg.strip():
            l_strip = len(seg) - len(seg.lstrip())
            r_strip = len(seg) - len(seg.rstrip())
            clause_spans.append((curr_start + l_strip, abs_end - r_strip))
        curr_start = s_start + match.end()

    if curr_start < s_end:
        seg = text[curr_start:s_end]
        if seg.strip():
            l_strip = len(seg) - len(seg.lstrip())
            r_strip = len(seg) - len(seg.rstrip())
            clause_spans.append((curr_start + l_strip, s_end - r_strip))

    if not clause_spans:
        clause_spans = [(s_start, s_end)]

    # Group/split clauses to respect max_words_per_chunk
    curr_chunk_start: Optional[int] = None
    curr_chunk_end: Optional[int] = None
    curr_word_count = 0

    for c_start, c_end in clause_spans:
        clause_str = text[c_start:c_end]
        c_words = len(clause_str.split())

        if curr_chunk_start is None:
            curr_chunk_start = c_start
            curr_chunk_end = c_end
            curr_word_count = c_words
        elif curr_word_count + c_words <= max_words_per_chunk:
            curr_chunk_end = c_end
            curr_word_count += c_words
        else:
            # Yield current chunk
            results.append((text[curr_chunk_start:curr_chunk_end], curr_chunk_start, curr_chunk_end))
            curr_chunk_start = c_start
            curr_chunk_end = c_end
            curr_word_count = c_words

    if curr_chunk_start is not None and curr_chunk_end is not None:
        results.append((text[curr_chunk_start:curr_chunk_end], curr_chunk_start, curr_chunk_end))

    return results


def split_sentences(
    text: str,
    max_words_per_chunk: int = 50,
) -> List[str]:
    """
    Splits text into clean sentence strings suitable for model training or inference.

    Parameters
    ----------
    text : str
        Input Sinhala text.
    max_words_per_chunk : int
        Maximum word count per chunk.

    Returns
    -------
    List[str]
        List of clean sentence chunks.
    """
    spans = split_sentences_with_spans(text, max_words_per_chunk=max_words_per_chunk)
    return [chunk_text for chunk_text, _, _ in spans if chunk_text.strip()]
