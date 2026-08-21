"""
NLM (Noisy Language Modeling) Candidate Word Dictionary for Character Channel Denoising.
Optimized for high-throughput parallel corpus processing (Wikipedia / large datasets).
"""

from collections import Counter
import concurrent.futures
import itertools
import json
import os
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Union
from sinlib.utils.preprocessing import normalize_sinhala

# Regex matching valid Sinhala words (including ZWJ ligatures) and alphanumeric tokens
WORD_EXTRACTION_REGEX = re.compile(r'[\u0D80-\u0DFA\u0DCA-\u0DDF\u0DF2-\u0DF4\u200D\w]+')


def _count_words_in_text_chunk(chunk_texts: List[str]) -> Counter:
    """
    Top-level picklable worker function for multiprocessing.
    Joins chunk texts, normalizes in batch, and extracts words using C-speed regex.
    """
    valid_texts = [t for t in chunk_texts if t and isinstance(t, str)]
    if not valid_texts:
        return Counter()

    # Batch join and normalize to minimize function call overhead
    joined = "\n".join(valid_texts)
    norm = normalize_sinhala(joined)
    words = WORD_EXTRACTION_REGEX.findall(norm)
    return Counter(words)


class SinhalaNLMDictionary:
    """
    Vocabulary dictionary of frequent Sinhala words used for NLM classification.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

    def __init__(self, vocab_map: Optional[Dict[str, int]] = None):
        if vocab_map is not None:
            self.vocab_map = dict(vocab_map)
        else:
            self.vocab_map = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}

        self.inv_vocab_map = {v: k for k, v in self.vocab_map.items()}
        self.pad_id = self.vocab_map[self.PAD_TOKEN]
        self.unk_id = self.vocab_map[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_map)

    def build_from_corpus(
        self,
        texts: Iterable[str],
        max_words: int = 32000,
        min_freq: int = 1,
        chunk_size: int = 5000,
        num_workers: Optional[int] = None,
    ) -> None:
        """
        High-performance parallel vocabulary builder from a stream or list of Sinhala texts.
        Uses C-accelerated regex and multiprocessing worker pools to process millions of words in seconds.

        Parameters
        ----------
        texts : Iterable[str]
            Corpus of Sinhala texts (e.g. Wikipedia articles, sentences, or streams).
        max_words : int
            Maximum number of vocabulary words to include.
        min_freq : int
            Minimum frequency threshold for a word to be added.
        chunk_size : int
            Batch size per worker chunk.
        num_workers : Optional[int]
            Number of parallel processes to use (default: CPU cores, capped at 16).
        """
        if num_workers is None:
            num_workers = min(os.cpu_count() or 1, 16)

        def _chunk_generator(iterable, size):
            iterator = iter(iterable)
            while True:
                chunk = list(itertools.islice(iterator, size))
                if not chunk:
                    break
                yield chunk

        counts = Counter()
        chunks = _chunk_generator(texts, chunk_size)

        if num_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                for chunk_counts in executor.map(_count_words_in_text_chunk, chunks):
                    counts.update(chunk_counts)
        else:
            for chunk in chunks:
                counts.update(_count_words_in_text_chunk(chunk))

        self.vocab_map = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        for word, count in counts.most_common(max_words - len(self.SPECIAL_TOKENS)):
            if count >= min_freq and word not in self.vocab_map:
                self.vocab_map[word] = len(self.vocab_map)

        self.inv_vocab_map = {v: k for k, v in self.vocab_map.items()}

    def word_to_id(self, word: str) -> int:
        """Looks up word ID with fallback to UNK ID."""
        return self.vocab_map.get(word, self.unk_id)

    def id_to_word(self, word_id: int) -> str:
        """Looks up word string with fallback to UNK token."""
        return self.inv_vocab_map.get(word_id, self.UNK_TOKEN)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves dictionary vocabulary to a JSON file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab_map, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SinhalaNLMDictionary":
        """Loads dictionary vocabulary from a JSON file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab_map=vocab)

    def load_vocab(self, filepath: Union[str, Path]) -> "SinhalaNLMDictionary":
        """Loads dictionary vocabulary in-place from a JSON file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            self.vocab_map = json.load(f)
        self.inv_vocab_map = {v: k for k, v in self.vocab_map.items()}
        self.pad_id = self.vocab_map.get(self.PAD_TOKEN, 0)
        self.unk_id = self.vocab_map.get(self.UNK_TOKEN, 1)
        return self
