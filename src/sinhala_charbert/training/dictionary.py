"""
NLM (Noisy Language Modeling) Candidate Word Dictionary for Character Channel Denoising.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union
from sinlib.utils.preprocessing import normalize_sinhala


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
        min_freq: int = 2,
    ) -> None:
        """
        Builds vocabulary from a stream of Sinhala texts.
        """
        counts = Counter()
        for t in texts:
            if not t or not isinstance(t, str):
                continue
            norm_t = normalize_sinhala(t)
            words = norm_t.split()
            for w in words:
                cleaned = w.strip(".,;:!?\"'()[]{}«»-")
                if cleaned:
                    counts[cleaned] += 1

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
        with open(filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab_map=vocab)
