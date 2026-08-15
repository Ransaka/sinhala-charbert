"""
Comprehensive Evaluation Metrics for Sinhala Typo Detection and Correction.
Implements CER, WER, Akshara Error Rate (AER via sinlib), and CoNLL/BEA edit-level Precision, Recall, F0.5, and F1.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from sinlib.utils.preprocessing import normalize_sinhala, process_text


def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """
    Computes standard Levenshtein edit distance between two sequences (characters, tokens, or Aksharas).
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[n][m]


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Computes Character Error Rate (CER) = Levenshtein(ref, hyp) / max(len(ref), 1).
    """
    ref_norm = normalize_sinhala(reference)
    hyp_norm = normalize_sinhala(hypothesis)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    dist = levenshtein_distance(list(ref_norm), list(hyp_norm))
    return dist / len(ref_norm)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Computes Word Error Rate (WER) = Levenshtein(ref_words, hyp_words) / max(len(ref_words), 1).
    """
    ref_words = normalize_sinhala(reference).split()
    hyp_words = normalize_sinhala(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_aer(reference: str, hypothesis: str) -> float:
    """
    Computes Akshara Error Rate (AER) over sinlib phonological grapheme clusters.
    AER accurately measures Brahmic script errors by treating consonants with combining
    vowels and ligatures as atomic phonetic units rather than raw Unicode points.
    """
    ref_units = process_text(normalize_sinhala(reference))
    hyp_units = process_text(normalize_sinhala(hypothesis))
    if not ref_units:
        return 0.0 if not hyp_units else 1.0

    dist = levenshtein_distance(ref_units, hyp_units)
    return dist / len(ref_units)


@dataclass
class CorrectionFScore:
    """Edit-level detection and correction precision, recall, and F-measures."""
    precision: float
    recall: float
    f0_5: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def compute_edit_f_score(
    references: List[str],
    hypotheses: List[str],
    sources: List[str],
    beta: float = 0.5,
) -> CorrectionFScore:
    """
    Computes word-level correction metrics (Precision, Recall, F0.5, F1) across a corpus.
    A true positive occurs when an error in source is corrected to match reference.
    """
    tp = 0
    fp = 0
    fn = 0

    for src_text, ref_text, hyp_text in zip(sources, references, hypotheses):
        src_words = normalize_sinhala(src_text).split()
        ref_words = normalize_sinhala(ref_text).split()
        hyp_words = normalize_sinhala(hyp_text).split()

        # Word-level alignment
        max_len = max(len(src_words), len(ref_words), len(hyp_words))
        for i in range(max_len):
            s = src_words[i] if i < len(src_words) else ""
            r = ref_words[i] if i < len(ref_words) else ""
            h = hyp_words[i] if i < len(hyp_words) else ""

            is_corrupted = (s != r)
            is_modified = (s != h)
            is_correct = (h == r)

            if is_corrupted:
                if is_correct:
                    tp += 1
                else:
                    fn += 1
                    if is_modified:
                        fp += 1  # Modified into wrong word
            else:
                if is_modified:
                    fp += 1  # Falsely modified clean word

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    
    b2 = beta ** 2
    f0_5 = (1 + b2) * (precision * recall) / max((b2 * precision) + recall, 1e-12)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-12)

    return CorrectionFScore(
        precision=precision,
        recall=recall,
        f0_5=f0_5,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


@dataclass
class EvaluationMetrics:
    """Container aggregating all corpus evaluation metrics."""
    cer: float
    wer: float
    aer: float
    f_score: Optional[CorrectionFScore] = None
    exact_match_accuracy: float = 0.0
    num_samples: int = 0
    throughput_samples_per_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "aer": round(self.aer, 4),
            "exact_match_acc": round(self.exact_match_accuracy, 4),
            "precision": round(self.f_score.precision, 4) if self.f_score else None,
            "recall": round(self.f_score.recall, 4) if self.f_score else None,
            "f0_5": round(self.f_score.f0_5, 4) if self.f_score else None,
            "f1": round(self.f_score.f1, 4) if self.f_score else None,
            "num_samples": self.num_samples,
            "throughput_fps": round(self.throughput_samples_per_sec, 2),
        }


def compute_corpus_metrics(
    references: List[str],
    hypotheses: List[str],
    sources: Optional[List[str]] = None,
    throughput_fps: float = 0.0,
) -> EvaluationMetrics:
    """
    Computes aggregate metrics across an entire evaluation corpus.
    """
    total_cer = 0.0
    total_wer = 0.0
    total_aer = 0.0
    exact_matches = 0
    n = len(references)

    for r, h in zip(references, hypotheses):
        total_cer += compute_cer(r, h)
        total_wer += compute_wer(r, h)
        total_aer += compute_aer(r, h)
        if normalize_sinhala(r) == normalize_sinhala(h):
            exact_matches += 1

    f_score = None
    if sources is not None and len(sources) == n:
        f_score = compute_edit_f_score(references, hypotheses, sources)

    return EvaluationMetrics(
        cer=total_cer / max(n, 1),
        wer=total_wer / max(n, 1),
        aer=total_aer / max(n, 1),
        f_score=f_score,
        exact_match_accuracy=exact_matches / max(n, 1),
        num_samples=n,
        throughput_samples_per_sec=throughput_fps,
    )
