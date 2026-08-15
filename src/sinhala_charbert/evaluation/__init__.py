"""
Evaluation metrics, baselines, and benchmarking suite for Sinhala Typo Correction.
"""

from sinhala_charbert.evaluation.metrics import (
    compute_cer,
    compute_wer,
    compute_aer,
    compute_edit_f_score,
    compute_corpus_metrics,
    EvaluationMetrics,
    CorrectionFScore,
    levenshtein_distance,
)
from sinhala_charbert.evaluation.baselines import (
    IdentityBaseline,
    RuleBasedSinhalaCorrector,
    StandardBERTMLMCorrector,
)
from sinhala_charbert.evaluation.evaluator import TypoCorrectionEvaluator

__all__ = [
    "compute_cer",
    "compute_wer",
    "compute_aer",
    "compute_edit_f_score",
    "compute_corpus_metrics",
    "EvaluationMetrics",
    "CorrectionFScore",
    "levenshtein_distance",
    "IdentityBaseline",
    "RuleBasedSinhalaCorrector",
    "StandardBERTMLMCorrector",
    "TypoCorrectionEvaluator",
]
