"""
Unit tests for Phase 7: Comprehensive Evaluation & Benchmarking Suite.
"""

import pytest
import torch

from sinhala_charbert.evaluation.metrics import (
    compute_aer,
    compute_cer,
    compute_corpus_metrics,
    compute_edit_f_score,
    compute_wer,
    levenshtein_distance,
)
from sinhala_charbert.evaluation.baselines import (
    IdentityBaseline,
    RuleBasedSinhalaCorrector,
)
from sinhala_charbert.evaluation.evaluator import TypoCorrectionEvaluator
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary


class TestEvaluationSuite:

    def test_levenshtein_distance(self):
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("", "test") == 4
        assert levenshtein_distance(["ආ", "යු"], ["ආ", "යු"]) == 0
        assert levenshtein_distance(["ආ", "යු"], ["ආ", "බෝ"]) == 1

    def test_cer_wer_aer_metrics(self):
        ref = "මම ගෙදර යනවා"
        hyp_exact = "මම ගෙදර යනවා"
        hyp_err = "මම ගෙදර යන්ඩ"

        # Exact match
        assert compute_cer(ref, hyp_exact) == 0.0
        assert compute_wer(ref, hyp_exact) == 0.0
        assert compute_aer(ref, hyp_exact) == 0.0

        # Mismatch
        cer = compute_cer(ref, hyp_err)
        wer = compute_wer(ref, hyp_err)
        aer = compute_aer(ref, hyp_err)

        assert cer > 0.0
        assert wer > 0.0
        assert aer > 0.0

    def test_edit_f_score_calculation(self):
        sources = ["මම ගෙදර යන්ඩ", "ඔහු කරුනාවන්තයි"]
        references = ["මම ගෙදර යන්න", "ඔහු කරුණාවන්තයි"]
        hypotheses = ["මම ගෙදර යන්න", "ඔහු කරුණාවන්තයි"]

        f_res = compute_edit_f_score(references, hypotheses, sources, beta=0.5)
        assert f_res.precision == 1.0
        assert f_res.recall == 1.0
        assert f_res.f0_5 == 1.0
        assert f_res.f1 == 1.0
        assert f_res.true_positives == 2
        assert f_res.false_positives == 0

    def test_rule_based_corrector_baseline(self):
        nlm_dict = SinhalaNLMDictionary()
        nlm_dict.build_from_corpus(["කරුණාවන්ත", "යන්න", "ක්‍රමය"], max_words=50)

        corrector = RuleBasedSinhalaCorrector(dictionary=nlm_dict)

        # Murdhaja / Dantaja correction
        corr_murdhaja = corrector.correct("කරුනාවන්ත")
        assert "කරුණාවන්ත" in corr_murdhaja

        # ZWJ ligature correction
        corr_ligature = corrector.correct("ක්රමය")
        assert "ක්‍රමය" in corr_ligature

    def test_evaluator_and_markdown_table(self):
        test_samples = [
            {
                "source_noisy": "මම ගෙදර යන්ඩ",
                "target_clean": "මම ගෙදර යන්න",
                "error_category": "dialectal",
            },
            {
                "source_noisy": "කරුනාවන්ත මිනිස්සු",
                "target_clean": "කරුණාවන්ත මිනිස්සු",
                "error_category": "orthographic",
            },
        ]

        evaluator = TypoCorrectionEvaluator(test_samples)
        models = {
            "Identity Baseline": IdentityBaseline().correct_batch,
        }

        results = evaluator.run_benchmark(models)
        assert "Identity Baseline" in results
        assert "overall" in results["Identity Baseline"]
        assert "categories" in results["Identity Baseline"]

        table_str = evaluator.format_markdown_table(results)
        assert isinstance(table_str, str)
        assert "Identity Baseline" in table_str
        assert "CER" in table_str
