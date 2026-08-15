"""
Typo Correction Evaluation Engine and Multi-Model Benchmarking Runner.
"""

from collections import defaultdict
import time
from typing import Any, Callable, Dict, List, Optional, Union
from tabulate import tabulate

from sinlib.utils.preprocessing import normalize_sinhala
from sinhala_charbert.evaluation.metrics import (
    EvaluationMetrics,
    compute_aer,
    compute_cer,
    compute_corpus_metrics,
    compute_wer,
)


class TypoCorrectionEvaluator:
    """
    Evaluator for comparing Sinhala Typo Correction models and baselines.
    Computes overall error rates, fine-grained category breakdowns, and throughput.
    """

    def __init__(
        self,
        test_samples: List[Dict[str, Any]],
    ):
        """
        Args:
            test_samples: List of dicts containing:
                - 'source_noisy': Corrupted Sinhala sentence.
                - 'target_clean': Gold clean reference sentence.
                - 'error_category': Optional category string (e.g. 'orthographic', 'ligature', 'dialect').
        """
        self.test_samples = test_samples
        self.sources = [s["source_noisy"] for s in test_samples]
        self.references = [s["target_clean"] for s in test_samples]
        self.categories = [s.get("error_category", "general") for s in test_samples]

    def evaluate_model(
        self,
        model_name: str,
        predict_fn: Callable[[List[str]], List[str]],
    ) -> Dict[str, Any]:
        """
        Evaluates a single model or pipeline.
        Args:
            model_name: Identifier name for the model (e.g., 'Sinhala-CharBERT (Mode A)').
            predict_fn: Function mapping List[str] (sources) -> List[str] (hypotheses).
        """
        start_time = time.time()
        hypotheses = predict_fn(self.sources)
        elapsed_time = time.time() - start_time
        fps = len(self.sources) / max(elapsed_time, 1e-6)

        # 1. Overall Corpus Metrics
        overall_metrics = compute_corpus_metrics(
            references=self.references,
            hypotheses=hypotheses,
            sources=self.sources,
            throughput_fps=fps,
        )

        # 2. Category Breakdown
        category_indices = defaultdict(list)
        for idx, cat in enumerate(self.categories):
            category_indices[cat].append(idx)

        category_aer = {}
        for cat, indices in category_indices.items():
            cat_refs = [self.references[i] for i in indices]
            cat_hyps = [hypotheses[i] for i in indices]
            cat_metrics = compute_corpus_metrics(cat_refs, cat_hyps)
            category_aer[cat] = {
                "aer": round(cat_metrics.aer, 4),
                "cer": round(cat_metrics.cer, 4),
                "exact_acc": round(cat_metrics.exact_match_accuracy, 4),
                "count": len(indices),
            }

        return {
            "model_name": model_name,
            "overall": overall_metrics.to_dict(),
            "categories": category_aer,
            "latency_ms_per_sample": round((elapsed_time / max(len(self.sources), 1)) * 1000, 2),
            "hypotheses": hypotheses,
        }

    def run_benchmark(
        self,
        models_dict: Dict[str, Callable[[List[str]], List[str]]],
    ) -> Dict[str, Any]:
        """
        Runs full comparative benchmark across multiple models.
        """
        results = {}
        for model_name, predict_fn in models_dict.items():
            print(f"Evaluating '{model_name}' on {len(self.sources)} samples...")
            results[model_name] = self.evaluate_model(model_name, predict_fn)
        return results

    def format_markdown_table(self, benchmark_results: Dict[str, Any]) -> str:
        """
        Formats benchmark results into a clean markdown table.
        """
        headers = [
            "Model / System",
            "CER (↓)",
            "AER (↓)",
            "WER (↓)",
            "F0.5 (↑)",
            "F1 (↑)",
            "Exact Acc (↑)",
            "Speed (FPS)",
        ]

        rows = []
        for model_name, res in benchmark_results.items():
            ov = res["overall"]
            f0_5_str = f"{ov['f0_5']:.4f}" if ov.get("f0_5") is not None else "-"
            f1_str = f"{ov['f1']:.4f}" if ov.get("f1") is not None else "-"
            rows.append([
                model_name,
                f"{ov['cer']:.4f}",
                f"{ov['aer']:.4f}",
                f"{ov['wer']:.4f}",
                f0_5_str,
                f1_str,
                f"{ov['exact_match_acc']:.2%}",
                f"{ov['throughput_fps']:.1f}",
            ])

        return tabulate(rows, headers=headers, tablefmt="github")
