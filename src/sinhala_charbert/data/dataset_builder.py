"""
Dataset builder and parallel processor for generating synthetic typo correction datasets using SynTypo-SI.
"""

from typing import Any, Dict, List, Optional
# pyrefly: ignore [missing-import]
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm

from sinhala_charbert.config.noise_config import NoiseProfile
from sinhala_charbert.data.syntypo import SinhalaTypoSynthesizer


def synthesize_sample(
    example: Dict[str, Any],
    synthesizer: SinhalaTypoSynthesizer,
    text_column: str = "text",
) -> Dict[str, Any]:
    """Applies SynTypo-SI to a single sample."""
    clean_text = example.get(text_column, "")
    if not clean_text or not isinstance(clean_text, str):
        return {"source_noisy": "", "target_clean": "", "error_ops": [], "has_error": False}

    res = synthesizer.generate_pair(clean_text)
    return res


def build_synthetic_dataset(
    dataset_name_or_path: str = "Ransaka/sinhala-450M-sample",
    split: str = "train",
    num_samples: Optional[int] = None,
    noise_profile: Optional[NoiseProfile] = None,
    text_column: str = "text",
    num_proc: int = 1,
) -> Dataset:
    """
    Loads a clean text dataset and produces a parallel (source_noisy, target_clean, metadata) dataset.

    Parameters
    ----------
    dataset_name_or_path : str
        HuggingFace dataset repository name or local path.
    split : str
        Dataset split to load (e.g., 'train').
    num_samples : int, optional
        Number of samples to process from the dataset.
    noise_profile : NoiseProfile, optional
        Custom noise profile configuration.
    text_column : str
        Column name containing raw clean text.
    num_proc : int
        Number of parallel worker processes.
    """
    raw_dataset = load_dataset(dataset_name_or_path, split=split)
    if num_samples is not None and num_samples > 0:
        raw_dataset = raw_dataset.select(range(min(num_samples, len(raw_dataset))))

    synthesizer = SinhalaTypoSynthesizer(profile=noise_profile)

    def _map_fn(example):
        return synthesize_sample(example, synthesizer=synthesizer, text_column=text_column)

    processed_dataset = raw_dataset.map(
        _map_fn,
        num_proc=num_proc if num_proc > 1 else None,
        desc="Synthesizing SynTypo-SI Typo Dataset",
    )
    return processed_dataset
