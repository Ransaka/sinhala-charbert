"""
Data processing, tokenization, sequence alignment, and SynTypo-SI noise generation.
"""

from .wijesekara import KeyLocation, WijesekaraSpatialKernel, QWERTY_GRID, CHAR_TO_WIJESEKARA
from .syntypo import SinhalaTypoSynthesizer, ORTHOGRAPHIC_CONFUSIONS, KANDYAN_DIALECT_RULES
from .dataset_builder import build_synthetic_dataset, synthesize_sample
from .char_tokenizer import SinhalaCharTokenizer
from .alignment import AlignedSequence, SequenceAlignmentEngine
from .collator import DualChannelDataCollator

__all__ = [
    "KeyLocation",
    "WijesekaraSpatialKernel",
    "QWERTY_GRID",
    "CHAR_TO_WIJESEKARA",
    "SinhalaTypoSynthesizer",
    "ORTHOGRAPHIC_CONFUSIONS",
    "KANDYAN_DIALECT_RULES",
    "build_synthetic_dataset",
    "synthesize_sample",
    "SinhalaCharTokenizer",
    "AlignedSequence",
    "SequenceAlignmentEngine",
    "DualChannelDataCollator",
]
