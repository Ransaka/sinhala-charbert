"""
Noise synthesis configuration dataclasses for SynTypo-SI.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class NoiseProfile:
    """Configuration parameter profile for the SynTypo-SI noise pipeline."""
    dialect_rate: float = 0.20
    wijesekara_keystroke_rate: float = 0.08
    orthographic_rate: float = 0.12
    unicode_decompose_rate: float = 0.05
    zwj_strip_rate: float = 0.10
    space_mutation_rate: float = 0.05
    code_switch_rate: float = 0.03
    spatial_sigma: float = 0.75
    distance_threshold: float = 1.8
    shift_error_prob: float = 0.15


@dataclass
class CurriculumPhaseConfig:
    """Configuration for a specific stage in the noise training curriculum."""
    name: str
    active_stages: List[str]
    noise_multiplier: float = 1.0
