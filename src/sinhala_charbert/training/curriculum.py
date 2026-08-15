"""
Noise Curriculum Scheduler for phased SynTypo-SI training.
Progressively increases noise complexity across pre-training steps.
"""

from typing import Dict, List, Optional
from sinhala_charbert.config.noise_config import NoiseProfile, CurriculumPhaseConfig


class NoiseCurriculumScheduler:
    """
    3-Phase Curriculum Scheduler for SynTypo-SI.
    Phase 1 (Warmup): Simple orthographic confusion substitutions.
    Phase 2 (Keystrokes & Ligatures): Wijesekara physical drift, shift inversion, ZWJ stripping, and punctuation mutations.
    Phase 3 (Robustness & Code-Switching): Full DAG with regional dialects, particle detachment, and conversational code-mixing.
    """

    def __init__(
        self,
        total_steps: int = 100000,
        phase1_ratio: float = 0.20,
        phase2_ratio: float = 0.40,
    ):
        self.total_steps = total_steps
        self.phase1_end_step = int(total_steps * phase1_ratio)
        self.phase2_end_step = int(total_steps * (phase1_ratio + phase2_ratio))

        # Phase 1 Profile: Warmup (Orthographic only)
        self.phase1_profile = NoiseProfile(
            dialect_rate=0.0,
            wijesekara_keystroke_rate=0.0,
            orthographic_rate=0.15,
            unicode_decompose_rate=0.0,
            zwj_strip_rate=0.0,
            space_mutation_rate=0.0,
            code_switch_rate=0.0,
            spatial_sigma=0.75,
            shift_error_prob=0.0,
        )

        # Phase 2 Profile: Keystroke + ZWJ + Punctuation
        self.phase2_profile = NoiseProfile(
            dialect_rate=0.05,
            wijesekara_keystroke_rate=0.08,
            orthographic_rate=0.12,
            unicode_decompose_rate=0.05,
            zwj_strip_rate=0.10,
            space_mutation_rate=0.05,
            code_switch_rate=0.0,
            spatial_sigma=0.75,
            shift_error_prob=0.15,
        )

        # Phase 3 Profile: Full Robustness & Code-Switching
        self.phase3_profile = NoiseProfile(
            dialect_rate=0.20,
            wijesekara_keystroke_rate=0.08,
            orthographic_rate=0.12,
            unicode_decompose_rate=0.05,
            zwj_strip_rate=0.10,
            space_mutation_rate=0.05,
            code_switch_rate=0.05,
            spatial_sigma=0.75,
            shift_error_prob=0.15,
        )

    def get_phase_name(self, step: int) -> str:
        """Returns the active curriculum phase name for the given training step."""
        if step < self.phase1_end_step:
            return "Phase 1: Warmup (Orthographic Substitutions)"
        elif step < self.phase2_end_step:
            return "Phase 2: Complex Keystroke, ZWJ & Punctuation"
        else:
            return "Phase 3: Full Robustness, Dialects & Code-Switching"

    def get_profile(self, step: int) -> NoiseProfile:
        """Returns the active NoiseProfile dataclass for the given training step."""
        if step < self.phase1_end_step:
            return self.phase1_profile
        elif step < self.phase2_end_step:
            return self.phase2_profile
        else:
            return self.phase3_profile
