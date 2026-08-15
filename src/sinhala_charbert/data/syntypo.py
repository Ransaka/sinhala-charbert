"""
SynTypo-SI: Multi-Stage Probabilistic Typographical, Linguistic, and Code-Mixed Noise Generation Engine.
"""

import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# pyrefly: ignore [missing-import]
from sinlib.utils.preprocessing import normalize_sinhala, process_text
from sinhala_charbert.config.noise_config import NoiseProfile
from sinhala_charbert.data.wijesekara import WijesekaraSpatialKernel

# Orthographic & classical confusion sets
ORTHOGRAPHIC_CONFUSIONS: Dict[str, List[str]] = {
    "න": ["ණ"],
    "ණ": ["න"],
    "ල": ["ළ"],
    "ළ": ["ල"],
    "ච": ["ඡ"],
    "ඡ": ["ච"],
    "ක": ["ඛ"],
    "ඛ": ["ක"],
    "ග": ["ඝ"],
    "ඝ": ["ග"],
    "ට": ["ඨ"],
    "ඨ": ["ට"],
    "ඩ": ["ඪ"],
    "ඪ": ["ඩ"],
    "ත": ["ථ"],
    "ථ": ["ත"],
    "ද": ["ධ"],
    "ධ": ["ද"],
    "ප": ["ඵ"],
    "ඵ": ["ප"],
    "බ": ["භ"],
    "භ": ["බ"],
    "ස": ["ශ", "ෂ"],
    "ශ": ["ස"],
    "ෂ": ["ස"],
    "ඟ": ["ග"],
    "ඳ": ["ද"],
    "ඬ": ["ඩ"],
    "ඹ": ["බ"],
    "ං": ["න්", "ම්"],
}

# Regional / Kandyan morphology transformation rules
KANDYAN_DIALECT_RULES: List[Tuple[str, str]] = [
    (r"(\b[ක-ෆ]?[්-ෞ]*[ක-ෆ])න්නට\b", r"\1න්ට"),
    (r"(\b[ක-ෆ]?[්-ෞ]*[ක-ෆ])න්න\b", r"\1න්ඩ"),
    (r"(\b[ක-ෆ]?[්-ෞ]*[ක-ෆ])ලා\b", r"\1ල"),
]

# Emphatic clitics dictionary
EMPHATIC_CLITIC_MAP: Dict[str, List[str]] = {
    "තමයි": ["තමා", "තමෙයි"],
}

# Postpositional case markers for particle detachment
CASE_MARKERS: Set[str] = {"ට", "ගේ", "ගෙන්", "ද", "නම්", "මයි", "වත්"}

# Code-switching conversational and UI markers
CODE_SWITCH_INSERTS: List[str] = [
    "thanks",
    "siyatha news",
    "Read more",
    "Breaking news",
    "Fake news",
    "Good job",
    "WTF",
    "fyi",
    "pls",
]


class SinhalaTypoSynthesizer:
    """
    SynTypo-SI end-to-end multi-stage noise synthesizer.
    Executes a 4-stage probabilistic DAG to convert clean Sinhala text to realistic noisy text.
    """

    def __init__(self, profile: Optional[NoiseProfile] = None):
        self.cfg = profile or NoiseProfile()
        self.wijesekara_kernel = WijesekaraSpatialKernel(
            spatial_sigma=self.cfg.spatial_sigma,
            distance_threshold=self.cfg.distance_threshold,
            shift_error_prob=self.cfg.shift_error_prob,
        )

    def apply_dialect_transform(self, text: str) -> Tuple[str, List[str]]:
        """Stage 1: Linguistic & Dialectal Transformation Layer."""
        applied_ops = []
        if random.random() > self.cfg.dialect_rate:
            return text, applied_ops

        # Check emphatic clitics
        words = text.split()
        mutated_words = []
        for w in words:
            if w in EMPHATIC_CLITIC_MAP and random.random() < 0.6:
                replacement = random.choice(EMPHATIC_CLITIC_MAP[w])
                mutated_words.append(replacement)
                applied_ops.append(f"EMPHATIC_CLITIC:{w}->{replacement}")
            else:
                mutated_words.append(w)
        text = " ".join(mutated_words)

        # Apply Kandyan regex rules
        for pattern, replacement in KANDYAN_DIALECT_RULES:
            new_text, count = re.subn(pattern, replacement, text)
            if count > 0:
                applied_ops.append(f"DIALECT_RULE:{pattern}")
                text = new_text

        return text, applied_ops

    def apply_ime_and_keystroke_noise(self, text: str) -> Tuple[str, List[str]]:
        """Stage 2: Input Method Engine (IME) Simulation (Wijesekara 2D spatial & shift noise)."""
        applied_ops = []
        output_chars = []
        for char in text:
            rand_val = random.random()
            if rand_val < self.cfg.wijesekara_keystroke_rate:
                noisy_char = self.wijesekara_kernel.sample_noisy_character(char)
                if noisy_char and noisy_char != char:
                    output_chars.append(noisy_char)
                    applied_ops.append(f"WIJESEKARA_KEYSTROKE:{char}->{noisy_char}")
                    continue
            output_chars.append(char)
        return "".join(output_chars), applied_ops

    def apply_unicode_and_orthographic_corruption(self, text: str) -> Tuple[str, List[str]]:
        """Stage 3: Orthographic, Glyph & Unicode Corruption Layer."""
        applied_ops = []

        # 1. ZWJ Sequence Stripping (broken ligatures for Rakaranshaya/Yanshaya/Bandi Akuru)
        if "\u200D" in text and random.random() < self.cfg.zwj_strip_rate:
            text = text.replace("\u200D", "")
            applied_ops.append("ZWJ_STRIP")

        # 2. Unicode long vowel composite decomposition (ේ -> ෙ + ි, ෝ -> ෙ + ා)
        if random.random() < self.cfg.unicode_decompose_rate:
            if "ේ" in text:
                text = text.replace("ේ", "ෙි")
                applied_ops.append("UNICODE_DECOMPOSE:ේ->ෙි")
            if "ෝ" in text:
                text = text.replace("ෝ", "ො")
                applied_ops.append("UNICODE_DECOMPOSE:ෝ->ො")

        # 3. Corrupt Reph / diacritic sequencing
        if "ුරු" in text and random.random() < self.cfg.unicode_decompose_rate:
            text = text.replace("ුරු", "ුරැ")
            applied_ops.append("DIACRITIC_MUTATION:ුරු->ුරැ")

        # 4. Orthographic / Classical Confusion Substitutions
        output_chars = []
        for char in text:
            if random.random() < self.cfg.orthographic_rate and char in ORTHOGRAPHIC_CONFUSIONS:
                sub = random.choice(ORTHOGRAPHIC_CONFUSIONS[char])
                output_chars.append(sub)
                applied_ops.append(f"ORTHO_CONFUSION:{char}->{sub}")
            else:
                output_chars.append(char)

        return "".join(output_chars), applied_ops

    def apply_structural_and_cs_noise(self, text: str) -> Tuple[str, List[str]]:
        """Stage 4: Structural, Punctuation & Code-Switch Injection Layer."""
        applied_ops = []
        tokens = text.split(" ")
        mutated_tokens: List[str] = []

        for token in tokens:
            # Punctuation mutation 1: Skipping space after full stop
            if "." in token and not token.endswith(".") and random.random() < self.cfg.space_mutation_rate:
                token = token.replace(". ", ".")
                applied_ops.append("PUNCT_NO_SPACE_AFTER_FULLSTOP")

            # Punctuation mutation 2: Particle detachment before postpositions
            for marker in CASE_MARKERS:
                if token.endswith(marker) and len(token) > len(marker):
                    if random.random() < self.cfg.space_mutation_rate:
                        stem = token[:-len(marker)]
                        token = f"{stem} {marker}"
                        applied_ops.append(f"PARTICLE_DETACHMENT:{marker}")
                        break

            # Whitespace perturbation: Accidental token fusion (drop space between adjacent tokens)
            if mutated_tokens and random.random() < (self.cfg.space_mutation_rate * 0.5):
                prev_token = mutated_tokens.pop()
                token = f"{prev_token}{token}"
                applied_ops.append("TOKEN_FUSION")

            mutated_tokens.append(token)

        # Full stop as space delimiter (replace ' ' with '.')
        if random.random() < (self.cfg.space_mutation_rate * 0.5) and len(mutated_tokens) > 2:
            idx = random.randint(0, len(mutated_tokens) - 2)
            mutated_tokens[idx] = f"{mutated_tokens[idx]}."
            applied_ops.append("FULLSTOP_AS_SPACE")

        # Code-switching insertion
        if random.random() < self.cfg.code_switch_rate:
            cs_phrase = random.choice(CODE_SWITCH_INSERTS)
            pos = random.choice([0, len(mutated_tokens)])
            mutated_tokens.insert(pos, cs_phrase)
            applied_ops.append(f"CODE_SWITCH:{cs_phrase}")

        return " ".join(mutated_tokens), applied_ops

    def generate_pair(self, clean_text: str) -> Dict[str, Any]:
        """
        Executes end-to-end DAG pipeline:
        Clean Text -> Dialect (Stage 1) -> IME (Stage 2) -> Ortho/Unicode (Stage 3) -> Structural (Stage 4) -> Noisy Text
        """
        clean_text = normalize_sinhala(clean_text)
        all_ops = []

        noisy, ops1 = self.apply_dialect_transform(clean_text)
        all_ops.extend(ops1)

        noisy, ops2 = self.apply_ime_and_keystroke_noise(noisy)
        all_ops.extend(ops2)

        noisy, ops3 = self.apply_unicode_and_orthographic_corruption(noisy)
        all_ops.extend(ops3)

        noisy, ops4 = self.apply_structural_and_cs_noise(noisy)
        all_ops.extend(ops4)

        return {
            "source_noisy": noisy,
            "target_clean": clean_text,
            "error_ops": all_ops,
            "has_error": len(all_ops) > 0,
        }
