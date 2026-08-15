"""
Wijesekara Keyboard Layout (SLS 1134 standard) physical coordinate mapping and spatial kernel.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class KeyLocation:
    x: float
    y: float
    unshifted: str
    shifted: str


# Physical 2D layout mapping based on SLS 1134 Wijesekara standard on QWERTY
QWERTY_GRID: Dict[str, KeyLocation] = {
    "q": KeyLocation(1.0, 0.0, "ු", "ූ"),
    "w": KeyLocation(1.0, 1.0, "අ", "උ"),
    "e": KeyLocation(1.0, 2.0, "ැ", "ෑ"),
    "r": KeyLocation(1.0, 3.0, "ර", "්ර"),
    "t": KeyLocation(1.0, 4.0, "ඔ", "ඖ"),
    "y": KeyLocation(1.0, 5.0, "හ", "ශ"),
    "u": KeyLocation(1.0, 6.0, "ම", "ඹ"),
    "i": KeyLocation(1.0, 7.0, "ස", "ෂ"),
    "o": KeyLocation(1.0, 8.0, "ද", "ධ"),
    "p": KeyLocation(1.0, 9.0, "ච", "ඡ"),
    "a": KeyLocation(2.0, 0.25, "්", "ෟ"),
    "s": KeyLocation(2.0, 1.25, "ි", "ී"),
    "d": KeyLocation(2.0, 2.25, "ා", "ෘ"),
    "f": KeyLocation(2.0, 3.25, "ෙ", "ේ"),
    "g": KeyLocation(2.0, 4.25, "ට", "ඨ"),
    "h": KeyLocation(2.0, 5.25, "ය", "්ය"),
    "j": KeyLocation(2.0, 6.25, "ව", "ළු"),
    "k": KeyLocation(2.0, 7.25, "න", "ණ"),
    "l": KeyLocation(2.0, 8.25, "ක", "ඛ"),
    ";": KeyLocation(2.0, 9.25, "ත", "ථ"),
    "z": KeyLocation(3.0, 0.75, "raw", "raw"),
    "x": KeyLocation(3.0, 1.75, "ං", "ඃ"),
    "c": KeyLocation(3.0, 2.75, "ජ", "ඣ"),
    "v": KeyLocation(3.0, 3.75, "ඩ", "ඪ"),
    "b": KeyLocation(3.0, 4.75, "ඉ", "ඊ"),
    "n": KeyLocation(3.0, 5.75, "බ", "භ"),
    "m": KeyLocation(3.0, 6.75, "ප", "ඵ"),
    ",": KeyLocation(3.0, 7.75, "ල", "ළ"),
}

# Reverse lookup table for characters on Wijesekara
CHAR_TO_WIJESEKARA: Dict[str, Tuple[str, bool]] = {}
for qkey, kloc in QWERTY_GRID.items():
    if kloc.unshifted and kloc.unshifted != "raw":
        CHAR_TO_WIJESEKARA[kloc.unshifted] = (qkey, False)
    if kloc.shifted and kloc.shifted != "raw":
        CHAR_TO_WIJESEKARA[kloc.shifted] = (qkey, True)


class WijesekaraSpatialKernel:
    """Calculates Euclidean spatial drift and shift modifier drops on Wijesekara layout."""

    def __init__(self, spatial_sigma: float = 0.75, distance_threshold: float = 1.8, shift_error_prob: float = 0.15):
        self.spatial_sigma = spatial_sigma
        self.distance_threshold = distance_threshold
        self.shift_error_prob = shift_error_prob
        self._distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> Dict[str, List[Tuple[str, float]]]:
        """Precomputes distances and Gaussian weights between adjacent keys."""
        dist_map: Dict[str, List[Tuple[str, float]]] = {}
        for src_key, src_loc in QWERTY_GRID.items():
            candidates = []
            for tgt_key, tgt_loc in QWERTY_GRID.items():
                if src_key == tgt_key:
                    continue
                dist = math.sqrt((src_loc.x - tgt_loc.x) ** 2 + (src_loc.y - tgt_loc.y) ** 2)
                if dist <= self.distance_threshold:
                    weight = math.exp(-(dist ** 2) / (2 * (self.spatial_sigma ** 2)))
                    candidates.append((tgt_key, weight))
            dist_map[src_key] = candidates
        return dist_map

    def sample_noisy_character(self, char: str) -> Optional[str]:
        """Returns a spatially corrupted or shift-inverted character on Wijesekara layout."""
        if char not in CHAR_TO_WIJESEKARA:
            return None

        qkey, is_shifted = CHAR_TO_WIJESEKARA[char]
        src_loc = QWERTY_GRID[qkey]

        # Shift modifier inversion (e.g. unshifted hit when shifted intended)
        if random.random() < self.shift_error_prob:
            inverted = src_loc.unshifted if is_shifted else src_loc.shifted
            if inverted and inverted != "raw":
                return inverted

        # Spatial neighbor selection based on Gaussian probability
        candidates = self._distance_matrix.get(qkey, [])
        if not candidates:
            return None

        keys, weights = zip(*candidates)
        chosen_key = random.choices(keys, weights=weights, k=1)[0]
        chosen_loc = QWERTY_GRID[chosen_key]

        result = chosen_loc.shifted if is_shifted else chosen_loc.unshifted
        return result if result != "raw" else None
