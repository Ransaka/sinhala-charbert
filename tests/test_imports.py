"""
Sanity tests for Sinhala-CharBERT configuration and sinlib importability.
"""

import sys
import unittest


class TestImportsAndConfigs(unittest.TestCase):
    def test_import_configs(self):
        from sinhala_charbert.config import (
            SinhalaCharBERTConfig,
            NoiseProfile,
            CurriculumPhaseConfig,
            TrainingConfig,
        )
        
        cfg = SinhalaCharBERTConfig()
        self.assertEqual(cfg.hidden_size, 786)
        self.assertEqual(cfg.num_hidden_layers, 6)
        self.assertEqual(cfg.hi_kernel_sizes, [1, 3, 5])
        
        noise_cfg = NoiseProfile()
        self.assertEqual(noise_cfg.dialect_rate, 0.20)
        self.assertEqual(noise_cfg.spatial_sigma, 0.75)
        
        train_cfg = TrainingConfig()
        self.assertEqual(train_cfg.learning_rate, 5e-5)

    def test_sinlib_availability(self):
        """Verify that sinlib is importable and basic tokenization/normalization functions work."""
        from sinlib import Tokenizer
        from sinlib.utils.preprocessing import normalize_sinhala, process_text

        text = "ආයුබෝවන්"
        normalized = normalize_sinhala(text)
        self.assertEqual(normalized, text)
        
        units = process_text(text)
        self.assertEqual(units, ["ආ", "යු", "බෝ", "ව", "න්"])


if __name__ == "__main__":
    unittest.main()
