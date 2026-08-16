"""
Unit tests for the SynTypo-SI multi-stage noise generation pipeline.
"""

import unittest
from sinhala_charbert.config.noise_config import NoiseProfile
from sinhala_charbert.data.wijesekara import WijesekaraSpatialKernel, CHAR_TO_WIJESEKARA
from sinhala_charbert.data.syntypo import SinhalaTypoSynthesizer


class TestSynTypoSI(unittest.TestCase):
    def setUp(self):
        self.profile = NoiseProfile(
            dialect_rate=1.0,
            wijesekara_keystroke_rate=0.5,
            orthographic_rate=0.5,
            unicode_decompose_rate=1.0,
            zwj_strip_rate=1.0,
            space_mutation_rate=1.0,
            code_switch_rate=1.0,
            spatial_sigma=0.75,
        )
        self.synthesizer = SinhalaTypoSynthesizer(profile=self.profile)

    def test_wijesekara_kernel(self):
        kernel = WijesekaraSpatialKernel(spatial_sigma=0.75)
        # Test character lookup
        self.assertIn("ක", CHAR_TO_WIJESEKARA)
        self.assertIn("න", CHAR_TO_WIJESEKARA)
        
        # Test sampling a neighbor
        sampled = kernel.sample_noisy_character("ක")
        self.assertTrue(sampled is not None and isinstance(sampled, str))

    def test_dialect_transformation(self):
        text = "ඔබ එහි යන්න සහ එය දෙන්නට අවශ්‍යයි තමයි"
        noisy, ops = self.synthesizer.apply_dialect_transform(text)
        self.assertTrue(any("DIALECT_RULE" in op or "EMPHATIC_CLITIC" in op for op in ops))
        self.assertTrue("යන්ඩ" in noisy or "දෙන්ට" in noisy or "තමා" in noisy or "තමෙයි" in noisy)

    def test_unicode_and_zwj_corruption(self):
        # Text with ZWJ ligature and composite long vowel
        text = "ක්‍රිකට් ක්‍රීඩකයා සහ ගුරුවරුන්ගේ ප්‍රශ්නයක්"
        noisy, ops = self.synthesizer.apply_unicode_and_orthographic_corruption(text)
        self.assertTrue(len(ops) > 0)
        # Verify ZWJ was stripped or vowel was decomposed
        self.assertTrue("\u200D" not in noisy or "ෙි" in noisy or "ුරැ" in noisy or any("ORTHO_CONFUSION" in op for op in ops))

    def test_structural_and_code_switch_noise(self):
        text = "මම ගෙදර ගියා. ඔහු පාසලට ගියා"
        noisy, ops = self.synthesizer.apply_structural_and_cs_noise(text)
        self.assertTrue(len(ops) > 0)
        # Should have particle detachment or code switch or full stop mutation
        self.assertTrue(any("CODE_SWITCH" in op or "PARTICLE_DETACHMENT" in op or "PUNCT" in op for op in ops))

    def test_end_to_end_pair_generation(self):
        clean_text = "සිංහල භාෂාව ආරක්ෂා කර ගැනීම අපේ යුතුකමකි."
        res = self.synthesizer.generate_pair(clean_text)
        self.assertIn("source_noisy", res)
        self.assertIn("target_clean", res)
        self.assertIn("word_alignments", res)
        self.assertEqual(res["target_clean"], clean_text)
        self.assertTrue(isinstance(res["source_noisy"], str))
        self.assertTrue(len(res["source_noisy"]) > 0)
        self.assertTrue(len(res["word_alignments"]) > 0)
        # Verify word alignment structure
        first_align = res["word_alignments"][0]
        self.assertIn("clean_word", first_align)
        self.assertIn("noisy_word", first_align)
        self.assertIn("noisy_span", first_align)



if __name__ == "__main__":
    unittest.main()
