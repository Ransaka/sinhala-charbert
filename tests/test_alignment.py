"""
Unit tests for Sinhala-CharBERT Tokenization, Sequence Alignment Engine, and Data Collator.
"""

import unittest
import torch
from transformers import AutoTokenizer

from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.data.alignment import SequenceAlignmentEngine, AlignedSequence
from sinhala_charbert.data.collator import DualChannelDataCollator


class TestAlignmentAndCollator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subword_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        cls.char_tokenizer = SinhalaCharTokenizer()
        cls.char_tokenizer.train_on_corpus([
            "ආයුබෝවන්",
            "මම ගෙදර යනවා",
            "සිංහල භාෂාව ආරක්ෂා කර ගැනීම අපේ යුතුකමකි.",
        ])
        cls.alignment_engine = SequenceAlignmentEngine(
            subword_tokenizer=cls.subword_tokenizer,
            char_tokenizer=cls.char_tokenizer,
        )

    def test_char_tokenizer_encode_decode(self):
        text = "ආයුබෝවන්"
        encoded = self.char_tokenizer.encode(text, add_special_tokens=True)
        self.assertEqual(encoded[0], self.char_tokenizer.bos_token_id)
        self.assertEqual(encoded[-1], self.char_tokenizer.eos_token_id)
        decoded = self.char_tokenizer.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, text)

    def test_sequence_alignment(self):
        text = "මම ගෙදර යනවා"
        aligned = self.alignment_engine.align(text)
        
        m_len = len(aligned.input_ids)
        n_len = len(aligned.char_input_ids)

        self.assertGreater(n_len, 0)
        self.assertGreater(m_len, 0)
        self.assertEqual(len(aligned.start_char_idx), m_len)
        self.assertEqual(len(aligned.end_char_idx), m_len)

        # Check CLS and SEP alignment
        self.assertEqual(aligned.start_char_idx[0], 0)
        self.assertEqual(aligned.end_char_idx[0], 0)
        self.assertEqual(aligned.start_char_idx[-1], n_len - 1)
        self.assertEqual(aligned.end_char_idx[-1], n_len - 1)

        # Check valid range and order for all indices
        for i in range(m_len):
            self.assertGreaterEqual(aligned.start_char_idx[i], 0)
            self.assertLess(aligned.start_char_idx[i], n_len)
            self.assertGreaterEqual(aligned.end_char_idx[i], 0)
            self.assertLess(aligned.end_char_idx[i], n_len)
            self.assertLessEqual(aligned.start_char_idx[i], aligned.end_char_idx[i])

    def test_dual_channel_collator(self):
        samples = [
            self.alignment_engine.align("මම ගෙදර යනවා"),
            self.alignment_engine.align("සිංහල භාෂාව ආරක්ෂා කරමු"),
        ]

        collator = DualChannelDataCollator(
            subword_pad_token_id=0,
            subword_mask_token_id=103,
            subword_vocab_size=self.subword_tokenizer.vocab_size,
            mlm_probability=0.15,
        )

        batch = collator(samples)

        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("char_input_ids", batch)
        self.assertIn("char_attention_mask", batch)
        self.assertIn("start_char_idx", batch)
        self.assertIn("end_char_idx", batch)
        self.assertIn("mlm_labels", batch)

        # Check tensor batch dimensions
        b_size = len(samples)
        m_max = max(len(s.input_ids) for s in samples)
        n_max = max(len(s.char_input_ids) for s in samples)

        self.assertEqual(batch["input_ids"].shape, (b_size, m_max))
        self.assertEqual(batch["attention_mask"].shape, (b_size, m_max))
        self.assertEqual(batch["char_input_ids"].shape, (b_size, n_max))
        self.assertEqual(batch["char_attention_mask"].shape, (b_size, n_max))
        self.assertEqual(batch["start_char_idx"].shape, (b_size, m_max))
        self.assertEqual(batch["end_char_idx"].shape, (b_size, m_max))
        self.assertEqual(batch["mlm_labels"].shape, (b_size, m_max))


if __name__ == "__main__":
    unittest.main()
