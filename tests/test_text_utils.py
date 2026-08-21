"""
Unit tests for Sinhala-aware text chunking and sentence-level segmentation utilities.
"""

import unittest
from sinhala_charbert.data.text_utils import split_sentences, split_sentences_with_spans


class TestTextUtils(unittest.TestCase):
    """Test suite for sentence splitting and text chunking utilities."""

    def test_single_sentence(self):
        text = "ශ්‍රී ලංකාව සුන්දර දූපතකි."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0], "ශ්‍රී ලංකාව සුන්දර දූපතකි.")

    def test_multiple_sentences_standard_punctuation(self):
        text = "මම ගෙදර ගියා. ඔහු ආවා! ඇයි ඔයා එහෙම කිව්වේ? අපි හෙට යමු."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0], "මම ගෙදර ගියා.")
        self.assertEqual(sentences[1], "ඔහු ආවා!")
        self.assertEqual(sentences[2], "ඇයි ඔයා එහෙම කිව්වේ?")
        self.assertEqual(sentences[3], "අපි හෙට යමු.")

    def test_split_with_spans_exact_reconstruction(self):
        text = "  පළමු වාක්‍යය මෙන්න.   දෙවන වාක්‍යය මෙයයි!\n\nතෙවන වාක්‍යය මෙසේය?  "
        spans = split_sentences_with_spans(text)
        self.assertEqual(len(spans), 3)

        # Verify that extracting spans and reconstructing with original separators yields exact text
        reconstructed = []
        last_pos = 0
        for sent_str, s_start, s_end in spans:
            reconstructed.append(text[last_pos:s_start])
            reconstructed.append(sent_str)
            last_pos = s_end
        reconstructed.append(text[last_pos:])

        self.assertEqual("".join(reconstructed), text)

    def test_long_sentence_sub_splitting(self):
        # Create a sentence with 60 words (exceeding default 30 word limit for test)
        clause1 = "මෙය දීර්ඝ වාක්‍යයක පළමු කොටස වන අතර මෙහි බොහෝ වචන අඩංගු වේ"
        clause2 = "දෙවන කොටස ලෙස තවත් කරුණු කිහිපයක් මෙහි විස්තර කර ඇත"
        clause3 = "තෙවන කොටස ලෙස අවසාන නිගමනය මෙසේ ඉදිරිපත් කරමු."
        long_sentence = f"{clause1}, {clause2}, {clause3}"

        chunks = split_sentences(long_sentence, max_words_per_chunk=15)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 20)

    def test_empty_and_whitespace_input(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   \n\t  "), [])
        self.assertEqual(split_sentences_with_spans(""), [])


if __name__ == "__main__":
    unittest.main()
