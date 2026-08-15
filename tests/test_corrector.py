"""
Unit tests for Phase 6: Downstream Typo Correction Execution (Mode A & Mode B).
"""

import pytest
import torch
from transformers import AutoTokenizer

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.data.alignment import SequenceAlignmentEngine
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining
from sinhala_charbert.models.word_corrector import BoundedWordCorrector, WordCorrectionCandidate
from sinhala_charbert.models.seq2seq_decoder import SinhalaCharBERTSeq2SeqModel, Seq2SeqCorrectionOutput
from sinhala_charbert.models.pipeline import SinhalaCharBERTCorrector, CorrectionResult, EditOp
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary


@pytest.fixture(scope="module")
def shared_pipeline_resources():
    corpus = [
        "මම ගෙදර යනවා සහ පොතක් කියවනවා.",
        "ශ්‍රී ලංකාව සුන්දර රටකි.",
        "ඔහු පාසල් ගියේය.",
    ]
    subword_tok = AutoTokenizer.from_pretrained("Ransaka/sinhala-bert-medium-v2")
    char_tok = SinhalaCharTokenizer()
    char_tok.train_on_corpus(corpus)

    nlm_dict = SinhalaNLMDictionary()
    nlm_dict.build_from_corpus(corpus, max_words=100)

    alignment_engine = SequenceAlignmentEngine(
        subword_tokenizer=subword_tok,
        char_tokenizer=char_tok,
        max_subword_length=64,
        max_char_length=128,
    )

    config = SinhalaCharBERTConfig(
        vocab_size=subword_tok.vocab_size,
        char_vocab_size=char_tok.vocab_size,
        nlm_vocab_size=nlm_dict.vocab_size,
        hidden_size=786,
        char_gru_hidden_size=393,
        num_hidden_layers=2,
        num_attention_heads=6,
        intermediate_size=512,
        max_position_embeddings=64,
    )

    return {
        "subword_tok": subword_tok,
        "char_tok": char_tok,
        "nlm_dict": nlm_dict,
        "alignment_engine": alignment_engine,
        "config": config,
    }


class TestDownstreamCorrector:

    def test_mode_a_word_corrector(self, shared_pipeline_resources):
        res = shared_pipeline_resources
        model = SinhalaCharBERTForPreTraining(res["config"])
        corrector = BoundedWordCorrector(
            model=model,
            subword_tokenizer=res["subword_tok"],
            char_tokenizer=res["char_tok"],
            alignment_engine=res["alignment_engine"],
            nlm_dictionary=res["nlm_dict"],
            device=torch.device("cpu"),
        )

        test_text = "මම ගෙදර යනවා"
        corrected, candidates = corrector.correct_sequence(test_text)
        assert isinstance(corrected, str)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert isinstance(candidates[0], WordCorrectionCandidate)

    def test_mode_b_seq2seq_model_forward_and_generate(self, shared_pipeline_resources):
        res = shared_pipeline_resources
        seq2seq_model = SinhalaCharBERTSeq2SeqModel(res["config"], num_decoder_layers=2)
        seq2seq_model.eval()

        batch_size = 2
        m_len = 10
        n_len = 25
        target_len = 15

        input_ids = torch.randint(0, res["config"].vocab_size, (batch_size, m_len))
        char_input_ids = torch.randint(0, res["config"].char_vocab_size, (batch_size, n_len))
        start_char_idx = torch.randint(0, 5, (batch_size, m_len))
        end_char_idx = torch.randint(5, 20, (batch_size, m_len))
        decoder_in = torch.randint(0, res["config"].char_vocab_size, (batch_size, target_len))
        labels = torch.randint(0, res["config"].char_vocab_size, (batch_size, target_len))

        # Forward pass
        output = seq2seq_model(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            decoder_input_ids=decoder_in,
            labels=labels,
        )

        assert isinstance(output, Seq2SeqCorrectionOutput)
        assert output.loss is not None
        assert output.logits.shape == (batch_size, target_len, res["config"].char_vocab_size)

        # Greedy generation
        gen_tokens = seq2seq_model.generate(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            max_length=10,
        )
        assert gen_tokens.dim() == 2
        assert gen_tokens.size(0) == batch_size
        assert gen_tokens.size(1) >= 1

    def test_unified_corrector_pipeline(self, shared_pipeline_resources):
        res = shared_pipeline_resources
        model = SinhalaCharBERTForPreTraining(res["config"])
        corrector = SinhalaCharBERTCorrector(
            model=model,
            subword_tokenizer=res["subword_tok"],
            char_tokenizer=res["char_tok"],
            alignment_engine=res["alignment_engine"],
            nlm_dictionary=res["nlm_dict"],
            device=torch.device("cpu"),
        )

        test_sentence = "මම ගෙදර යනවා"
        result = corrector.correct(test_sentence, mode="word_denoise")

        assert isinstance(result, CorrectionResult)
        assert isinstance(result.text, str)
        assert result.original_text == test_sentence
        assert isinstance(result.edits, list)
        assert isinstance(result.summary(), str)

    def test_edit_computation_and_categorization(self, shared_pipeline_resources):
        res = shared_pipeline_resources
        model = SinhalaCharBERTForPreTraining(res["config"])
        corrector = SinhalaCharBERTCorrector(
            model=model,
            subword_tokenizer=res["subword_tok"],
            char_tokenizer=res["char_tok"],
            alignment_engine=res["alignment_engine"],
            nlm_dictionary=res["nlm_dict"],
            device=torch.device("cpu"),
        )

        # Test murdhaja / dantaja category
        edits = corrector._compute_edits("කරුනාව", "කරුණාව")
        mod_edits = [e for e in edits if e.op_type != "equal"]
        assert len(mod_edits) == 1
        assert mod_edits[0].category == "murdhaja_dantaja"

        # Test dialectal shift category
        edits = corrector._compute_edits("යන්ඩ", "යන්න")
        mod_edits = [e for e in edits if e.op_type != "equal"]
        assert len(mod_edits) == 1
        assert mod_edits[0].category == "dialectal_morphology"

        # Test ligature repair
        edits = corrector._compute_edits("ශ්රී", "ශ්‍රී")
        mod_edits = [e for e in edits if e.op_type != "equal"]
        assert len(mod_edits) == 1
        assert mod_edits[0].category == "ligature_repair"
