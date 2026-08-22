"""
Unit tests for Sinhala-CharBERT model architecture, Bi-GRU boundary pooling, HI module, and pre-training objectives.
"""

import unittest
import torch

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.models.embeddings import SinhalaTokenEmbeddings, SinhalaCharEmbeddings
from sinhala_charbert.models.char_encoder import CharacterBiGRUEncoder
from sinhala_charbert.models.hi_module import HeterogeneousInteractionModule
from sinhala_charbert.models.modeling_charbert import (
    SinhalaCharBERTModel,
    SinhalaCharBERTForPreTraining,
)


class TestSinhalaCharBERTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SinhalaCharBERTConfig(
            vocab_size=1000,
            char_vocab_size=200,
            nlm_vocab_size=500,
            hidden_size=64,
            char_embedding_dim=32,
            char_gru_hidden_size=32,  # 32 * 2 = 64 bidirectional hidden size
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            hi_kernel_sizes=[1, 3, 5],
        )

    def test_embeddings(self):
        batch_size, m_len, n_len = 2, 8, 20
        token_ids = torch.randint(0, self.config.vocab_size, (batch_size, m_len))
        char_ids = torch.randint(0, self.config.char_vocab_size, (batch_size, n_len))

        token_emb = SinhalaTokenEmbeddings(self.config)
        char_emb = SinhalaCharEmbeddings(self.config)

        t_out = token_emb(token_ids)
        c_out = char_emb(char_ids)

        self.assertEqual(t_out.shape, (batch_size, m_len, self.config.hidden_size))
        self.assertEqual(c_out.shape, (batch_size, n_len, self.config.char_embedding_dim))

    def test_char_encoder_boundary_pooling(self):
        batch_size, m_len, n_len = 2, 6, 18
        char_emb = SinhalaCharEmbeddings(self.config)
        char_encoder = CharacterBiGRUEncoder(self.config)

        char_ids = torch.randint(0, self.config.char_vocab_size, (batch_size, n_len))
        c_out = char_emb(char_ids)

        start_char_idx = torch.tensor([[0, 1, 4, 7, 10, 17], [0, 2, 5, 8, 12, 17]])
        end_char_idx = torch.tensor([[0, 3, 6, 9, 15, 17], [0, 4, 7, 11, 16, 17]])

        pooled_char = char_encoder(
            char_embeddings=c_out,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
        )

        self.assertEqual(pooled_char.shape, (batch_size, m_len, self.config.hidden_size))

    def test_hi_module_forward(self):
        batch_size, m_len = 2, 8
        hi_module = HeterogeneousInteractionModule(self.config)

        token_repr = torch.randn(batch_size, m_len, self.config.hidden_size)
        char_repr = torch.randn(batch_size, m_len, self.config.hidden_size)

        updated_token, updated_char = hi_module(token_repr, char_repr)

        self.assertEqual(updated_token.shape, (batch_size, m_len, self.config.hidden_size))
        self.assertEqual(updated_char.shape, (batch_size, m_len, self.config.hidden_size))

    def test_full_backbone_model(self):
        batch_size, m_len, n_len = 2, 10, 25
        model = SinhalaCharBERTModel(self.config)

        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, m_len))
        char_input_ids = torch.randint(0, self.config.char_vocab_size, (batch_size, n_len))
        start_char_idx = torch.randint(0, n_len // 2, (batch_size, m_len))
        end_char_idx = torch.randint(n_len // 2, n_len, (batch_size, m_len))

        outputs = model(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            output_hidden_states=True,
        )

        self.assertEqual(outputs.last_hidden_state.shape, (batch_size, m_len, self.config.hidden_size))
        self.assertEqual(outputs.last_char_hidden_state.shape, (batch_size, m_len, self.config.hidden_size))
        self.assertEqual(outputs.fused_hidden_state.shape, (batch_size, m_len, self.config.hidden_size))
        self.assertEqual(len(outputs.all_token_hidden_states), self.config.num_hidden_layers + 1)

    def test_pretraining_loss_and_backward(self):
        batch_size, m_len, n_len = 2, 8, 20
        model = SinhalaCharBERTForPreTraining(self.config)

        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, m_len))
        char_input_ids = torch.randint(0, self.config.char_vocab_size, (batch_size, n_len))
        start_char_idx = torch.randint(0, n_len // 2, (batch_size, m_len))
        end_char_idx = torch.randint(n_len // 2, n_len, (batch_size, m_len))

        mlm_labels = torch.randint(0, self.config.vocab_size, (batch_size, m_len))
        nlm_labels = torch.randint(0, self.config.nlm_vocab_size, (batch_size, m_len))

        outputs = model(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            mlm_labels=mlm_labels,
            nlm_labels=nlm_labels,
            return_logits=True,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.mlm_loss)
        self.assertIsNotNone(outputs.nlm_loss)
        self.assertEqual(outputs.token_logits.shape, (batch_size, m_len, self.config.vocab_size))
        self.assertEqual(outputs.char_logits.shape, (batch_size, m_len, self.config.nlm_vocab_size))

        # Test backward pass gradient propagation
        outputs.loss.backward()

        # Verify gradients exist in TokenEmbeddings, CharBiGRU, and HI module
        self.assertIsNotNone(model.charbert.token_embeddings.word_embeddings.weight.grad)
        self.assertIsNotNone(model.charbert.char_encoder.gru.weight_ih_l0.grad)
        self.assertIsNotNone(model.charbert.encoder.layers[0].hi_module.conv_proj.weight.grad)
        self.assertIsNotNone(model.nlm_head.decoder.weight.grad)

    def test_from_pretrained_bert_for_masked_lm(self):
        from transformers import BertConfig, BertForMaskedLM

        # Exact structure matching the user's pre-trained BERT backbone
        backbone_config = BertConfig(
            vocab_size=32000,
            hidden_size=786,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1024,
            max_position_embeddings=256,
        )
        backbone_model = BertForMaskedLM(backbone_config)

        # Transfer directly from instantiated BertForMaskedLM
        charbert = SinhalaCharBERTForPreTraining.from_pretrained_backbone(backbone_model)
        self.assertIsNotNone(charbert)
        self.assertEqual(charbert.config.vocab_size, 32000)
        self.assertEqual(charbert.config.hidden_size, 786)
        self.assertEqual(charbert.config.char_gru_hidden_size, 393)  # 393 * 2 = 786
        self.assertEqual(len(charbert.charbert.encoder.layers), 6)
        self.assertEqual(charbert.config.max_position_embeddings, 256)
        self.assertEqual(charbert.mlm_head.decoder.out_features, 32000)


if __name__ == "__main__":
    unittest.main()
