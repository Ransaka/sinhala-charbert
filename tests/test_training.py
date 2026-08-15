"""
Unit tests for Sinhala-CharBERT Pre-Training Engine, Noise Curriculum, and NLM Dictionary.
"""

import tempfile
import unittest
import torch
from transformers import AutoTokenizer

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.config.training_config import TrainingConfig
from sinhala_charbert.data.char_tokenizer import SinhalaCharTokenizer
from sinhala_charbert.data.alignment import SequenceAlignmentEngine
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTForPreTraining
from sinhala_charbert.training.curriculum import NoiseCurriculumScheduler
from sinhala_charbert.training.dictionary import SinhalaNLMDictionary
from sinhala_charbert.training.dataset import (
    SinhalaCharBERTPretrainDataset,
    PretrainDualChannelCollator,
)
from sinhala_charbert.training.trainer import SinhalaCharBERTTrainer


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_texts = [
            "මම ගෙදර යනවා සහ පොතක් කියවනවා.",
            "සිංහල භාෂාව ආරක්ෂා කර ගැනීම අපේ යුතුකමකි.",
            "ක්‍රිකට් ක්‍රීඩාව ලංකාවේ ඉතා ජනප්‍රියයි.",
            "අද කාලගුණය ඉතා යහපත් වේ.",
        ]

    def test_curriculum_scheduler(self):
        scheduler = NoiseCurriculumScheduler(total_steps=1000, phase1_ratio=0.2, phase2_ratio=0.4)
        
        # Step 50 -> Phase 1
        p1 = scheduler.get_profile(50)
        self.assertEqual(p1.dialect_rate, 0.0)
        self.assertEqual(p1.wijesekara_keystroke_rate, 0.0)
        self.assertTrue("Phase 1" in scheduler.get_phase_name(50))

        # Step 300 -> Phase 2
        p2 = scheduler.get_profile(300)
        self.assertGreater(p2.wijesekara_keystroke_rate, 0.0)
        self.assertTrue("Phase 2" in scheduler.get_phase_name(300))

        # Step 800 -> Phase 3
        p3 = scheduler.get_profile(800)
        self.assertGreater(p3.code_switch_rate, 0.0)
        self.assertTrue("Phase 3" in scheduler.get_phase_name(800))

    def test_nlm_dictionary_build_and_save(self):
        nlm_dict = SinhalaNLMDictionary()
        nlm_dict.build_from_corpus(self.sample_texts, max_words=50, min_freq=1)

        self.assertGreater(nlm_dict.vocab_size, 2)
        self.assertIn("ගෙදර", nlm_dict.vocab_map)
        
        word_id = nlm_dict.word_to_id("ගෙදර")
        self.assertEqual(nlm_dict.id_to_word(word_id), "ගෙදර")

        # Test unknown token lookup
        self.assertEqual(nlm_dict.word_to_id("non_existent_token_123"), nlm_dict.unk_id)

        # Test save and load
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = f"{tmp_dir}/nlm_vocab.json"
            nlm_dict.save(save_path)
            loaded_dict = SinhalaNLMDictionary.load(save_path)
            self.assertEqual(loaded_dict.vocab_size, nlm_dict.vocab_size)
            self.assertEqual(loaded_dict.word_to_id("ගෙදර"), word_id)

    def test_pretrain_dataset_and_collator(self):
        subword_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        char_tokenizer = SinhalaCharTokenizer()
        char_tokenizer.train_on_corpus(self.sample_texts)

        alignment_engine = SequenceAlignmentEngine(
            subword_tokenizer=subword_tokenizer,
            char_tokenizer=char_tokenizer,
        )

        nlm_dict = SinhalaNLMDictionary()
        nlm_dict.build_from_corpus(self.sample_texts, max_words=50, min_freq=1)

        curriculum = NoiseCurriculumScheduler(total_steps=100)
        dataset = SinhalaCharBERTPretrainDataset(
            texts=self.sample_texts,
            subword_tokenizer=subword_tokenizer,
            char_tokenizer=char_tokenizer,
            alignment_engine=alignment_engine,
            nlm_dictionary=nlm_dict,
            curriculum_scheduler=curriculum,
            nlm_probability=0.5,
        )

        self.assertEqual(len(dataset), len(self.sample_texts))
        item = dataset[0]
        self.assertIn("aligned_seq", item)
        self.assertIn("nlm_labels", item)

        # Test batch collation
        collator = PretrainDualChannelCollator(
            subword_pad_token_id=0,
            subword_mask_token_id=103,
            subword_vocab_size=subword_tokenizer.vocab_size,
        )
        batch = collator([dataset[0], dataset[1]])

        self.assertIn("input_ids", batch)
        self.assertIn("char_input_ids", batch)
        self.assertIn("mlm_labels", batch)
        self.assertIn("nlm_labels", batch)
        self.assertEqual(batch["nlm_labels"].shape, batch["input_ids"].shape)

    def test_trainer_single_step(self):
        subword_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        char_tokenizer = SinhalaCharTokenizer()
        char_tokenizer.train_on_corpus(self.sample_texts)

        alignment_engine = SequenceAlignmentEngine(
            subword_tokenizer=subword_tokenizer,
            char_tokenizer=char_tokenizer,
        )

        nlm_dict = SinhalaNLMDictionary()
        nlm_dict.build_from_corpus(self.sample_texts, max_words=50, min_freq=1)

        config = SinhalaCharBERTConfig(
            vocab_size=subword_tokenizer.vocab_size,
            char_vocab_size=char_tokenizer.vocab_size + 10,
            nlm_vocab_size=nlm_dict.vocab_size + 10,
            hidden_size=64,
            char_embedding_dim=32,
            char_gru_hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
        )
        model = SinhalaCharBERTForPreTraining(config)

        dataset = SinhalaCharBERTPretrainDataset(
            texts=self.sample_texts,
            subword_tokenizer=subword_tokenizer,
            char_tokenizer=char_tokenizer,
            alignment_engine=alignment_engine,
            nlm_dictionary=nlm_dict,
        )

        train_cfg = TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            max_steps=2,
            warmup_steps=1,
            fp16=False,
        )

        trainer = SinhalaCharBERTTrainer(
            model=model,
            train_dataset=dataset,
            config=train_cfg,
            device=torch.device("cpu"),
        )

        collator = PretrainDualChannelCollator(
            subword_pad_token_id=0,
            subword_vocab_size=config.vocab_size,
        )
        batch = collator([dataset[0], dataset[1]])

        step_metrics = trainer.train_step(batch)
        self.assertIn("loss", step_metrics)
        self.assertGreater(step_metrics["loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
