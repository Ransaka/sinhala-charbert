"""
Mode B: Open-Vocabulary Seq2Seq Character-Level Transformer Decoder for Sinhala-CharBERT.
Autoregressively decodes sinlib phonological Akshara units conditioned on CharBERT's fused sequence states.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.file_utils import ModelOutput

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.models.modeling_charbert import SinhalaCharBERTModel, SinhalaCharBERTOutput


@dataclass
class Seq2SeqCorrectionOutput(ModelOutput):
    """Output container for SinhalaCharBERTSeq2SeqModel."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    encoder_hidden_states: Optional[SinhalaCharBERTOutput] = None
    decoder_hidden_states: Optional[torch.Tensor] = None


class SinhalaCharBERTDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with:
    1. Masked Causal Self-Attention
    2. Cross-Attention over CharBERT fused encoder sequence states
    3. GELU Position-wise Feed-Forward Network with residual LayerNorm
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 1. Causal Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 2. Cross-Attention over CharBERT Encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 3. Feed-Forward Network
        self.ffn_dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_act = nn.GELU() if config.hidden_act == "gelu" else nn.ReLU()
        self.ffn_dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Step 1: Masked Causal Self-Attention
        residual = hidden_states
        attn_out, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden_states = self.self_attn_layer_norm(residual + self.dropout(attn_out))

        # Step 2: Cross-Attention over CharBERT Fused State
        residual = hidden_states
        cross_out, _ = self.cross_attn(
            query=hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=encoder_key_padding_mask,
            need_weights=False,
        )
        hidden_states = self.cross_attn_layer_norm(residual + self.dropout(cross_out))

        # Step 3: Feed-Forward Network
        residual = hidden_states
        ffn_out = self.ffn_dense2(self.dropout(self.ffn_act(self.ffn_dense1(hidden_states))))
        hidden_states = self.ffn_layer_norm(residual + self.dropout(ffn_out))

        return hidden_states


class SinhalaCharBERTSeq2SeqModel(nn.Module):
    """
    Open-Vocabulary Seq2Seq Corrector (Mode B).
    Pairs the Sinhala-CharBERT dual-channel encoder with an autoregressive
    Transformer Decoder producing sinlib phonological Akshara units.
    """

    def __init__(
        self,
        config: SinhalaCharBERTConfig,
        num_decoder_layers: int = 4,
        max_target_positions: int = 512,
    ):
        super().__init__()
        self.config = config
        self.max_target_positions = max_target_positions

        # Encoder: Sinhala-CharBERT Dual-Channel Backbone
        self.encoder = SinhalaCharBERTModel(config)

        # Decoder Target Character Embeddings & Positional Embeddings
        self.target_char_embeddings = nn.Embedding(
            config.char_vocab_size,
            config.hidden_size,
            padding_idx=config.char_pad_token_id,
        )
        self.position_embeddings = nn.Embedding(max_target_positions, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Decoder Transformer Layers
        self.decoder_layers = nn.ModuleList(
            [SinhalaCharBERTDecoderLayer(config) for _ in range(num_decoder_layers)]
        )

        # Output LM Head over Character Vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.char_vocab_size, bias=False)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_target_positions, max_target_positions), float("-inf")), diagonal=1),
            persistent=False,
        )
        self.register_buffer(
            "position_ids",
            torch.arange(max_target_positions).expand((1, -1)),
            persistent=False,
        )

        self.init_decoder_weights()

    def init_decoder_weights(self):
        """Initializes decoder embeddings and projections."""
        for module in [self.target_char_embeddings, self.position_embeddings, self.lm_head]:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if hasattr(module, "padding_idx") and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        char_input_ids: torch.Tensor,
        start_char_idx: torch.Tensor,
        end_char_idx: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> Seq2SeqCorrectionOutput:
        """
        Forward pass through Seq2Seq Encoder-Decoder.
        """
        # 1. Encode with Sinhala-CharBERT Dual-Channel Backbone
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            attention_mask=attention_mask,
            char_attention_mask=char_attention_mask,
        )
        fused_hidden = encoder_outputs.fused_hidden_state  # (batch, seq_len, hidden_size)

        # 2. Embed Decoder Target Inputs
        target_seq_len = decoder_input_ids.size(1)
        if target_seq_len > self.max_target_positions:
            decoder_input_ids = decoder_input_ids[:, : self.max_target_positions]
            target_seq_len = self.max_target_positions

        target_pos_ids = self.position_ids[:, :target_seq_len]
        char_embeds = self.target_char_embeddings(decoder_input_ids)
        pos_embeds = self.position_embeddings(target_pos_ids)

        decoder_hidden = self.embed_layer_norm(char_embeds + pos_embeds)
        decoder_hidden = self.embed_dropout(decoder_hidden)

        # Construct causal mask
        causal_mask = self.causal_mask[:target_seq_len, :target_seq_len]

        # Encoder key padding mask (True for padding positions)
        enc_key_padding = None
        if attention_mask is not None:
            enc_key_padding = (attention_mask == 0)

        # 3. Pass through Decoder Stack
        for layer in self.decoder_layers:
            decoder_hidden = layer(
                hidden_states=decoder_hidden,
                encoder_hidden_states=fused_hidden,
                causal_mask=causal_mask,
                encoder_key_padding_mask=enc_key_padding,
            )

        # 4. Predict Character Logits
        logits = self.lm_head(decoder_hidden)  # (batch, target_seq_len, char_vocab_size)

        loss = None
        if labels is not None:
            if labels.size(1) > target_seq_len:
                labels = labels[:, :target_seq_len]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
            loss = loss_fct(
                logits.view(-1, self.config.char_vocab_size),
                labels.contiguous().view(-1),
            )

        return Seq2SeqCorrectionOutput(
            loss=loss,
            logits=logits,
            encoder_hidden_states=encoder_outputs,
            decoder_hidden_states=decoder_hidden,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        char_input_ids: torch.Tensor,
        start_char_idx: torch.Tensor,
        end_char_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressively generates target phonological character units via greedy / temperature sampling.
        Returns generated character ID tensor of shape (batch, generated_len).
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Encode input sequence once
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            attention_mask=attention_mask,
            char_attention_mask=char_attention_mask,
        )
        fused_hidden = encoder_outputs.fused_hidden_state

        enc_key_padding = None
        if attention_mask is not None:
            enc_key_padding = (attention_mask == 0)

        # Initialize generation buffer with <bos>
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length):
            curr_seq_len = generated.size(1)
            target_pos_ids = self.position_ids[:, :curr_seq_len]
            char_embeds = self.target_char_embeddings(generated)
            pos_embeds = self.position_embeddings(target_pos_ids)

            dec_hidden = self.embed_layer_norm(char_embeds + pos_embeds)
            causal_mask = self.causal_mask[:curr_seq_len, :curr_seq_len]

            for layer in self.decoder_layers:
                dec_hidden = layer(
                    hidden_states=dec_hidden,
                    encoder_hidden_states=fused_hidden,
                    causal_mask=causal_mask,
                    encoder_key_padding_mask=enc_key_padding,
                )

            logits = self.lm_head(dec_hidden[:, -1, :])  # (batch, char_vocab_size)

            if temperature <= 0.0 or temperature == 1.0:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            # Update is_finished mask if <eos> is produced
            is_finished |= (next_tokens.squeeze(-1) == eos_token_id)
            generated = torch.cat([generated, next_tokens], dim=-1)

            if is_finished.all():
                break

        return generated
