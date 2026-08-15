"""
Sinhala-CharBERT Core PyTorch Model Architecture and Pre-Training Engine.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel
from transformers.file_utils import ModelOutput

from sinhala_charbert.config.model_config import SinhalaCharBERTConfig
from sinhala_charbert.models.embeddings import SinhalaTokenEmbeddings, SinhalaCharEmbeddings
from sinhala_charbert.models.char_encoder import CharacterBiGRUEncoder
from sinhala_charbert.models.encoder import SinhalaCharBERTEncoder
from sinhala_charbert.models.heads import SinhalaTokenMLMHead, SinhalaCharNLMHead


@dataclass
class SinhalaCharBERTOutput(ModelOutput):
    """Base model output for Sinhala-CharBERT."""
    last_hidden_state: torch.Tensor = None
    last_char_hidden_state: torch.Tensor = None
    fused_hidden_state: torch.Tensor = None
    all_token_hidden_states: Optional[List[torch.Tensor]] = None
    all_char_hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None


@dataclass
class SinhalaCharBERTPreTrainingOutput(ModelOutput):
    """Pre-training output for Sinhala-CharBERT with MLM and NLM losses and logits."""
    loss: Optional[torch.Tensor] = None
    mlm_loss: Optional[torch.Tensor] = None
    nlm_loss: Optional[torch.Tensor] = None
    token_logits: torch.Tensor = None
    char_logits: torch.Tensor = None
    hidden_states: Optional[SinhalaCharBERTOutput] = None


class SinhalaCharBERTModel(nn.Module):
    """
    Dual-Channel Transformer Backbone for Sinhala-CharBERT.
    Processes subword tokens in parallel with phonological character units,
    bridging them with boundary pooling and Heterogeneous Interaction modules.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = SinhalaTokenEmbeddings(config)
        self.char_embeddings = SinhalaCharEmbeddings(config)
        self.char_encoder = CharacterBiGRUEncoder(config)
        self.encoder = SinhalaCharBERTEncoder(config)

        # Final fusion projection
        self.fused_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.init_weights()

    def init_weights(self):
        """Initializes model weights following standard normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.zero_()

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Converts 2D (batch, seq_len) attention mask into 4D (batch, 1, 1, seq_len) broadcastable mask."""
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            extended_attention_mask = attention_mask

        extended_attention_mask = (1.0 - extended_attention_mask.to(dtype=torch.float32)) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        char_input_ids: torch.Tensor,
        start_char_idx: torch.Tensor,
        end_char_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> SinhalaCharBERTOutput:
        """
        Forward pass through the dual-channel backbone.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if char_attention_mask is None:
            char_attention_mask = torch.ones_like(char_input_ids)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        # 1. Embed Token and Character channels
        token_embeds = self.token_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        char_embeds = self.char_embeddings(char_input_ids=char_input_ids)

        # 2. Encode characters via Bi-GRU and perform boundary pooling
        token_aligned_char_embeds = self.char_encoder(
            char_embeddings=char_embeds,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            char_attention_mask=char_attention_mask,
        )

        # 3. Pass through interleaved Transformer + HI Encoder stack
        (
            final_token_hidden,
            final_char_hidden,
            all_token_hidden,
            all_char_hidden,
            attentions,
        ) = self.encoder(
            token_hidden_states=token_embeds,
            char_hidden_states=token_aligned_char_embeds,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # 4. Synthesize unified fused representation Z = LayerNorm(W_z [T ; H])
        fused_cat = torch.cat([final_token_hidden, final_char_hidden], dim=-1)
        fused_hidden = self.layer_norm(self.fused_proj(fused_cat))

        return SinhalaCharBERTOutput(
            last_hidden_state=final_token_hidden,
            last_char_hidden_state=final_char_hidden,
            fused_hidden_state=fused_hidden,
            all_token_hidden_states=all_token_hidden,
            all_char_hidden_states=all_char_hidden,
            attentions=attentions,
        )


class SinhalaCharBERTForPreTraining(nn.Module):
    """
    Sinhala-CharBERT model with Masked Language Modeling (Token Channel)
    and Noisy Language Modeling (Character Channel Denoising) pre-training heads.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.config = config
        self.charbert = SinhalaCharBERTModel(config)
        self.mlm_head = SinhalaTokenMLMHead(config)
        self.nlm_head = SinhalaCharNLMHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        char_input_ids: torch.Tensor,
        start_char_idx: torch.Tensor,
        end_char_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        nlm_labels: Optional[torch.Tensor] = None,
        mlm_loss_weight: float = 1.0,
        nlm_loss_weight: float = 1.0,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> SinhalaCharBERTPreTrainingOutput:
        """
        Forward pass with dual MLM + NLM pre-training loss computation.
        """
        outputs = self.charbert(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            attention_mask=attention_mask,
            char_attention_mask=char_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        token_hidden = outputs.last_hidden_state
        char_hidden = outputs.last_char_hidden_state

        token_logits = self.mlm_head(token_hidden)
        char_logits = self.nlm_head(char_hidden)

        total_loss = None
        mlm_loss = None
        nlm_loss = None

        if mlm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                token_logits.view(-1, self.config.vocab_size),
                mlm_labels.view(-1),
            )

        if nlm_labels is not None:
            nlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            nlm_loss = nlm_loss_fct(
                char_logits.view(-1, self.config.nlm_vocab_size),
                nlm_labels.view(-1),
            )

        if mlm_loss is not None and nlm_loss is not None:
            total_loss = (mlm_loss_weight * mlm_loss) + (nlm_loss_weight * nlm_loss)
        elif mlm_loss is not None:
            total_loss = mlm_loss
        elif nlm_loss is not None:
            total_loss = nlm_loss

        return SinhalaCharBERTPreTrainingOutput(
            loss=total_loss,
            mlm_loss=mlm_loss,
            nlm_loss=nlm_loss,
            token_logits=token_logits,
            char_logits=char_logits,
            hidden_states=outputs,
        )

    @classmethod
    def from_pretrained_backbone(
        cls,
        backbone_model_or_path: Union[str, nn.Module],
        config: Optional[SinhalaCharBERTConfig] = None,
        char_vocab_size: Optional[int] = None,
        nlm_vocab_size: Optional[int] = None,
        **kwargs,
    ) -> "SinhalaCharBERTForPreTraining":
        """
        Loads pre-trained weights from a standard HuggingFace BERT or BertForMaskedLM checkpoint
        or instance into Token Embeddings, Transformer Encoder layers, and MLM Prediction Head.
        The Bi-GRU, HI modules, and NLM head are randomly initialized.
        """
        if isinstance(backbone_model_or_path, nn.Module):
            backbone_model = backbone_model_or_path
            backbone_config = backbone_model.config
            source_desc = type(backbone_model).__name__
        else:
            print(f"Loading pre-trained backbone weights from '{backbone_model_or_path}'...")
            try:
                from transformers import AutoModelForMaskedLM
                backbone_model = AutoModelForMaskedLM.from_pretrained(backbone_model_or_path, **kwargs)
            except Exception:
                backbone_model = AutoModel.from_pretrained(backbone_model_or_path, **kwargs)
            backbone_config = backbone_model.config
            source_desc = str(backbone_model_or_path)

        if config is None:
            config = SinhalaCharBERTConfig(
                vocab_size=backbone_config.vocab_size,
                char_vocab_size=char_vocab_size or 1500,
                nlm_vocab_size=nlm_vocab_size or 32000,
                hidden_size=backbone_config.hidden_size,
                char_gru_hidden_size=backbone_config.hidden_size // 2,
                num_hidden_layers=backbone_config.num_hidden_layers,
                num_attention_heads=getattr(backbone_config, "num_attention_heads", 12),
                intermediate_size=getattr(backbone_config, "intermediate_size", 3072),
                hidden_act=getattr(backbone_config, "hidden_act", "gelu"),
                max_position_embeddings=getattr(backbone_config, "max_position_embeddings", 256),
                type_vocab_size=getattr(backbone_config, "type_vocab_size", 2),
                layer_norm_eps=getattr(backbone_config, "layer_norm_eps", 1e-12),
                pad_token_id=getattr(backbone_config, "pad_token_id", 0) or 0,
                backbone_model_name_or_path=source_desc if isinstance(backbone_model_or_path, str) else None,
            )

        model = cls(config)

        # Unpack BertModel if wrapped in BertForMaskedLM / BertForPreTraining
        bert_module = getattr(backbone_model, "bert", backbone_model)

        # 1. Transfer Token Embeddings
        if hasattr(bert_module, "embeddings"):
            b_emb = bert_module.embeddings
            model.charbert.token_embeddings.word_embeddings.load_state_dict(
                b_emb.word_embeddings.state_dict()
            )
            model.charbert.token_embeddings.position_embeddings.load_state_dict(
                b_emb.position_embeddings.state_dict()
            )
            if hasattr(b_emb, "token_type_embeddings") and b_emb.token_type_embeddings is not None:
                model.charbert.token_embeddings.token_type_embeddings.load_state_dict(
                    b_emb.token_type_embeddings.state_dict()
                )
            if hasattr(b_emb, "LayerNorm") and b_emb.LayerNorm is not None:
                model.charbert.token_embeddings.layer_norm.load_state_dict(
                    b_emb.LayerNorm.state_dict()
                )

        # 2. Transfer Transformer Encoder Layers
        if hasattr(bert_module, "encoder") and hasattr(bert_module.encoder, "layer"):
            for l_idx, b_layer in enumerate(bert_module.encoder.layer):
                if l_idx < len(model.charbert.encoder.layers):
                    target_tf_layer = model.charbert.encoder.layers[l_idx].transformer_layer
                    target_tf_layer.load_state_dict(b_layer.state_dict())

        # 3. Transfer MLM Prediction Head (if available in BertForMaskedLM)
        cls_module = getattr(backbone_model, "cls", None)
        if cls_module is not None and hasattr(cls_module, "predictions"):
            preds = cls_module.predictions
            if hasattr(preds, "transform"):
                if hasattr(preds.transform, "dense"):
                    model.mlm_head.dense.load_state_dict(preds.transform.dense.state_dict())
                if hasattr(preds.transform, "LayerNorm"):
                    model.mlm_head.layer_norm.load_state_dict(preds.transform.LayerNorm.state_dict())
            if hasattr(preds, "decoder"):
                model.mlm_head.decoder.load_state_dict(preds.decoder.state_dict())

        print(f"Successfully transferred backbone weights from {source_desc} into Sinhala-CharBERT.")
        return model
