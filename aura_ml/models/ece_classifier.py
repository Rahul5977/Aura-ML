"""
Emotion Cause Extraction Model using RoBERTa
==============================================

Two-stage architecture for ECE:
1. Clause-level binary classification (contains cause or not)
2. Token-level BIO tagging (B-CAUSE, I-CAUSE, O)

Author: Senior ML Research Engineer
Date: November 2025
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ECEModelOutput:
    """
    Output class for ECE model predictions.
    
    Attributes:
        loss: Combined loss from both classification heads
        clause_loss: Binary classification loss
        token_loss: Token classification (BIO tagging) loss
        clause_logits: Logits for clause-level classification [batch_size, 2]
        token_logits: Logits for token-level classification [batch_size, seq_len, 3]
        hidden_states: Optional hidden states from RoBERTa
        attentions: Optional attention weights from RoBERTa
    """
    loss: Optional[torch.FloatTensor] = None
    clause_loss: Optional[torch.FloatTensor] = None
    token_loss: Optional[torch.FloatTensor] = None
    clause_logits: torch.FloatTensor = None
    token_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RoBERTaForECE(RobertaPreTrainedModel):
    """
    RoBERTa-based model for Emotion Cause Extraction with two-stage architecture.
    
    Architecture:
        1. RoBERTa Base Encoder (roberta-base)
        2. Stage 1: Clause-level binary classifier (CLS token → FC → 2 classes)
        3. Stage 2: Token-level BIO tagger (All tokens → FC → 3 classes)
    
    The model jointly optimizes both objectives with weighted loss combination.
    """
    
    def __init__(self, config: RobertaConfig):
        """
        Initialize the ECE model.
        
        Args:
            config: RoBERTa configuration object
        """
        super().__init__(config)
        self.num_clause_labels = 2  # Binary: has_cause (1) or no_cause (0)
        self.num_token_labels = 3   # BIO: B-CAUSE (0), I-CAUSE (1), O (2)
        self.config = config
        
        # RoBERTa encoder
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # Dropout for regularization
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Stage 1: Clause-level binary classification head
        # Takes [CLS] token representation and predicts if clause contains cause
        self.clause_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, self.num_clause_labels)
        )
        
        # Stage 2: Token-level BIO tagging head
        # Takes all token representations and predicts BIO tag for each
        self.token_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, self.num_token_labels)
        )
        
        # Loss weights for multi-task learning
        self.clause_loss_weight = 0.3  # Weight for clause classification
        self.token_loss_weight = 0.7   # Weight for token classification
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        clause_labels: Optional[torch.LongTensor] = None,
        token_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ECEModelOutput:
        """
        Forward pass through the ECE model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            head_mask: Head mask for attention [num_heads] or [num_layers, num_heads]
            inputs_embeds: Pre-computed embeddings [batch_size, seq_len, hidden_size]
            clause_labels: Binary labels for clause classification [batch_size]
            token_labels: BIO labels for token classification [batch_size, seq_len]
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return ECEModelOutput object
            
        Returns:
            ECEModelOutput containing losses and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # ====================================================================
        # STEP 1: Encode input with RoBERTa
        # ====================================================================
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get hidden states [batch_size, seq_len, hidden_size]
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # ====================================================================
        # STEP 2: Stage 1 - Clause-level Binary Classification
        # ====================================================================
        # Extract [CLS] token representation (first token)
        cls_token_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Pass through clause classifier
        clause_logits = self.clause_classifier(cls_token_output)  # [batch_size, 2]
        
        # ====================================================================
        # STEP 3: Stage 2 - Token-level BIO Tagging
        # ====================================================================
        # Pass all tokens through token classifier
        token_logits = self.token_classifier(sequence_output)  # [batch_size, seq_len, 3]
        
        # ====================================================================
        # STEP 4: Calculate Losses (if labels provided)
        # ====================================================================
        total_loss = None
        clause_loss = None
        token_loss = None
        
        if clause_labels is not None:
            # Clause classification loss (CrossEntropy)
            loss_fct_clause = CrossEntropyLoss()
            clause_loss = loss_fct_clause(
                clause_logits.view(-1, self.num_clause_labels),
                clause_labels.view(-1)
            )
        
        if token_labels is not None:
            # Token classification loss (CrossEntropy with attention mask)
            loss_fct_token = CrossEntropyLoss()
            
            # Only compute loss on attended tokens (mask out padding)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.num_token_labels)[active_loss]
                active_labels = token_labels.view(-1)[active_loss]
                token_loss = loss_fct_token(active_logits, active_labels)
            else:
                token_loss = loss_fct_token(
                    token_logits.view(-1, self.num_token_labels),
                    token_labels.view(-1)
                )
        
        # Combine losses with weights
        if clause_loss is not None and token_loss is not None:
            total_loss = (
                self.clause_loss_weight * clause_loss + 
                self.token_loss_weight * token_loss
            )
        elif clause_loss is not None:
            total_loss = clause_loss
        elif token_loss is not None:
            total_loss = token_loss
        
        # ====================================================================
        # STEP 5: Return Output
        # ====================================================================
        return ECEModelOutput(
            loss=total_loss,
            clause_loss=clause_loss,
            token_loss=token_loss,
            clause_logits=clause_logits,
            token_logits=token_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
    
    def predict_causes(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        tokenizer,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        High-level prediction method for inference.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            tokenizer: Tokenizer for decoding tokens
            threshold: Confidence threshold for clause classification
            
        Returns:
            Dictionary containing:
                - has_cause: Boolean indicating if cause is present
                - cause_confidence: Confidence score for cause presence
                - cause_tokens: List of tokens identified as cause
                - cause_spans: List of (start, end) positions for cause spans
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Stage 1: Check if clause contains cause
        clause_probs = torch.softmax(outputs.clause_logits, dim=-1)
        has_cause_prob = clause_probs[0, 1].item()  # Probability of class 1 (has cause)
        has_cause = has_cause_prob > threshold
        
        # Stage 2: Extract cause tokens if present
        cause_tokens = []
        cause_spans = []
        
        if has_cause:
            token_predictions = torch.argmax(outputs.token_logits[0], dim=-1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            current_span = []
            start_idx = None
            
            for idx, (token, pred) in enumerate(zip(tokens, token_predictions)):
                if attention_mask[0, idx] == 0:  # Skip padding
                    continue
                    
                # BIO tagging: 0=B-CAUSE, 1=I-CAUSE, 2=O
                if pred == 0:  # B-CAUSE
                    if current_span:  # Save previous span
                        cause_tokens.append(current_span)
                        cause_spans.append((start_idx, idx - 1))
                    current_span = [token]
                    start_idx = idx
                elif pred == 1 and current_span:  # I-CAUSE
                    current_span.append(token)
                else:  # O or I-CAUSE without B-CAUSE
                    if current_span:
                        cause_tokens.append(current_span)
                        cause_spans.append((start_idx, idx - 1))
                        current_span = []
                        start_idx = None
            
            # Add final span if exists
            if current_span:
                cause_tokens.append(current_span)
                cause_spans.append((start_idx, len(tokens) - 1))
        
        return {
            'has_cause': has_cause,
            'cause_confidence': has_cause_prob,
            'cause_tokens': cause_tokens,
            'cause_spans': cause_spans
        }


def load_ece_model(
    model_name_or_path: str = "roberta-base",
    num_clause_labels: int = 2,
    num_token_labels: int = 3,
    device: Optional[torch.device] = None
) -> RoBERTaForECE:
    """
    Load and initialize the ECE model.
    
    Args:
        model_name_or_path: Path to pretrained model or model identifier
        num_clause_labels: Number of clause classification labels (default: 2)
        num_token_labels: Number of token classification labels (default: 3)
        device: Device to load model on (if None, automatically selects)
        
    Returns:
        Initialized RoBERTaForECE model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration and model
    config = RobertaConfig.from_pretrained(model_name_or_path)
    model = RoBERTaForECE.from_pretrained(model_name_or_path, config=config)
    model.to(device)
    
    print(f"✅ Loaded ECE model from {model_name_or_path}")
    print(f"📍 Device: {device}")
    print(f"📊 Clause labels: {num_clause_labels}, Token labels: {num_token_labels}")
    print(f"🔢 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎓 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


if __name__ == "__main__":
    """Test the model architecture."""
    print("="*80)
    print("Testing RoBERTaForECE Model Architecture")
    print("="*80)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ece_model(device=device)
    
    # Create dummy input (on same device as model)
    batch_size = 2
    seq_length = 32
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    dummy_attention_mask = torch.ones((batch_size, seq_length)).to(device)
    dummy_clause_labels = torch.randint(0, 2, (batch_size,)).to(device)
    dummy_token_labels = torch.randint(0, 3, (batch_size, seq_length)).to(device)
    
    # Forward pass
    print("\n🔄 Running forward pass...")
    outputs = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        clause_labels=dummy_clause_labels,
        token_labels=dummy_token_labels
    )
    
    print(f"\n📊 Output shapes:")
    print(f"   Total loss: {outputs.loss.item():.4f}")
    print(f"   Clause loss: {outputs.clause_loss.item():.4f}")
    print(f"   Token loss: {outputs.token_loss.item():.4f}")
    print(f"   Clause logits shape: {outputs.clause_logits.shape}")
    print(f"   Token logits shape: {outputs.token_logits.shape}")
    
    print("\n✅ Model architecture test passed!")
