import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union


class HistoryEncoder(nn.Module):
    """
    Encoder module for processing history sequences using Transformer architecture.
    
    Wraps nn.TransformerEncoder with configurable parameters.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        """
        Initialize the history encoder.
        
        Args:
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Create encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,  # Standard feedforward dimension
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # Create transformer encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
               - B: Batch size
               - T: Sequence length (history length)
               - d_model: Model dimension
            return_attention: If True, returns attention weights for visualization
               
        Returns:
            If return_attention=False:
                Encoded tensor of shape (B, T, d_model)
            If return_attention=True:
                Tuple of (encoded_tensor, attention_dict) where:
                - encoded_tensor: (B, T, d_model)
                - attention_dict: Dictionary with keys "encoder_attn" containing
                  list of attention weights per layer, each of shape (B, n_heads, T, T)
        """
        if not return_attention:
            return self.encoder(x)
        
        # Extract attention weights by manually forwarding through layers
        # and capturing attention from self-attention modules
        attention_weights = []
        encoded = x
        
        # Manually forward through each encoder layer to capture attention
        for layer in self.encoder.layers:
            # Store original self_attn forward
            original_self_attn = layer.self_attn.forward
            
            # Create wrapper to capture attention weights
            captured_attn = [None]
            
            def make_attn_wrapper(captured_ref):
                def attn_wrapper(query, key, value, key_padding_mask=None, need_weights=True, 
                                attn_mask=None, average_attn_weights=False, is_causal=False):
                    # Force need_weights=True to get attention
                    result = original_self_attn(
                        query, key, value, 
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                        attn_mask=attn_mask,
                        average_attn_weights=False,  # Get per-head attention
                        is_causal=is_causal
                    )
                    if isinstance(result, tuple):
                        attn_output, attn_weights = result
                        # attn_weights: (B, n_heads, T, T)
                        captured_ref[0] = attn_weights.detach()
                        return attn_output
                    return result
                return attn_wrapper
            
            # Temporarily replace self_attn forward
            layer.self_attn.forward = make_attn_wrapper(captured_attn)
            
            # Forward through layer
            encoded = layer(encoded)
            
            # Restore original forward
            layer.self_attn.forward = original_self_attn
            
            # Store captured attention
            if captured_attn[0] is not None:
                attention_weights.append(captured_attn[0])
        
        # Return encoded output and attention weights
        attention_dict = {
            "encoder_attn": attention_weights  # List of (B, n_heads, T, T) per layer
        }
        
        return encoded, attention_dict

