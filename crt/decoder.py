import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union


def generate_square_subsequent_mask(L: int) -> torch.Tensor:
    """
    Generate a square subsequent mask for causal (masked) self-attention.
    
    The mask prevents positions from attending to future positions.
    True values are masked (set to -inf), False values are allowed.
    
    Args:
        L: Sequence length
        
    Returns:
        Boolean mask tensor of shape (L, L) where mask[i, j] = True if j > i
    """
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    return mask


class CounterfactualDecoder(nn.Module):
    """
    Decoder module for counterfactual prediction using Transformer architecture.
    
    Performs masked self-attention over future tokens and cross-attention to encoded history.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        """
        Initialize the counterfactual decoder.
        
        Args:
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Create decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,  # Standard feedforward dimension
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # Create transformer decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers
        )
    
    def forward(
        self,
        u_tilde: torch.Tensor,
        E_t: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the decoder.
        
        Performs:
        - Masked self-attention over future tokens (u_tilde) with causal mask
        - Cross-attention from future tokens to encoded history (E_t)
        
        Args:
            u_tilde: Future tokens of shape (B, H, d_model)
                     - B: Batch size
                     - H: Forecast horizon (future sequence length)
                     - d_model: Model dimension
            E_t: Encoded history of shape (B, T, d_model)
                 - B: Batch size
                 - T: History length
                 - d_model: Model dimension
            return_attention: If True, returns attention weights for visualization
                 
        Returns:
            If return_attention=False:
                Decoded tensor D of shape (B, H, d_model)
            If return_attention=True:
                Tuple of (decoded_tensor, attention_dict) where:
                - decoded_tensor: (B, H, d_model)
                - attention_dict: Dictionary with keys:
                  - "self_attn": List of self-attention weights per layer (B, n_heads, H, H)
                  - "cross_attn": List of cross-attention weights per layer (B, n_heads, H, T)
        """
        B, H, d_model = u_tilde.shape
        
        # Generate causal mask for masked self-attention
        # Prevents positions from attending to future positions
        tgt_mask = generate_square_subsequent_mask(H).to(u_tilde.device)  # (H, H)
        
        if not return_attention:
            # Standard forward pass
            D = self.decoder(
                tgt=u_tilde,      # (B, H, d_model) - future tokens
                memory=E_t,       # (B, T, d_model) - encoded history
                tgt_mask=tgt_mask # (H, H) - causal mask
            )  # (B, H, d_model)
            return D
        
        # Extract attention weights by patching attention modules
        self_attn_weights = []
        cross_attn_weights = []
        
        # Store original forwards and create wrappers
        original_forwards_self = []
        original_forwards_cross = []
        
        for i, layer in enumerate(self.decoder.layers):
            # Store original forwards
            original_forwards_self.append(layer.self_attn.forward)
            original_forwards_cross.append(layer.multihead_attn.forward)
            
            # Create wrappers to capture attention
            def make_self_attn_wrapper(layer_idx, captured_list):
                original = original_forwards_self[layer_idx]
                def self_attn_wrapper(query, key, value, key_padding_mask=None, need_weights=True,
                                     attn_mask=None, average_attn_weights=False, is_causal=False):
                    result = original(
                        query, key, value,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                        attn_mask=attn_mask,
                        average_attn_weights=False,
                        is_causal=is_causal
                    )
                    if isinstance(result, tuple):
                        attn_output, attn_weights = result
                        captured_list.append(attn_weights.detach())
                        return attn_output
                    return result
                return self_attn_wrapper
            
            def make_cross_attn_wrapper(layer_idx, captured_list):
                original = original_forwards_cross[layer_idx]
                def cross_attn_wrapper(query, key, value, key_padding_mask=None, need_weights=True,
                                      attn_mask=None, average_attn_weights=False, is_causal=False):
                    result = original(
                        query, key, value,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                        attn_mask=attn_mask,
                        average_attn_weights=False,
                        is_causal=is_causal
                    )
                    if isinstance(result, tuple):
                        attn_output, attn_weights = result
                        captured_list.append(attn_weights.detach())
                        return attn_output
                    return result
                return cross_attn_wrapper
            
            # Temporarily replace attention forwards
            layer.self_attn.forward = make_self_attn_wrapper(i, self_attn_weights)
            layer.multihead_attn.forward = make_cross_attn_wrapper(i, cross_attn_weights)
        
        # Forward through decoder
        decoded = self.decoder(
            tgt=u_tilde,
            memory=E_t,
            tgt_mask=tgt_mask
        )
        
        # Restore original forwards
        for i, layer in enumerate(self.decoder.layers):
            layer.self_attn.forward = original_forwards_self[i]
            layer.multihead_attn.forward = original_forwards_cross[i]
        
        # Return decoded output and attention weights
        attention_dict = {
            "self_attn": self_attn_weights,      # List of (B, n_heads, H, H) per layer
            "cross_attn": cross_attn_weights     # List of (B, n_heads, H, T) per layer
        }
        
        return decoded, attention_dict

