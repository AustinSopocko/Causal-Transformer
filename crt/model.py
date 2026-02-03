import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple, Dict, Union
from .config import CRTConfig
from .embeddings import (
    StateEmbedding,
    ActionEmbedding,
    OutcomeEmbedding,
    build_history_tokens,
    build_future_tokens
)
from .encoder import HistoryEncoder
from .decoder import CounterfactualDecoder


class PredictionHead(nn.Module):
    """Prediction head for model outputs. Currently supports regression."""
    
    def __init__(
        self,
        d_model: int,
        d_y: int,
        task_type: Literal["regression", "gaussian", "classification"] = "regression"
    ):
        """Initialize prediction head."""
        super().__init__()
        self.d_model = d_model
        self.d_y = d_y
        self.task_type = task_type
        
        if task_type == "regression":
            self.head = nn.Linear(d_model, d_y)
        elif task_type == "gaussian":
            raise NotImplementedError("Gaussian prediction head not yet implemented")
        elif task_type == "classification":
            raise NotImplementedError("Classification prediction head not yet implemented")
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Must be 'regression', 'gaussian', or 'classification'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns predictions of shape (..., d_y) for regression."""
        if self.task_type == "regression":
            return self.head(x)
        else:
            raise NotImplementedError(f"Forward pass for {self.task_type} not yet implemented")


class CRTModel(nn.Module):
    """Causal Rollout Transformer for counterfactual time series prediction."""
    
    def __init__(self, config: CRTConfig):
        """Initialize CRT model with given config."""
        super().__init__()
        self.config = config
        
        self.state_emb = StateEmbedding(config.d_x, config.d_model)
        self.action_emb = ActionEmbedding(config.d_a, config.d_model)
        self.outcome_emb = OutcomeEmbedding(config.d_y, config.d_model)
        
        self.W_h = nn.Linear(3 * config.d_model, config.d_model)
        self.W_u = nn.Linear(2 * config.d_model, config.d_model)
        self.encoder = HistoryEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers_enc,
            dropout=config.dropout
        )
        self.decoder = CounterfactualDecoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers_dec,
            dropout=config.dropout
        )
        
        self.pred_head = PredictionHead(config.d_model, config.d_y, task_type="regression")
    
    def forward(
        self,
        x_hist: torch.Tensor,
        a_hist: torch.Tensor,
        y_hist: torch.Tensor,
        a_fut: torch.Tensor,
        y_fut: Optional[torch.Tensor] = None,
        teacher_forcing_prob: Optional[float] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        
        Training: y_fut provided, uses teacher forcing/scheduled sampling.
        Inference: y_fut=None, performs autoregressive rollout.
        
        Returns predictions (B, H, d_y) or (predictions, attention_dict) if return_attention=True.
        """
        B, T, _ = x_hist.shape
        H = a_fut.shape[1]
        device = x_hist.device
        
        is_training = y_fut is not None
        if teacher_forcing_prob is None:
            teacher_forcing_prob = self.config.teacher_forcing_start if is_training else 0.0
        
        h_tokens = build_history_tokens(
            x_hist, a_hist, y_hist,
            self.state_emb, self.action_emb, self.outcome_emb,
            self.W_h
        )
        
        if return_attention:
            E_t, encoder_attn_dict = self.encoder(h_tokens, return_attention=True)
        else:
            E_t = self.encoder(h_tokens)
            encoder_attn_dict = None
        
        y_last = y_hist[:, -1:, :]
        
        if is_training:
            y_prev = y_last.repeat(1, H, 1)
            
            if teacher_forcing_prob >= 1.0:
                y_prev = y_fut
            else:
                predictions = []
                y_prev_list = [y_last]
                
                for h in range(H):
                    y_prev_so_far = torch.cat(y_prev_list, dim=1)
                    a_fut_so_far = a_fut[:, :h+1, :]
                    
                    u_tilde = build_future_tokens(
                        a_fut_so_far, y_prev_so_far,
                        self.action_emb, self.outcome_emb,
                        self.W_u
                    )
                    
                    if return_attention:
                        D, decoder_attn_dict = self.decoder(u_tilde, E_t, return_attention=True)
                    else:
                        D = self.decoder(u_tilde, E_t)
                    
                    y_pred_h = self.pred_head(D[:, -1:, :])
                    predictions.append(y_pred_h)
                    
                    if h + 1 < H:
                        if torch.rand(1, device=device).item() < teacher_forcing_prob:
                            y_prev_list.append(y_fut[:, h:h+1, :])
                        else:
                            y_prev_list.append(y_pred_h)
                
                y_pred = torch.cat(predictions, dim=1)
                
                if return_attention:
                    attention_dict = {
                        "encoder_attn": encoder_attn_dict["encoder_attn"] if encoder_attn_dict else [],
                        "decoder_self_attn": decoder_attn_dict["self_attn"] if return_attention else [],
                        "decoder_cross_attn": decoder_attn_dict["cross_attn"] if return_attention else []
                    }
                    return y_pred, attention_dict
                return y_pred
            
            u_tilde = build_future_tokens(
                a_fut, y_prev,
                self.action_emb, self.outcome_emb,
                self.W_u
            )
            
            if return_attention:
                D, decoder_attn_dict = self.decoder(u_tilde, E_t, return_attention=True)
            else:
                D = self.decoder(u_tilde, E_t)
            
            y_pred = self.pred_head(D)
            
            if return_attention:
                attention_dict = {
                    "encoder_attn": encoder_attn_dict["encoder_attn"] if encoder_attn_dict else [],
                    "decoder_self_attn": decoder_attn_dict["self_attn"],
                    "decoder_cross_attn": decoder_attn_dict["cross_attn"]
                }
                return y_pred, attention_dict
            
        else:
            predictions = []
            y_prev_list = [y_last]
            
            for h in range(H):
                y_prev_so_far = torch.cat(y_prev_list, dim=1)
                a_fut_so_far = a_fut[:, :h+1, :]
                
                u_tilde = build_future_tokens(
                    a_fut_so_far, y_prev_so_far,
                    self.action_emb, self.outcome_emb,
                    self.W_u
                )
                
                if return_attention:
                    D, decoder_attn_dict = self.decoder(u_tilde, E_t, return_attention=True)
                else:
                    D = self.decoder(u_tilde, E_t)
                
                y_pred_h = self.pred_head(D[:, -1:, :])
                predictions.append(y_pred_h)
                
                if h + 1 < H:
                    y_prev_list.append(y_pred_h)
            
            y_pred = torch.cat(predictions, dim=1)
            
            if return_attention:
                attention_dict = {
                    "encoder_attn": encoder_attn_dict["encoder_attn"] if encoder_attn_dict else [],
                    "decoder_self_attn": decoder_attn_dict["self_attn"] if return_attention else [],
                    "decoder_cross_attn": decoder_attn_dict["cross_attn"] if return_attention else []
                }
                return y_pred, attention_dict
        
        if return_attention:
            attention_dict = {
                "encoder_attn": encoder_attn_dict["encoder_attn"] if encoder_attn_dict else [],
                "decoder_self_attn": decoder_attn_dict["self_attn"] if 'decoder_attn_dict' in locals() else [],
                "decoder_cross_attn": decoder_attn_dict["cross_attn"] if 'decoder_attn_dict' in locals() else []
            }
            return y_pred, attention_dict
        
        return y_pred

