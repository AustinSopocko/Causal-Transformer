import torch
import torch.nn as nn
from typing import Optional
from .config import CRTConfig


class GRUSeq2Seq(nn.Module):
    """
    GRU-based Sequence-to-Sequence baseline model.
    
    Input: concat(x, a, y) for history
    Encodes history with GRU encoder, decodes future with GRU decoder.
    """
    
    def __init__(self, config: CRTConfig):
        """
        Initialize GRU Seq2Seq model.
        
        Args:
            config: CRTConfig instance
        """
        super().__init__()
        self.config = config
        
        # Input dimension: concatenated (x, a, y)
        d_input = config.d_x + config.d_a + config.d_y
        d_hidden = config.d_model
        
        # GRU Encoder
        self.encoder = nn.GRU(
            input_size=d_input,
            hidden_size=d_hidden,
            num_layers=config.n_layers_enc,
            dropout=config.dropout if config.n_layers_enc > 1 else 0,
            batch_first=True
        )
        
        # GRU Decoder
        self.decoder = nn.GRU(
            input_size=d_input,  # Decoder also takes concat(x, a, y) but only uses a_fut and y_prev
            hidden_size=d_hidden,
            num_layers=config.n_layers_dec,
            dropout=config.dropout if config.n_layers_dec > 1 else 0,
            batch_first=True
        )
        
        # Projection to output
        self.output_proj = nn.Linear(d_hidden, config.d_y)
        
        # Input projection for decoder (to handle a_fut + y_prev)
        self.decoder_input_proj = nn.Linear(config.d_a + config.d_y, d_input)
    
    def forward(
        self,
        x_hist: torch.Tensor,
        a_hist: torch.Tensor,
        y_hist: torch.Tensor,
        a_fut: torch.Tensor,
        y_fut: Optional[torch.Tensor] = None,
        teacher_forcing_prob: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_hist: Historical states (B, T, d_x)
            a_hist: Historical actions (B, T, d_a)
            y_hist: Historical outcomes (B, T, d_y)
            a_fut: Future actions (B, H, d_a)
            y_fut: Optional future outcomes for teacher forcing (B, H, d_y)
            teacher_forcing_prob: Probability of teacher forcing
            
        Returns:
            Predicted outcomes (B, H, d_y)
        """
        B, T, _ = x_hist.shape
        H = a_fut.shape[1]
        device = x_hist.device
        
        # Concatenate history: (B, T, d_x + d_a + d_y)
        hist_input = torch.cat([x_hist, a_hist, y_hist], dim=-1)  # (B, T, d_input)
        
        # Encode history
        encoder_output, hidden = self.encoder(hist_input)  # hidden: (n_layers, B, d_hidden)
        
        # Initialize decoder input with last history step
        y_prev = y_hist[:, -1:, :]  # (B, 1, d_y)
        
        # Decode future
        predictions = []
        decoder_hidden = hidden
        
        for h in range(H):
            # Prepare decoder input: concat(a_fut[h], y_prev)
            a_fut_h = a_fut[:, h:h+1, :]  # (B, 1, d_a)
            decoder_input = torch.cat([a_fut_h, y_prev], dim=-1)  # (B, 1, d_a + d_y)
            decoder_input = self.decoder_input_proj(decoder_input)  # (B, 1, d_input)
            
            # Decode one step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # decoder_output: (B, 1, d_hidden)
            
            # Predict y
            y_pred_h = self.output_proj(decoder_output)  # (B, 1, d_y)
            predictions.append(y_pred_h)
            
            # Update y_prev for next step (teacher forcing or prediction)
            if y_fut is not None and teacher_forcing_prob is not None:
                if torch.rand(1, device=device).item() < teacher_forcing_prob:
                    y_prev = y_fut[:, h:h+1, :]  # Teacher forcing
                else:
                    y_prev = y_pred_h  # Use prediction
            else:
                y_prev = y_pred_h  # Inference: use prediction
        
        # Concatenate predictions: (B, H, d_y)
        y_pred = torch.cat(predictions, dim=1)
        return y_pred


class TCNBaseline(nn.Module):
    """
    Temporal Convolutional Network (TCN) baseline model.
    
    Applies temporal convolutions over history, then MLP to predict H steps.
    """
    
    def __init__(self, config: CRTConfig, num_filters: int = 64, kernel_size: int = 3):
        """
        Initialize TCN model.
        
        Args:
            config: CRTConfig instance
            num_filters: Number of filters in TCN layers
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.config = config
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
        # Input dimension: concatenated (x, a, y)
        d_input = config.d_x + config.d_a + config.d_y
        
        # TCN layers (dilated convolutions)
        tcn_layers = []
        num_levels = config.n_layers_enc
        for i in range(num_levels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            
            tcn_layers.append(nn.Conv1d(
                in_channels=num_filters if i > 0 else d_input,
                out_channels=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            ))
            tcn_layers.append(nn.ReLU())
            tcn_layers.append(nn.Dropout(config.dropout))
        
        self.tcn = nn.Sequential(*tcn_layers)
        
        # Global pooling over time dimension
        # Then MLP to predict H steps
        self.mlp = nn.Sequential(
            nn.Linear(num_filters, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.forecast_horizon * config.d_y)
        )
    
    def forward(
        self,
        x_hist: torch.Tensor,
        a_hist: torch.Tensor,
        y_hist: torch.Tensor,
        a_fut: torch.Tensor,
        y_fut: Optional[torch.Tensor] = None,
        teacher_forcing_prob: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_hist: Historical states (B, T, d_x)
            a_hist: Historical actions (B, T, d_a)
            y_hist: Historical outcomes (B, T, d_y)
            a_fut: Future actions (B, H, d_a) - not used in TCN
            y_fut: Optional future outcomes - not used
            teacher_forcing_prob: Not used in TCN
            
        Returns:
            Predicted outcomes (B, H, d_y)
        """
        # Concatenate history: (B, T, d_x + d_a + d_y)
        hist_input = torch.cat([x_hist, a_hist, y_hist], dim=-1)  # (B, T, d_input)
        
        # TCN expects (B, C, T) format
        hist_input = hist_input.transpose(1, 2)  # (B, d_input, T)
        
        # Apply TCN: (B, num_filters, T)
        tcn_output = self.tcn(hist_input)  # (B, num_filters, T)
        
        # Global average pooling over time: (B, num_filters)
        pooled = torch.mean(tcn_output, dim=2)  # (B, num_filters)
        
        # MLP to predict all H steps: (B, H * d_y)
        flat_predictions = self.mlp(pooled)  # (B, H * d_y)
        
        # Reshape to (B, H, d_y)
        y_pred = flat_predictions.view(-1, self.config.forecast_horizon, self.config.d_y)
        
        return y_pred


class TransformerForecaster(nn.Module):
    """
    Standard Transformer encoder-decoder baseline (without counterfactual conditioning).
    
    Encodes history, decodes future without using a_fut in decoder.
    """
    
    def __init__(self, config: CRTConfig):
        """
        Initialize Transformer Forecaster.
        
        Args:
            config: CRTConfig instance
        """
        super().__init__()
        self.config = config
        
        # Input dimension: concatenated (x, a, y)
        d_input = config.d_x + config.d_a + config.d_y
        
        # Input projection to d_model
        self.input_proj = nn.Linear(d_input, config.d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.randn(1, max(config.history_len, config.forecast_horizon), config.d_model) * 0.02
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers_enc)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers_dec)
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.d_y)
    
    def forward(
        self,
        x_hist: torch.Tensor,
        a_hist: torch.Tensor,
        y_hist: torch.Tensor,
        a_fut: torch.Tensor,
        y_fut: Optional[torch.Tensor] = None,
        teacher_forcing_prob: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_hist: Historical states (B, T, d_x)
            a_hist: Historical actions (B, T, d_a)
            y_hist: Historical outcomes (B, T, d_y)
            a_fut: Future actions (B, H, d_a) - not used in decoder
            y_fut: Optional future outcomes for teacher forcing (B, H, d_y)
            teacher_forcing_prob: Probability of teacher forcing
            
        Returns:
            Predicted outcomes (B, H, d_y)
        """
        B, T, _ = x_hist.shape
        H = self.config.forecast_horizon
        device = x_hist.device
        
        # Concatenate history: (B, T, d_input)
        hist_input = torch.cat([x_hist, a_hist, y_hist], dim=-1)  # (B, T, d_input)
        
        # Project and add positional encoding
        hist_emb = self.input_proj(hist_input)  # (B, T, d_model)
        hist_emb = hist_emb + self.pos_encoder[:, :T, :]  # (B, T, d_model)
        
        # Encode history
        encoded = self.encoder(hist_emb)  # (B, T, d_model)
        
        # Initialize decoder input with last y (repeated for all H steps)
        # Note: This baseline does NOT use a_fut (no counterfactual conditioning)
        y_prev = y_hist[:, -1:, :]  # (B, 1, d_y)
        y_prev = y_prev.repeat(1, H, 1)  # (B, H, d_y)
        
        # Project decoder input (zeros for x and a since we don't condition on future)
        decoder_input = self.input_proj(
            torch.cat([torch.zeros(B, H, self.config.d_x + self.config.d_a, device=device), y_prev], dim=-1)
        )  # (B, H, d_model)
        decoder_input = decoder_input + self.pos_encoder[:, :H, :]  # (B, H, d_model)
        
        # Generate causal mask for decoder (prevents attending to future positions)
        tgt_mask = torch.triu(torch.ones(H, H, dtype=torch.bool, device=device), diagonal=1)
        
        # Decode all steps in parallel (with causal masking)
        decoded = self.decoder(
            tgt=decoder_input,      # (B, H, d_model)
            memory=encoded,         # (B, T, d_model)
            tgt_mask=tgt_mask       # (H, H) - causal mask
        )  # (B, H, d_model)
        
        # Predict outputs
        y_pred = self.output_proj(decoded)  # (B, H, d_y)
        
        return y_pred

