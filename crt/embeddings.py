import torch
import torch.nn as nn


class StateEmbedding(nn.Module):
    """State embedding: Linear -> ReLU -> Linear."""
    
    def __init__(self, d_x: int, d_model: int):
        super().__init__()
        self.linear1 = nn.Linear(d_x, d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ActionEmbedding(nn.Module):
    """Action embedding: Linear -> ReLU -> Linear."""
    
    def __init__(self, d_a: int, d_model: int):
        super().__init__()
        self.linear1 = nn.Linear(d_a, d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
    
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a = self.linear1(a)
        a = self.relu(a)
        a = self.linear2(a)
        return a


class OutcomeEmbedding(nn.Module):
    """Outcome embedding: Linear -> ReLU -> Linear."""
    
    def __init__(self, d_y: int, d_model: int):
        """
        Initialize the outcome embedding module.
        
        Args:
            d_y: Input dimension (outcome feature dimension)
            d_model: Output dimension (model dimension)
        """
        super().__init__()
        self.linear1 = nn.Linear(d_y, d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            y: Input tensor of shape (..., d_y)
            
        Returns:
            Embedded tensor of shape (..., d_model)
        """
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        return y


class PositionalEncoding(nn.Module):
    """
    Positional encoding module that adds positional embeddings to input sequences.
    
    Supports both sinusoidal (fixed) and learned positional encodings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, learned: bool = False):
        """
        Initialize the positional encoding module.
        
        Args:
            d_model: Model dimension (must be even for sinusoidal encoding)
            max_len: Maximum sequence length for positional encoding
            learned: If True, use learned positional embeddings; if False, use sinusoidal
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.learned = learned
        
        if learned:
            # Learned positional embeddings
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pos_embedding, std=0.02)
        else:
            # Sinusoidal positional encoding (fixed, no parameters)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            Tensor of shape (B, T, d_model) with positional encodings added
        """
        if self.learned:
            # Learned positional embeddings
            seq_len = x.size(1)
            pos_emb = self.pos_embedding[:, :seq_len, :]  # (1, T, d_model)
            return x + pos_emb
        else:
            # Sinusoidal positional encoding
            seq_len = x.size(1)
            pos_emb = self.pe[:, :seq_len, :]  # (1, T, d_model)
            return x + pos_emb


def build_history_tokens(
    x_hist: torch.Tensor,
    a_hist: torch.Tensor,
    y_hist: torch.Tensor,
    state_emb: StateEmbedding,
    action_emb: ActionEmbedding,
    outcome_emb: OutcomeEmbedding,
    W_h: nn.Linear
) -> torch.Tensor:
    """
    Build history tokens by concatenating embeddings and applying linear transformation.
    
    Process: concat(emb_x, emb_a, emb_y) -> linear -> add positional encoding
    
    Args:
        x_hist: Historical states of shape (B, T, d_x)
        a_hist: Historical actions of shape (B, T, d_a)
        y_hist: Historical outcomes of shape (B, T, d_y)
        state_emb: State embedding module
        action_emb: Action embedding module
        outcome_emb: Outcome embedding module
        W_h: Linear layer mapping from 3*d_model to d_model
        
    Returns:
        History tokens of shape (B, T, d_model)
    """
    # Embed each component: (B, T, d_model) each
    emb_x = state_emb(x_hist)      # (B, T, d_model)
    emb_a = action_emb(a_hist)     # (B, T, d_model)
    emb_y = outcome_emb(y_hist)    # (B, T, d_model)
    
    # Concatenate embeddings: (B, T, 3*d_model)
    concat_emb = torch.cat([emb_x, emb_a, emb_y], dim=-1)  # (B, T, 3*d_model)
    
    # Apply linear transformation: (B, T, d_model)
    tokens = W_h(concat_emb)  # (B, T, d_model)
    
    # Add positional encoding: (B, T, d_model)
    # Sinusoidal positional encoding
    B, T, d_model = tokens.shape
    position = torch.arange(0, T, dtype=torch.float, device=tokens.device).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=tokens.device) * 
                        (-torch.log(torch.tensor(10000.0, device=tokens.device)) / d_model))  # (d_model//2,)
    pe = torch.zeros(1, T, d_model, device=tokens.device)  # (1, T, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)  # (T, d_model//2)
    pe[0, :, 1::2] = torch.cos(position * div_term)  # (T, d_model//2)
    tokens = tokens + pe  # (B, T, d_model)
    
    return tokens


def build_future_tokens(
    a_fut: torch.Tensor,
    y_prev: torch.Tensor,
    action_emb: ActionEmbedding,
    outcome_emb: OutcomeEmbedding,
    W_u: nn.Linear
) -> torch.Tensor:
    """
    Build future tokens by concatenating embeddings and applying linear transformation.
    
    Process: concat(emb_a, emb_y) -> linear -> add positional encoding
    
    Args:
        a_fut: Future actions of shape (B, H, d_a)
        y_prev: Previous outcomes of shape (B, H, d_y)
        action_emb: Action embedding module
        outcome_emb: Outcome embedding module
        W_u: Linear layer mapping from 2*d_model to d_model
        
    Returns:
        Future tokens of shape (B, H, d_model)
    """
    # Embed each component: (B, H, d_model) each
    emb_a = action_emb(a_fut)      # (B, H, d_model)
    emb_y = outcome_emb(y_prev)    # (B, H, d_model)
    
    # Concatenate embeddings: (B, H, 2*d_model)
    concat_emb = torch.cat([emb_a, emb_y], dim=-1)  # (B, H, 2*d_model)
    
    # Apply linear transformation: (B, H, d_model)
    tokens = W_u(concat_emb)  # (B, H, d_model)
    
    # Add positional encoding: (B, H, d_model)
    # Sinusoidal positional encoding
    B, H, d_model = tokens.shape
    position = torch.arange(0, H, dtype=torch.float, device=tokens.device).unsqueeze(1)  # (H, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=tokens.device) * 
                        (-torch.log(torch.tensor(10000.0, device=tokens.device)) / d_model))  # (d_model//2,)
    pe = torch.zeros(1, H, d_model, device=tokens.device)  # (1, H, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)  # (H, d_model//2)
    pe[0, :, 1::2] = torch.cos(position * div_term)  # (H, d_model//2)
    tokens = tokens + pe  # (B, H, d_model)
    
    return tokens

