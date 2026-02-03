import torch
from typing import TYPE_CHECKING, Tuple, Dict, Union

if TYPE_CHECKING:
    from .model import CRTModel


def rollout(
    model: "CRTModel",
    x_hist: torch.Tensor,
    a_hist: torch.Tensor,
    y_hist: torch.Tensor,
    a_fut: torch.Tensor,
    return_attention: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Perform autoregressive rollout inference. Returns predictions (B, H, d_y)."""
    model.eval()
    with torch.no_grad():
        result = model(
            x_hist=x_hist,
            a_hist=a_hist,
            y_hist=y_hist,
            a_fut=a_fut,
            y_fut=None,
            return_attention=return_attention
        )
    return result

