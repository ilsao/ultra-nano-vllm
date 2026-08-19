import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """ 
        Samples tokens, using greedy decoding where temperature is zero.
        
        param:
            logits (torch.Tensor): The logits from which to sample tokens.
            temperatures (torch.Tensor): The temperature values for sampling.
            
        return:
            torch.Tensor: The sampled token indices.
        """
        greedy = temperatures == 0
        safe_temperatures = torch.where(
            greedy,
            torch.ones_like(temperatures),
            temperatures,
        )
        scaled_logits = logits.float().div(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(scaled_logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy, greedy_tokens, sample_tokens)
