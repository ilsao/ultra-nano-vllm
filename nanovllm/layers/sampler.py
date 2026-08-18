import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """ 
        Samples tokens from the given logits based on the provided temperatures.
        
        param:
            logits (torch.Tensor): The logits from which to sample tokens.
            temperatures (torch.Tensor): The temperature values for sampling.
            
        return:
            torch.Tensor: The sampled token indices.
        """
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
