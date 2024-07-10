import torch

class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, rotate_weights: torch.Tensor):
        super().__init__()
        self.weight = rotate_weights
        self.weight.to("cuda")

    def forward(self, x):
        return torch.matmul(x, self.weight)

class ConsreftIntervention:
    def __init__(self, rotate_weights: torch.Tensor, learned_source: torch.Tensor):
        self.rotate_layer = LowRankRotateLayer(rotate_weights)
        self.learned_source = learned_source
        self.learned_source.to("cuda")
        
    def forward(self, x:torch.Tensor):
        rotated_base = self.rotate_layer(x)
        output = x + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(x.dtype)
        