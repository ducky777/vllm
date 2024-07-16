from typing import Any
import torch
import torch.nn as nn


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, rotate_weights: torch.Tensor):
        super().__init__()
        self.weight = rotate_weights.to("cuda").to(torch.half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.to(x.device).to(x.dtype))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @property
    def device(self):
        return self.weight.device


class InterventionLayer:

    @staticmethod
    def load_layer(cfg: str, data: dict):
        intervention_type = cfg["intervention_type"]

        if intervention_type.lower() == "loreft":
            return LoreftIntervention(**data)
        elif intervention_type.lower() == "consreft":
            return ConsreftIntervention(
                data["rotate_layer.parametrizations.weight.original"],
                data["learned_source"],
            )
        elif intervention_type.lower() == "nodireft":
            return NodireftIntervention(
                data["embed_dim"],
                data["interchange_dim"],
                cfg["low_rank_dim"],
                data["proj_layer.weight"],
                data["proj_layer.bias"],
                data["learned_source.weight"],
                data["learned_source.bias"],
            )

        raise ValueError(f"Unsupported Intervention type {intervention_type}")


class ConsreftIntervention:
    def __init__(self, rotate_weights: torch.Tensor, learned_source: torch.Tensor):
        self.rotate_layer = LowRankRotateLayer(rotate_weights)
        self.learned_source = learned_source.to("cuda:0").to(torch.half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.half)
        rotated_base = self.rotate_layer(x)
        output = x + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.t()
        )
        return output.to(x.dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class LoreftIntervention:
    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor, rotate_layer: torch.Tensor
    ):
        self.rotate_layer = LowRankRotateLayer(rotate_layer)

        self.learned_source = torch.nn.Linear(4096, 4, dtype=torch.half)
        self.learned_source.weight.data = weight
        self.learned_source.bias.data = bias
        self.learned_source = self.learned_source.to("cuda").to(torch.half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.half)
        rotated_base = self.rotate_layer(x)
        output = x + torch.matmul(
            (self.learned_source(x) - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(torch.half)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class NodireftIntervention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        interchange_dim: int,
        low_rank_dim: int,
        proj_layer_weight: torch.Tensor,
        proj_layer_bias: torch.Tensor,
        learned_source_weights: torch.Tensor,
        learned_source_bias: torch.Tensor,
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.interchange_dim = interchange_dim

        self.proj_layer = nn.Linear(self.embed_dim, low_rank_dim, bias=True).to(
            torch.half
        )
        self.learned_source = torch.nn.Linear(self.embed_dim, low_rank_dim).to(
            torch.half
        )

        # self.proj_weight = proj_layer_weight.to(torch.half)
        # self.proj_bias = proj_layer_bias.to(torch.half)

        # # Load weights and biases
        # self.proj_layer.weight.data = proj_layer_weight.to(torch.half)
        # self.proj_layer.bias.data = proj_layer_bias.to(torch.half)
        # self.learned_source.weight.data = learned_source_weights.to(torch.half)
        # self.learned_source.bias.data = learned_source_bias.to(torch.half)

        # Set requires_grad to False for all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NodiReFT(h) = h + proj_layer(learned_source(h))
        with torch.no_grad():
            output = x + (torch.matmul(self.learned_source(x), self.proj_layer.weight))
        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
