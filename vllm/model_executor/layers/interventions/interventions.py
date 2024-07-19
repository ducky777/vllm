from typing import Union

import torch
import torch.nn as nn


class LowRankRotateLayer(nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, rotate_weights: torch.Tensor):
        super().__init__()
        self.weight = rotate_weights.to("cuda").to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @property
    def device(self):
        return self.weight.device


class BaseIntervention(nn.Module):
    def forward(self, _: torch.Tensor) -> torch.Tensor:
        """Requires forward method"""

        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class ConsreftIntervention(BaseIntervention):
    def __init__(self, rotate_weights: torch.Tensor, learned_source: torch.Tensor):
        super().__init__()
        self.rotate_layer = LowRankRotateLayer(rotate_weights)
        self.learned_source = learned_source.to("cuda:0").to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotated_base = self.rotate_layer(x)
        output = x + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.t()
        )
        return output.to(x.dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class LoreftIntervention(BaseIntervention):
    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor, rotate_layer: torch.Tensor
    ):
        super().__init__()
        self.rotate_layer = LowRankRotateLayer(rotate_layer)

        self.learned_source = torch.nn.Linear(4096, 4, dtype=torch.bfloat16)
        self.learned_source.weight.data = weight.to(torch.bfloat16)
        self.learned_source.bias.data = bias.to(torch.bfloat16)
        self.learned_source = self.learned_source.to("cuda").to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotated_base = self.rotate_layer(x)
        output = x + torch.matmul(
            (self.learned_source(x) - rotated_base), self.rotate_layer.weight.T
        )
        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class NodireftIntervention(BaseIntervention):
    """
    NodiReFT(h) = h + proj_layer(learned_source(h))
    """
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

        self.proj_layer = nn.Linear(self.embed_dim, low_rank_dim, bias=True).to(torch.bfloat16)
        self.learned_source = torch.nn.Linear(self.embed_dim, low_rank_dim).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = x + (torch.matmul(self.learned_source(x), self.proj_layer.weight))
        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class InterventionLayer:

    intervention: BaseIntervention

    def __init__(self, cfg: dict, data: dict):
        self.intervention = self.load_layer(cfg, data)

    @classmethod
    def load_layer(cls, cfg: dict, data: dict):
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

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        output = self.intervention.forward(x.to(torch.bfloat16))
        x = x + (output * scale)
        return x.to(torch.float16)

    def __call__(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return self.forward(x, scale)
