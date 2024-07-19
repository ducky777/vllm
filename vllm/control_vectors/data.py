from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ControlVector:
    name: str
    directions: dict[str, list[float]]


@dataclass
class ControlVectorData:
    name: Optional[str] = field(default=None)
    layers: Optional[list[int]] = field(default=None)
    strength: float = field(default=1.0)
    save_hidden_states: bool = field(default=False)
    intervene: bool = field(default=False)
    intervention_scale: float = field(default=1.0)
    intervention_type: Literal["pos", "full"] = field(default="full")
