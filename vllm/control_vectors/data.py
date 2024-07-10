from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ControlVector:
    name: str
    directions: dict[str, list[float]]


@dataclass
class ControlVectorData:
    name: Optional[str] = field(default = None)
    layers: Optional[list[int]] = field(default = None)
    strength: float = field(default = 1.0)
    save_hidden_states: bool = field(default = False)
