from .interventions import InterventionLayer


class InterventionData:
    _interventions: dict[tuple[int, str], InterventionLayer] = {}
    _layers_to_intervene: list[int] = []

    def __init__(self):
        pass

    def __getitem__(self, i: tuple[int, str]) -> InterventionLayer:
        return self._interventions[i]

    def get_available_interventions(self) -> list[tuple[int, str]]:
        """returns a list of available interventions in the tuple format (layer_num, name)."""
        return self._interventions.keys()
        

    def get_intervention(self, layer_num: int, name: str) -> InterventionLayer:
        if (layer_num, name) in self._interventions:
            return self._interventions[(layer_num, name)]
        raise ValueError(f"No valid intervention for ({layer_num}, {name})")

    def append(self, layer_num: int, name: str, intervention: InterventionLayer):
        """Adds a new intervention"""
        if (layer_num, name) not in self._interventions:
            self._interventions[(layer_num, name)] = intervention
            print(f"Added Intervention: ({layer_num}, {name})")
        else:
            print(f"Intervention ({layer_num}, {name}) already exists.")
        
        if layer_num not in self._layers_to_intervene:
            self._layers_to_intervene.append(layer_num)

    @property
    def intervenable_layers(self):
        return self._layers_to_intervene
    
    @property
    def intervenable_tuples(self) -> tuple[int, str]:
        return self._interventions.keys()
