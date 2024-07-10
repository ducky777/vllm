import dataclasses
import json
import typing
import warnings
from typing import List, Optional, Union

import gguf
import numpy as np
import torch
import tqdm
from transformers import (
    LlamaConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from vllm.attention import AttentionMetadata
from vllm.sequence import SamplerOutput


@dataclasses.dataclass
class ControlVectorConfig(LlamaConfig):
    control_vector_file: str


@dataclasses.dataclass
class DatasetEntry:
    positive: str
    negative: str


@dataclasses.dataclass
class ControlVector:
    name: str
    model_type: str
    directions: dict[int, np.ndarray]

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: Union[int, float, np.int_, np.float_]) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: Union[int, float, np.int_, np.float_]) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(
        self, other: Union[int, float, np.int_, np.float_]
    ) -> "ControlVector":
        return self.__mul__(1 / other)


def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            out = model(
                **tokenizer(batch, padding=True, return_tensors="pt").to(model.device),
                output_hidden_states=True,
            )
            for layer in hidden_layers:
                # if not indexing from end, account for embedding hiddens
                hidden_idx = layer + 1 if layer >= 0 else layer
                for batch in out.hidden_states[hidden_idx]:
                    hidden_states[layer].append(batch[-1, :].squeeze().cpu().numpy())
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag


class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(
        self, model, layer_ids: typing.Iterable[int]
    ):
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new ControlModel around a model instance, initializing control on
        the layers specified in `layer_ids`.
        """

        super().__init__()

        self.model = model

        layers = model_layer_list(model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]
        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, ControlModule):
                layers[layer_id] = ControlModule(layer)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        # return self.model.device
        return "cuda:0"

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
        After using this method, `set_control` and `reset` will not work.
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layers[layer_id] = layers[layer_id].block
        return self.model

    def set_control(
        self, control: "ControlVector", coeff: float = 1.0, **kwargs
    ) -> None:
        """
        Set a `ControlVector` for the layers this ControlModel handles, with a strength given
        by `coeff`. (Negative `coeff` values invert the control vector, e.g. happiness→sadness.)
        `coeff` defaults to `1.0`.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        raw_control = {}
        for layer_id in self.layer_ids:
            raw_control[layer_id] = (
                torch.tensor(coeff * control.directions[layer_id])
                .to("cuda:0")
                .half()
            )
        self.set_raw_control(raw_control, **kwargs)

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_raw_control(self, control: dict[int, torch.Tensor], **kwargs) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        len_layers = len(layers)
        for layer_id in self.layer_ids:
            layer: ControlModule = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> Optional[SamplerOutput]:
        self.set_control(self.cvec, 1.9)
        return self.model.forward(input_ids, positions, kv_caches, attn_metadata)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@dataclasses.dataclass
class BlockControlParams:
    control: torch.Tensor = None
    normalize: bool = False
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()


class ControlModule(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        self.params: BlockControlParams = BlockControlParams()

    def set_control(self, params: BlockControlParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(BlockControlParams.default())

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output
        # elif len(control.shape) == 1:
        control = control.reshape(1, -1)

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        op1_shape = output[0].shape
        op2_shape = output[1].shape
        ctrl_shape = control.shape
        mod_shape = modified.shape
        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)

        norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # we should ignore the padding tokens when doing the activation addition
        # mask has ones for non padding tokens and zeros at padding tokens.
        # only tested this on left padding
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            target_shape = modified.shape
            mask = (
                (col_indices >= zero_indices)
                .float()
                .reshape(target_shape[0], target_shape[1], 1)
            )
            mask = mask.to(modified.dtype).to(modified.device)
        else:
            mask = 1.0

        modified = self.params.operator(modified, control * mask)

        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output


def model_layer_list(
    model: Union[ControlModel, PreTrainedModel]
) -> torch.nn.ModuleList:
    if isinstance(model, ControlModel):
        model = model.model

    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
