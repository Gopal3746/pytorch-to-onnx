from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_shape: Tuple[int, int, int]
    num_classes: int


MODEL_SPECS = {
    "resnet18": ModelSpec("resnet18", (3, 224, 224), 1000),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}. Available: {sorted(MODEL_SPECS)}") from exc


def load_model(name: str, pretrained: bool = True):
    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name}")

    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.eval()
    return model


def make_cpu_input(batch_size: int, spec: ModelSpec, seed: int):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + batch_size)
    return torch.randn((batch_size, *spec.input_shape), generator=generator, dtype=torch.float32)
