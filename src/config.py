from dataclasses import dataclass
from typing import List
from hydra.core.config_store import ConfigStore


@dataclass
class Paths:
    input_filepath: str
    figures_filepath: str
    model_filepath: str


@dataclass
class Image:
    size: int
    channels: int


@dataclass
class MaxPool:
    kernel_size: int
    stride: int
    padding: int


@dataclass
class ConvLayer(MaxPool):
    out_channels: int


@dataclass
class Model:
    lr: float
    batch_size: int
    dropout: float
    classes: int

@dataclass
class LinearLayer:
    output: int

@dataclass
class DOGCATConfig:
    image: Image
    model: Model
    conv_layers: List[ConvLayer]
    maxpool: MaxPool
    paths: Paths
    linear_layer: LinearLayer



def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="dog_cat_config", node=DOGCATConfig)
