from dataclasses import dataclass

@dataclass
class Image:
    height: int
    width: int
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
class DOGCATConfig:
    image: Image
    model: Model
    conv_layers: list[ConvLayer]
    maxpool: MaxPool
