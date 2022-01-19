from src.models.model import *
import hydra
from hydra.core.config_store import ConfigStore
from src.config import *
import yaml
import pytest

def test_model_structure():
    with open('src/conf/config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    img = Image(
        height = parameters["image"]["height"],
        width = parameters["image"]["width"],
        channels = parameters["image"]["channels"]
    )
    pool = MaxPool(
        kernel_size = parameters["maxpool"]["kernel_size"],
        stride = parameters["maxpool"]["stride"],
        padding = parameters["maxpool"]["padding"]
    )
    conv_layers = []
    for layer in parameters["conv_layers"]:
        conv_layers.append(ConvLayer(
            out_channels = layer["out_channels"],
            stride = layer["stride"], 
            padding = layer["padding"], 
            kernel_size =layer["kernel_size"]))
    
    model = Model(
        lr = parameters["model"]["lr"],
        batch_size = parameters["model"]["batch_size"],
        dropout = parameters["model"]["dropout"],
        classes = parameters["model"]["classes"]
    )

    cfg = DOGCATConfig(
        image=img, 
        model=model, 
        conv_layers=conv_layers, 
        maxpool=pool)

    m = Classifier(cfg)


    assert parameters["image"]["channels"] == m.layers[0].in_channels
    assert parameters["maxpool"]["kernel_size"] == m.maxpool.kernel_size
    
    for i in range(len(m.layers)-1):
        assert m.layers[i].out_channels == m.layers[i+1].in_channels

    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        m(torch.randn(1,2,3))



