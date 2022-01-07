import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.contrib as K
import hydra
from hydra.core.config_store import ConfigStore
from src.config import DOGCATConfig
from torchsummary import summary

cs = ConfigStore().instance()
cs.store(name='dog_cat_config', node = DOGCATConfig)

def compute_conv_dim(dim_size, kernel_size_conv, padding_conv, stride_conv):
    return int((dim_size - kernel_size_conv + 2 * padding_conv) / stride_conv + 1)

class Classifier(nn.Module):
    def __init__(self, cfg: DOGCATConfig):
        super().__init__()
        self.layers = []
        hw = []
        
        for idx, conv_layer in enumerate(cfg.conv_layers):
            if idx == 0:
                self.layers.append(
                    nn.Conv2d(
                        in_channels=cfg.image.channels,
                        out_channels=conv_layer.out_channels,
                        kernel_size=conv_layer.kernel_size,
                        stride=conv_layer.stride,
                        padding=conv_layer.padding
                    )
                )
                h = compute_conv_dim(cfg.image.height, cfg.maxpool.kernel_size, cfg.maxpool.padding, cfg.maxpool.stride)
                w = compute_conv_dim(cfg.image.width, cfg.maxpool.kernel_size, cfg.maxpool.padding, cfg.maxpool.stride)
                hw.append([h,w])
            else:
                self.layers.append(
                        nn.Conv2d(
                        in_channels=cfg.conv_layers[idx - 1].out_channels,
                        out_channels=conv_layer.out_channels,
                        kernel_size=conv_layer.kernel_size,
                        stride=conv_layer.stride,
                        padding=conv_layer.padding
                    )
                )
                h = compute_conv_dim(hw[idx - 1][0], cfg.maxpool.kernel_size, cfg.maxpool.padding, cfg.maxpool.stride)
                w = compute_conv_dim(hw[idx - 1][1], cfg.maxpool.kernel_size, cfg.maxpool.padding, cfg.maxpool.stride)
                hw.append([h,w])
        
        self.l1_in_features = cfg.conv_layers[-1].out_channels * hw[-1][0] * hw[-1][1]

        self.maxpool = nn.MaxPool2d(cfg.maxpool.kernel_size, cfg.maxpool.stride, padding=cfg.maxpool.padding)

        self.out = nn.Linear(in_features=self.l1_in_features, 
                    out_features=cfg.model.classes,
                    bias=True)

        self.dropout = nn.Dropout(p = cfg.model.dropout)

    def forward(self, x):
        for layer in self.layers:
            x = self.maxpool(F.relu(layer(x)))

        x = x.view(-1, self.l1_in_features)
        
        return F.softmax(self.out(x), dim=1)

@hydra.main(config_path='../conf', config_name='config')
def summary_model(cfg: DOGCATConfig):
    m = Classifier(cfg)
    print(summary(m, (3,416,416)))

if __name__ == '__main__':
    summary_model()