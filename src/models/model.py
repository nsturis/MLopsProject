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
        self.cfg = cfg
        for idx, conv_layer in enumerate(self.cfg.conv_layers):
            if idx == 0:
                self.layers.append(
                    nn.Conv2d(
                        in_channels=self.cfg.image.channels,
                        out_channels=conv_layer.out_channels,
                        kernel_size=conv_layer.kernel_size,
                        stride=conv_layer.stride,
                        padding=conv_layer.padding
                    )
                )
                h = compute_conv_dim(self.cfg.image.height, self.cfg.maxpool.kernel_size, self.cfg.maxpool.padding, self.cfg.maxpool.stride)
                w = compute_conv_dim(self.cfg.image.width, self.cfg.maxpool.kernel_size, self.cfg.maxpool.padding, self.cfg.maxpool.stride)
                hw.append([h,w])
            else:
                self.layers.append(
                        nn.Conv2d(
                        in_channels=self.cfg.conv_layers[idx - 1].out_channels,
                        out_channels=conv_layer.out_channels,
                        kernel_size=conv_layer.kernel_size,
                        stride=conv_layer.stride,
                        padding=conv_layer.padding
                    )
                )
                h = compute_conv_dim(hw[idx - 1][0], self.cfg.maxpool.kernel_size, self.cfg.maxpool.padding, self.cfg.maxpool.stride)
                w = compute_conv_dim(hw[idx - 1][1], self.cfg.maxpool.kernel_size, self.cfg.maxpool.padding, self.cfg.maxpool.stride)
                hw.append([h,w])
        
        self.l1_in_features = self.cfg.conv_layers[-1].out_channels * hw[-1][0] * hw[-1][1]

        self.maxpool = nn.MaxPool2d(self.cfg.maxpool.kernel_size, self.cfg.maxpool.stride, padding=self.cfg.maxpool.padding)

        self.out = nn.Linear(in_features=self.l1_in_features, 
                    out_features=self.cfg.model.classes,
                    bias=True)

        self.dropout = nn.Dropout(p = self.cfg.model.dropout)

    def forward(self, x):
        for layer in self.layers:
            x = self.maxpool(F.relu(layer(x)))

        x = x.view(-1, self.l1_in_features)
        
        return F.softmax(self.out(x), dim=1)
    
    def loss(self, X, y):
        return torch.nn.CrossEntropyLoss(X, y)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters, lr=self.cfg.model.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        logits = self.forward(X)
        loss = self.loss(logits,y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss(logits,y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss(logits,y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)
        return loss

@hydra.main(config_path='../conf', config_name='config')
def summary_model(cfg: DOGCATConfig):
    m = Classifier(cfg)
    print(summary(m, (3,416,416)))

if __name__ == '__main__':
    summary_model()