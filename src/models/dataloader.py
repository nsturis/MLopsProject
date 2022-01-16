import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import kornia.augmentation as K
from kornia.utils import image_to_tensor
import numpy as np
from PIL import Image
import json

class DogCatDataset(Dataset):
    def __init__(self, data_split, data_dir, image_size, transform):
        with open(data_dir + "datasplit.json", "r") as f:
            dict_split = json.loads(f)
        self.image_paths = dict_split[data_split]
        self.image_size = image_size
        self.transform = transform

    def __len(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        images = np.array(Image.open(image).convert("RGB"))
        images = image_to_tensor(images)
        labels = torch.Tensor([0 if i.contains("Dog") else 1 for i in image])

        if self.transform:
            augmentations = self.transform(image=images)
            images = augmentations['image']
        
        return images, labels
        
class AnimalDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, image_size, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = nn.Sequential(
            #K.RandomHorizontalFlip(0.5),
            K.Normalize(torch.zeros(3), torch.ones(3)),
        )

        self.val_transform = nn.Sequential(
            K.Normalize(torch.zeros(3), torch.ones(3))
        )

    def train_dataloader(self):
        return DataLoader(DogCatDataset("training", data_dir=self.data_dir, image_size=self.image_size, transform=self.train_transform), 
        batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(DogCatDataset("validation", data_dir=self.data_dir, image_size=self.image_size, transform=self.val_transform), 
        batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(DogCatDataset("testing", data_dir=self.data_dir, image_size=self.image_size, transform=self.val_transform), 
        batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    data_module = AnimalDataModule(data_dir='data/processed', image_size=200, batch_size=4, num_workers=16)
    train_loader, val_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()
